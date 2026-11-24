"""Audited finite-horizon swap critics and RNG-isolated frozen-policy probes."""
from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .local_search import evaluate_pool, swap_subset
from .selector import _Transition
from .swap_rl import SwapContext, SwapQNetwork


ARMS = {
    "td_lr1e3": dict(value_mode="return", learning_rate=.001, bound_bootstrap=False),
    "td_lr1e4": dict(value_mode="return", learning_rate=.0001, bound_bootstrap=False),
    "td_bound": dict(value_mode="return", learning_rate=.001, bound_bootstrap=True),
    "terminal_cost": dict(value_mode="terminal_cost", learning_rate=.001, bound_bootstrap=False),
}


def values_for_actions(raw, states, value_mode):
    """All policies maximize expected NMSE improvement, with gamma=1."""
    if value_mode == "return":
        return raw
    if value_mode == "terminal_cost":
        return states[:, -3, None] - F.softplus(raw)
    raise ValueError(value_mode)


def compute_targets(rewards, done, next_error, next_values, *, value_mode, bound_bootstrap):
    """next_values are target raw Q, or positive target terminal cost.

    Only real horizon termination sets done=True. A sampling-budget cut is not
    an MDP terminal. Terminal cost learning is algebraically the same objective:
    Q(s,a)=E(s)-C(s,a); C_terminal=E(next), C_nonterminal=C(next,a*).
    """
    if value_mode == "terminal_cost":
        return torch.where(done, next_error, next_values)
    bootstrap = next_values
    if bound_bootstrap:
        bootstrap = torch.minimum(bootstrap, next_error)
    return rewards + torch.where(done, torch.zeros_like(bootstrap), bootstrap)


def learning_step(online, target, optimizer, transitions, config):
    states = torch.from_numpy(np.stack([t.state for t in transitions]))
    actions = torch.tensor([t.action for t in transitions], dtype=torch.long)
    rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
    next_states = torch.from_numpy(np.stack([t.next_state for t in transitions]))
    masks = torch.from_numpy(np.stack([t.next_mask for t in transitions])).bool()
    done = torch.tensor([t.done for t in transitions], dtype=torch.bool)
    mode = config["value_mode"]
    raw = online(states)
    chosen_q = values_for_actions(raw, states, mode).gather(1, actions[:, None]).squeeze(1)
    prediction = (F.softplus(raw) if mode == "terminal_cost" else raw).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_value = torch.zeros_like(rewards)
        active = ~done
        if active.any():
            if masks[active].all(1).any():
                raise ValueError("Nonterminal state has no valid action")
            next_raw = online(next_states[active])
            next_q = values_for_actions(next_raw, next_states[active], mode).masked_fill(masks[active], -torch.inf)
            next_actions = next_q.argmax(1)
            target_raw = target(next_states[active])
            if mode == "terminal_cost":
                target_raw = F.softplus(target_raw)
            next_value[active] = target_raw.gather(1, next_actions[:, None]).squeeze(1)
        targets = compute_targets(rewards, done, next_states[:, -3], next_value,
                                  value_mode=mode, bound_bootstrap=config["bound_bootstrap"])
        target_q = states[:, -3] - targets if mode == "terminal_cost" else targets
    loss = F.smooth_l1_loss(prediction, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite TD loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(online.parameters(), 5.)
    optimizer.step()
    return dict(loss=float(loss.detach()), gradient_norm_before_clip=float(grad_norm),
        sampled_q_abs_max=float(chosen_q.detach().abs().max()),
        target_q_abs_max=float(target_q.abs().max()),
        sampled_q_upper_violation=float((chosen_q.detach() - states[:, -3]).clamp_min(0).max()),
        target_q_upper_violation=float((target_q - states[:, -3]).clamp_min(0).max()),
        terminal_fraction=float(done.float().mean()))


class SwapTrainer:
    """Keep full process state to allow one predeclared low-epsilon continuation."""
    def __init__(self, training_evaluator, probe_evaluator, pool, *, seed, arm, output_dir,
                 short_budget=3200, horizon=16):
        if arm not in ARMS:
            raise ValueError(arm)
        self.started = time.perf_counter()
        self.ev, self.pev, self.pool = training_evaluator, probe_evaluator, pool
        self.seed, self.arm = seed, arm
        self.config = dict(ARMS[arm])
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=False)
        self.short_budget, self.horizon = short_budget, horizon
        self.scored = evaluate_pool(self.ev, pool)
        self.initial_fits = self.ev.fit_count
        self.best_train, self.best_train_error = min(self.scored, key=lambda item:item[1])
        self.best_probe, self.best_probe_error = None, float("inf")
        self.context = SwapContext(self.ev, hint_type="none", correlation=np.zeros(self.ev.n_bands), horizon=horizon)
        self.probe_context = SwapContext(self.pev, hint_type="none", correlation=np.zeros(self.ev.n_bands), horizon=horizon)
        torch.manual_seed(seed)
        self.online = SwapQNetwork(4*self.ev.n_bands+3, self.ev.n_bands)
        self.target = SwapQNetwork(4*self.ev.n_bands+3, self.ev.n_bands)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.initial_state_hash = hashlib.sha256(b"".join(t.detach().numpy().tobytes() for t in self.online.state_dict().values())).hexdigest()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config["learning_rate"])
        self.rng = np.random.default_rng(seed+200000)
        self.replay = deque(maxlen=4000)
        self.zero_hint = np.zeros(self.ev.n_bands, np.float32)
        self.episode = self.step = self.swaps = self.transitions = self.updates = 0
        self.current, self.current_error = self.scored[0]
        self.episode_start_error = self.current_error
        self.episode_return = 0.
        self.training_trace, self.td_trace, self.probe_trace, self.probe_summary = [], [], [], []

    def _select(self, state, invalid, epsilon=None):
        # Probes pass None and never touch a training random-number stream.
        if epsilon is not None and self.rng.random() < epsilon:
            return int(self.rng.choice(np.flatnonzero(~invalid)))
        with torch.no_grad():
            batch = torch.from_numpy(state)[None]
            q = values_for_actions(self.online(batch), batch, self.config["value_mode"]).squeeze(0)
            q = q.masked_fill(torch.from_numpy(invalid), -torch.inf)
            return int(q.argmax())

    def _learn(self, state, action, reward, next_state, next_invalid, done):
        self.replay.append(_Transition(state.copy(), self.zero_hint, action, reward,
                                       next_state.copy(), self.zero_hint, next_invalid.copy(), done))
        self.transitions += 1
        if len(self.replay) >= 32:
            indices = self.rng.choice(len(self.replay),32,replace=False)
            stats = learning_step(self.online,self.target,self.optimizer,[self.replay[int(i)] for i in indices],self.config)
            self.updates += 1
            self.td_trace.append(dict(transition=self.transitions, training_fit_count=self.ev.fit_count,
                                      phase="add" if state[-2] else "remove", **stats))
        if self.transitions % 64 == 0:
            self.target.load_state_dict(self.online.state_dict())

    def advance(self, fit_target):
        attempt_limit = 4*(fit_target-self.initial_fits)
        while self.ev.fit_count < fit_target and self.swaps < attempt_limit:
            progress = min(1.,(self.ev.fit_count-self.initial_fits)/max(self.short_budget-self.initial_fits-1,1))
            epsilon = max(.1,1.-.9*progress)
            state, invalid = self.context.state(self.current,self.current_error,self.step)
            remove = self._select(state,invalid,epsilon)
            add_state, add_invalid = self.context.state(self.current,self.current_error,self.step,pending=remove)
            self._learn(state,remove,0.,add_state,add_invalid,False)
            add = self._select(add_state,add_invalid,epsilon)
            after = swap_subset(self.current,remove,add)
            before_fits = self.ev.fit_count
            error = float(self.ev.evaluate(after))
            reward = self.current_error-error
            self.swaps += 1
            # The MDP has a 16-swap horizon, independent of collection budget.
            done = self.step+1 == self.horizon
            next_state,next_invalid = self.context.state(after,error,self.step+1)
            self._learn(add_state,add,reward,next_state,next_invalid,done)
            if error < self.best_train_error:
                self.best_train,self.best_train_error = after,error
            self.training_trace.append(dict(episode=self.episode+1,step=self.step+1,
                subset_json=json.dumps(after),current_subset_json=json.dumps(self.current),
                removed_1based=remove+1,added_1based=add+1,before_nmse=self.current_error,
                validation_nmse=error,reward=reward,done=done,epsilon=epsilon,
                training_fit_count=self.ev.fit_count,cache_hit=self.ev.fit_count==before_fits,
                elapsed_seconds=time.perf_counter()-self.started))
            self.episode_return += reward
            self.current,self.current_error = after,error
            self.step += 1
            if done:
                if not np.isclose(self.episode_return,self.episode_start_error-error,atol=1e-9):
                    raise AssertionError("Episode rewards failed to telescope")
                self.episode += 1
                self.step = 0
                self.current,self.current_error = self.scored[self.episode % len(self.scored)]
                self.episode_start_error = self.current_error
                self.episode_return = 0.
        return dict(reached_target=self.ev.fit_count==fit_target,training_fit_count=self.ev.fit_count,
                    attempt_limit_hit=self.swaps>=attempt_limit)

    def probe(self, checkpoint):
        rng_before = json.dumps(self.rng.bit_generator.state,sort_keys=True)
        torch_rng = torch.get_rng_state().clone()
        was_training = self.online.training
        self.online.eval()
        rows = []
        try:
            for index in [0,4,8,12]:
                current = tuple(self.pool[index])
                error = float(self.pev.evaluate(current))
                start_error = error
                states_seen = []
                q_first = None
                violation = 0.
                for step in range(self.horizon):
                    state,invalid = self.probe_context.state(current,error,step)
                    with torch.no_grad():
                        batch = torch.from_numpy(state)[None]
                        q = values_for_actions(self.online(batch),batch,self.config["value_mode"]).squeeze(0).numpy()
                    remove = self._select(state,invalid)
                    if q_first is None:
                        q_first = float(q[remove])
                    violation = max(violation,float(np.maximum(q[~invalid]-error,0).max()))
                    add_state,add_invalid = self.probe_context.state(current,error,step,pending=remove)
                    with torch.no_grad():
                        batch = torch.from_numpy(add_state)[None]
                        q_add = values_for_actions(self.online(batch),batch,self.config["value_mode"]).squeeze(0).numpy()
                    violation = max(violation,float(np.maximum(q_add[~add_invalid]-error,0).max()))
                    add = self._select(add_state,add_invalid)
                    after = swap_subset(current,remove,add)
                    next_error = float(self.pev.evaluate(after))
                    if next_error < self.best_probe_error:
                        self.best_probe,self.best_probe_error = after,next_error
                    self.probe_trace.append(dict(checkpoint=checkpoint,pool_index=index,step=step+1,
                        current_subset_json=json.dumps(current),subset_json=json.dumps(after),
                        before_nmse=error,validation_nmse=next_error,reward=error-next_error,
                        removed_1based=remove+1,added_1based=add+1,
                        probe_fit_count=self.pev.fit_count))
                    states_seen.append(after)
                    current,error = after,next_error
                actual_return = start_error-error
                rows.append(dict(pool_index=index,start_nmse=start_error,end_nmse=error,
                    actual_return=actual_return,predicted_initial_Q=q_first,
                    calibration_error=q_first-actual_return,
                    max_valid_action_Q_upper_violation=violation,
                    unique_subsets=len(set(states_seen)),final_subset=list(current)))
        finally:
            self.online.train(was_training)
        if rng_before != json.dumps(self.rng.bit_generator.state,sort_keys=True) or not torch.equal(torch_rng,torch.get_rng_state()):
            raise AssertionError("Policy probes consumed training RNG")
        recent = self.td_trace[-max(1,len(self.td_trace)//4):]
        preceding = self.td_trace[-max(2,len(self.td_trace)//2):-max(1,len(self.td_trace)//4)]
        row = dict(checkpoint=checkpoint,training_fit_count=self.ev.fit_count,
            probe_fit_count=self.pev.fit_count,total_actual_fits=self.ev.fit_count+self.pev.fit_count,
            greedy_mean_end_nmse=float(np.mean([r["end_nmse"] for r in rows])),
            greedy_mean_return=float(np.mean([r["actual_return"] for r in rows])),
            Q_calibration_mae=float(np.mean([abs(r["calibration_error"]) for r in rows])),
            max_Q_upper_violation=max(r["max_valid_action_Q_upper_violation"] for r in rows),
            mean_unique_subsets=float(np.mean([r["unique_subsets"] for r in rows])),
            median_recent_td_loss=float(np.median([r["loss"] for r in recent])) if recent else None,
            median_preceding_td_loss=float(np.median([r["loss"] for r in preceding])) if preceding else None,
            gradient_updates=self.updates,epsilon=(max(.1,1.-.9*min(1.,(self.ev.fit_count-self.initial_fits)/max(self.short_budget-self.initial_fits-1,1)))),
            best_search_nmse=min(self.best_train_error,self.best_probe_error),
            probe_rng_unchanged=True,panels=rows)
        self.probe_summary.append(row)
        checkpoint_file = self.out/f"checkpoint_{checkpoint}.pt"
        torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in self.online.state_dict().items()},
                        arm=self.arm,seed=self.seed,config=self.config,state_dim=4*self.ev.n_bands+3,
                        n_bands=self.ev.n_bands,checkpoint=checkpoint),checkpoint_file)
        return row

    def best_result(self):
        if self.best_train_error <= self.best_probe_error:
            subset,ev = self.best_train,self.ev
        else:
            subset,ev = self.best_probe,self.pev
        return list(subset),ev.reward_metrics(subset)
