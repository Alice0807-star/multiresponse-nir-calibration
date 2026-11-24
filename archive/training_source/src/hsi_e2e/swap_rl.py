"""Two-phase Double-DQN local rewriting; only complete K-subsets are scored.

This is a project adaptation inspired by complete-solution local rewriting,
not a reproduction of NeuRewriter's actor-critic algorithm.
"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .local_search import evaluate_pool, swap_subset
from .selector import _Transition, _learning_step, context_for


def normalize(values):
    values = np.asarray(values, dtype=np.float64)
    span = float(np.ptp(values))
    return (values - values.min()) / span if span > 1e-12 else np.zeros_like(values)


class SwapQNetwork(nn.Module):
    def __init__(self, state_dim, n_bands):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, n_bands))

    def forward(self, value):
        return self.layers(value)


class SwapContext:
    def __init__(self, evaluator, *, hint_type, correlation, A=None, R=None, horizon=16):
        self.evaluator, self.hint_type = evaluator, hint_type
        self.correlation = np.asarray(correlation)
        self.A, self.R, self.horizon = A, R, int(horizon)
        self.cache = {}
        self.hint_seconds = 0.

    def hint(self, subset):
        key = tuple(sorted(subset))
        if key in self.cache:
            return self.cache[key]
        started = time.perf_counter()
        mask = np.zeros(self.evaluator.n_bands, bool)
        mask[list(key)] = True
        add = drop = np.zeros(self.evaluator.n_bands)
        if self.hint_type == "static_correlation":
            add = normalize(self.correlation)
            drop = np.zeros_like(add)
            drop[mask] = 1. - normalize(self.correlation[mask])
        elif self.hint_type == "frozen_attention":
            add = context_for(mask, self.A, self.R, self.evaluator.k)["hint"].astype(float)
            importance = self.A.mean(axis=0)
            drop = np.zeros_like(add)
            drop[mask] = 1. - normalize(importance[mask])
        elif self.hint_type == "conditional_plsr":
            add_raw, importance = self.evaluator.conditional_hints(key)
            add = normalize(add_raw)
            drop = np.zeros_like(add)
            drop[mask] = 1. - normalize(importance[mask])
        elif self.hint_type != "none":
            raise ValueError(self.hint_type)
        self.cache[key] = (np.asarray(add, np.float32), np.asarray(drop, np.float32))
        self.hint_seconds += time.perf_counter() - started
        return self.cache[key]

    def state(self, subset, error, step, pending=None):
        B = self.evaluator.n_bands
        mask = np.zeros(B, bool)
        mask[list(subset)] = True
        marker = np.zeros(B, np.float32)
        if pending is not None:
            if not mask[pending]:
                raise ValueError("Pending removal must belong to the complete current subset")
            marker[pending] = 1.
        add, drop = self.hint(subset)
        state = np.concatenate([mask.astype(np.float32), marker, add, drop,
            np.array([error, float(pending is not None), (self.horizon - step) / self.horizon], np.float32)])
        # Removal: only selected columns. Addition: only columns outside the
        # original complete subset, so an identity replacement is impossible.
        invalid = mask.copy() if pending is not None else ~mask
        return state, invalid


def run_swap_dqn(evaluator, pool, *, seed, hint_type, correlation, A=None, R=None,
                 horizon=16, output_dir=None):
    started = time.perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 200000)
    B = evaluator.n_bands
    scored = evaluate_pool(evaluator, pool)
    best_subset, best_error = min(scored, key=lambda pair: pair[1])
    initial_best = best_error
    context = SwapContext(evaluator, hint_type=hint_type, correlation=correlation,
                          A=A, R=R, horizon=horizon)
    online, target = SwapQNetwork(4 * B + 3, B), SwapQNetwork(4 * B + 3, B)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=.001)
    replay = deque(maxlen=4000)
    zero_hint = np.zeros(B, np.float32)
    trace, episode_rows = [], []
    updates = transitions = swap_count = 0
    reserve = horizon
    train_limit = evaluator.fit_budget - reserve
    initial_fits = evaluator.fit_count
    if train_limit <= initial_fits:
        raise ValueError("Budget must cover initialization, training and policy rollout")
    max_attempts = 4 * (train_limit - initial_fits)

    def learn(state, invalid, action, reward, next_state, next_invalid, done):
        nonlocal updates, transitions
        replay.append(_Transition(state.copy(), zero_hint, action, float(reward),
                                  next_state.copy(), zero_hint, next_invalid.copy(), bool(done)))
        transitions += 1
        loss = None
        if len(replay) >= 32:
            indices = rng.choice(len(replay), 32, replace=False)
            loss = _learning_step(online, target, optimizer, [replay[int(i)] for i in indices],
                                  torch.device("cpu"), 0.)
            updates += 1
        if transitions % 64 == 0:
            target.load_state_dict(online.state_dict())
        return loss

    def action(state, invalid, epsilon):
        available = np.flatnonzero(~invalid)
        if rng.random() < epsilon:
            return int(rng.choice(available))
        with torch.no_grad():
            values = online(torch.from_numpy(state)[None]).squeeze(0)
            values = values.masked_fill(torch.from_numpy(invalid), -torch.inf)
            return int(values.argmax())

    for index, (subset, score) in enumerate(scored):
        trace.append(dict(phase="initial_pool", query=index + 1, fit_count=index + 1,
            subset_json=json.dumps(subset), validation_nmse=score,
            incumbent_nmse=min(v for _, v in scored[:index + 1]),
            elapsed_seconds=time.perf_counter() - started))
    episode = 0
    while evaluator.fit_count < train_limit and swap_count < max_attempts:
        current, current_error = scored[episode % len(scored)]
        start_error, episode_return = current_error, 0.
        for step in range(horizon):
            progress = (evaluator.fit_count - initial_fits) / max(train_limit - initial_fits - 1, 1)
            epsilon = max(.1, 1. - .9 * progress)
            state, invalid = context.state(current, current_error, step)
            remove = action(state, invalid, epsilon)
            add_state, add_invalid = context.state(current, current_error, step, pending=remove)
            remove_loss = learn(state, invalid, remove, 0., add_state, add_invalid, False)
            add = action(add_state, add_invalid, epsilon)
            next_subset = swap_subset(current, remove, add)
            before_fits = evaluator.fit_count
            next_error = float(evaluator.evaluate(next_subset))
            reward = current_error - next_error
            swap_count += 1
            done = (step + 1 == horizon or evaluator.fit_count >= train_limit or swap_count >= max_attempts)
            next_state, next_invalid = context.state(next_subset, next_error, min(step + 1, horizon))
            add_loss = learn(add_state, add_invalid, add, reward, next_state, next_invalid, done)
            if next_error < best_error:
                best_subset, best_error = next_subset, next_error
            trace.append(dict(phase="training", episode=episode + 1, step=step + 1,
                current_subset_json=json.dumps(current), subset_json=json.dumps(next_subset),
                removed_1based=remove + 1, added_1based=add + 1,
                before_nmse=current_error, validation_nmse=next_error, reward=reward,
                incumbent_nmse=best_error, epsilon=epsilon, done=done,
                cache_hit=evaluator.fit_count == before_fits, fit_count=evaluator.fit_count,
                remove_loss=remove_loss, add_loss=add_loss,
                elapsed_seconds=time.perf_counter() - started))
            episode_return += reward
            current, current_error = next_subset, next_error
            if done:
                break
        if not np.isclose(episode_return, start_error - current_error, atol=1e-10):
            raise AssertionError("Swap rewards failed to telescope")
        episode += 1
        episode_rows.append(dict(episode=episode, return_=episode_return,
            start_nmse=start_error, final_nmse=current_error, fits=evaluator.fit_count))
        if episode % 25 == 0:
            print(f"  swap-DQN {hint_type}: episodes={episode}, fits={evaluator.fit_count}, best={best_error:.6f}", flush=True)

    # Frozen final policy gets a reserved horizon starting from the common
    # initialization pool's best member; no additional learning is performed.
    online.eval()
    current, current_error = min(scored, key=lambda pair: pair[1])
    policy_start_error = current_error
    for step in range(horizon):
        state, invalid = context.state(current, current_error, step)
        remove = action(state, invalid, 0.)
        add_state, add_invalid = context.state(current, current_error, step, pending=remove)
        add = action(add_state, add_invalid, 0.)
        next_subset = swap_subset(current, remove, add)
        before_fits = evaluator.fit_count
        next_error = float(evaluator.evaluate(next_subset))
        if next_error < best_error:
            best_subset, best_error = next_subset, next_error
        trace.append(dict(phase="final_policy", episode=episode + 1, step=step + 1,
            current_subset_json=json.dumps(current), subset_json=json.dumps(next_subset),
            removed_1based=remove + 1, added_1based=add + 1,
            before_nmse=current_error, validation_nmse=next_error, reward=current_error - next_error,
            incumbent_nmse=best_error, epsilon=0., done=step + 1 == horizon,
            cache_hit=evaluator.fit_count == before_fits, fit_count=evaluator.fit_count,
            elapsed_seconds=time.perf_counter() - started))
        current, current_error = next_subset, next_error
    seconds = time.perf_counter() - started
    result = dict(best_subset=list(best_subset), best_validation_nmse=best_error,
        initial_best_nmse=initial_best, final_subset=list(current), final_policy_nmse=current_error,
        policy_start_nmse=policy_start_error, search_seconds=seconds,
        hint_seconds=context.hint_seconds, evaluator_counters=evaluator.snapshot(),
        training_swaps=swap_count, training_transitions=transitions, gradient_updates=updates,
        episodes=episode, horizon=horizon, training_fit_limit=train_limit,
        max_training_swap_attempts=max_attempts, stopped_on_attempt_limit=swap_count >= max_attempts,
        hint_type=hint_type, state_dim=4 * B + 3,
        configuration=dict(hidden=64, learning_rate=.001, batch_size=32, replay_capacity=4000,
            gamma=1., target_update_transitions=64, q_prior=0.,
            epsilon="1 to 0.1 by training real-fit progress", two_phase=True,
            remove_reward=0., add_reward="before_NMSE - after_NMSE",
            final_policy_reserved_queries=horizon, resume_supported=False),
        episode_summaries=episode_rows)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / "swap_dqn_checkpoint.pt"
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in online.state_dict().items()},
                        state_dim=4 * B + 3, n_bands=B, seed=seed,
                        configuration=result["configuration"], hint_type=hint_type), checkpoint)
        result["checkpoint_path"] = str(checkpoint.resolve())
    result["trace"] = trace
    return result
