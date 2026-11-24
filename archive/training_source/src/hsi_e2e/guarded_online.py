"""Online observation/stop adapter for the archived mean-R2 mixed trainers.

The optimizer, original Double-DQN targets, sampling and restart rules are not
modified. Quality stops are engineering decisions about approximate values,
not evidence of an invalid PLSR score or a proof of convergence/divergence.
Legacy source files are intentionally left unchanged.
"""
from __future__ import annotations
import copy
import math
import numpy as np
import torch
from torch.nn import functional as F
from .selector import _Transition
from .stability import values_for_actions,compute_targets
from .mixed_restart import MixedRestartTrainer
from .leaky_state import LeakyStateTrainer
from .r2_reward import r2_cost


DEFAULT_GUARD_RULES = dict(
    enabled=True,
    q_action='stop',
    warmup_updates=128,
    persistent_limit=.25,
    persistent_updates=8,
    immediate_limit=1.,
    interpretation='finite-window value-quality protection; not a convergence proof',
)


class NonfiniteTrainingError(FloatingPointError):
    """A numeric failure, always fatal irrespective of value-stop settings."""
    def __init__(self,message,details=None):
        super().__init__(message);self.details=details or {}


class ValueQualityStop(RuntimeError):
    """A predeclared approximate-value quality threshold was reached."""
    def __init__(self,message,details):
        super().__init__(message);self.details=details


class OnlineValueGuard:
    """Serializable rules and counters; observes but never clips a raw value."""
    version=1
    def __init__(self,rules=None):
        self.rules=copy.deepcopy(DEFAULT_GUARD_RULES)
        if rules is not None:
            unknown=set(rules)-set(self.rules)
            if unknown:raise ValueError(f'Unknown guard rules: {sorted(unknown)}')
            self.rules.update(copy.deepcopy(rules))
        r=self.rules
        if not isinstance(r['enabled'],bool) or r['q_action'] not in {'stop','record'}:
            raise ValueError('Invalid guard enabled/action')
        for key in ['warmup_updates','persistent_updates']:
            if isinstance(r[key],bool) or int(r[key])!=r[key] or r[key]<0:
                raise ValueError(f'Invalid guard {key}')
        if r['persistent_updates']<1:raise ValueError('persistent_updates must be positive')
        for key in ['persistent_limit','immediate_limit']:
            if not np.isfinite(r[key]) or r[key]<0.:raise ValueError(f'Invalid guard {key}')
        if r['immediate_limit']<r['persistent_limit']:
            raise ValueError('Immediate limit cannot be below persistent limit')
        self.streak=0;self.last_update=0;self.last_decision=None

    def observe(self,metrics,update_index):
        update_index=int(update_index)
        if update_index<=self.last_update:
            raise ValueError('Guard update indices must strictly increase')
        self.last_update=update_index
        numeric=[v for v in metrics.values() if isinstance(v,(int,float,np.number))]
        if not all(np.isfinite(v) for v in numeric):
            self.last_decision=dict(kind='numeric_failure',update=update_index,reason='nonfinite batch statistics')
            raise NonfiniteTrainingError('Nonfinite batch statistics',self.last_decision)
        r=self.rules
        violation=max(metrics['sampled_q_upper_violation'],metrics['target_q_upper_violation'],
                      metrics.get('max_legal_q_upper_violation',0.))
        decision=dict(kind='normal',update=update_index,violation=float(violation),
                      reason=None,convergence_proven=False)
        if not r['enabled']:
            self.streak=0;decision['kind']='value_stop_disabled'
        else:
            self.streak=(self.streak+1 if update_index>r['warmup_updates'] and
                         violation>r['persistent_limit'] else 0)
            if violation>r['immediate_limit']:
                decision.update(kind='value_quality_stop',reason='immediate severe R2-return bound violation')
            elif self.streak>=r['persistent_updates']:
                decision.update(kind='value_quality_stop',reason='persistent R2-return bound violation')
        decision['consecutive_batches']=self.streak
        if decision['kind']=='value_quality_stop' and r['q_action']=='record':
            decision['kind']='value_quality_warning'
        self.last_decision=copy.deepcopy(decision)
        if decision['kind']=='value_quality_stop':
            raise ValueQualityStop(decision['reason'],copy.deepcopy(decision))
        return decision


def _finite(value,name):
    if not torch.isfinite(value).all():
        raise NonfiniteTrainingError(f'Nonfinite {name}',dict(stage=name))


def _paid_r2_bounds(evaluator,states):
    """Read only cached predictions. There is no evaluator.evaluate call here."""
    B=evaluator.n_bands;costs=[]
    for state in states:
        subset=tuple(np.flatnonzero(state[:B]))
        if len(subset)!=evaluator.k or subset not in evaluator._cache:
            raise ValueError('R2 guard requires an already-paid complete state subset')
        if evaluator._prediction_cache is None or subset not in evaluator._prediction_cache:
            raise ValueError('R2 guard requires retained already-paid predictions')
        cost=r2_cost(evaluator,subset)
        if not np.isfinite(cost) or cost<0.:
            raise NonfiniteTrainingError('Invalid true-R2 return bound')
        costs.append(cost)
    return torch.tensor(costs,dtype=torch.float32)


def guarded_online_learning_step(online,target,optimizer,transitions,config,*,
                                 evaluator,guard,update_index):
    """The original return-mode update with pre-step safety observations.

    Only the audited R2 return mode without bootstrap clipping is supported.
    A caller must freeze any new learning-rule variant as a separate method.
    """
    if config['value_mode']!='return' or config['bound_bootstrap']:
        raise ValueError('Guarded R2 adapter supports original unclipped return targets only')
    states=torch.from_numpy(np.stack([t.state for t in transitions]))
    actions=torch.tensor([t.action for t in transitions],dtype=torch.long)
    rewards=torch.tensor([t.reward for t in transitions],dtype=torch.float32)
    next_states=torch.from_numpy(np.stack([t.next_state for t in transitions]))
    masks=torch.from_numpy(np.stack([t.next_mask for t in transitions])).bool()
    done=torch.tensor([t.done for t in transitions],dtype=torch.bool)
    _finite(states,'states');_finite(rewards,'rewards');_finite(next_states,'next states')
    mode=config['value_mode']
    raw=online(states);_finite(raw,'online Q')
    chosen_q=values_for_actions(raw,states,mode).gather(1,actions[:,None]).squeeze(1)
    prediction=raw.gather(1,actions[:,None]).squeeze(1)
    with torch.no_grad():
        next_value=torch.zeros_like(rewards);active=~done
        if active.any():
            if masks[active].all(1).any():raise ValueError('Nonterminal state has no valid action')
            next_raw=online(next_states[active]);_finite(next_raw,'next online Q')
            next_q=values_for_actions(next_raw,next_states[active],mode).masked_fill(masks[active],-torch.inf)
            next_actions=next_q.argmax(1)
            target_raw=target(next_states[active]);_finite(target_raw,'target Q')
            next_value[active]=target_raw.gather(1,next_actions[:,None]).squeeze(1)
        # Kept exactly equal to the archived unbounded Double-DQN target.
        targets=compute_targets(rewards,done,next_states[:,-3],next_value,
                                value_mode=mode,bound_bootstrap=False)
    _finite(targets,'TD target')
    loss=F.smooth_l1_loss(prediction,targets);_finite(loss,'TD loss')
    bounds=_paid_r2_bounds(evaluator,states.numpy())
    q_violation=(chosen_q.detach()-bounds).clamp_min(0)
    target_violation=(targets-bounds).clamp_min(0)
    selected_mask=states[:,:evaluator.n_bands].bool()
    valid=torch.where(states[:,-2,None].bool(),~selected_mask,selected_mask)
    if not valid.any(1).all():raise ValueError('Current state has no valid action')
    if not valid.gather(1,actions[:,None]).all():raise ValueError('Sampled current action is invalid')
    valid_violation=(raw.detach()-bounds[:,None]).clamp_min(0).masked_fill(~valid,0.)
    stats=dict(loss=float(loss.detach()),sampled_q_abs_max=float(chosen_q.detach().abs().max()),
        target_q_abs_max=float(targets.abs().max()),
        sampled_q_upper_violation=float(q_violation.max()),
        target_q_upper_violation=float(target_violation.max()),
        max_legal_q_upper_violation=float(valid_violation.max()),
        max_legal_q_abs_max=float(raw.detach().abs().masked_fill(~valid,0.).max()),
        sampled_q_upper_violation_nmse_reference=float((chosen_q.detach()-states[:,-3]).clamp_min(0).max()),
        target_q_upper_violation_nmse_reference=float((targets-states[:,-3]).clamp_min(0).max()),
        r2_bound_min=float(bounds.min()),r2_bound_max=float(bounds.max()),
        worst_sampled_raw_q=float(chosen_q.detach()[q_violation.argmax()]),
        worst_sampled_true_r2_bound=float(bounds[q_violation.argmax()]),
        worst_target_raw_q=float(targets[target_violation.argmax()]),
        worst_target_true_r2_bound=float(bounds[target_violation.argmax()]),
        terminal_fraction=float(done.float().mean()))
    try:
        decision=guard.observe(stats,update_index)
    except (ValueQualityStop,NonfiniteTrainingError) as exc:
        exc.details.update(batch_statistics=copy.deepcopy(stats),optimizer_step_performed=False)
        raise
    optimizer.zero_grad(set_to_none=True);loss.backward()
    for param in online.parameters():
        if param.grad is not None:_finite(param.grad,'gradient')
    grad_norm=torch.nn.utils.clip_grad_norm_(online.parameters(),5.)
    _finite(grad_norm,'gradient norm')
    optimizer.step()
    stats.update(gradient_norm_before_clip=float(grad_norm),value_guard_kind=decision['kind'],
        value_guard_streak=guard.streak,r2_bound_units='uniform_mean_R2_increment')
    return stats


class _GuardedOnlineMixin:
    def __init__(self,*args,value_guard_rules=None,**kwargs):
        super().__init__(*args,**kwargs)
        if self.config['value_mode']!='return' or self.config['bound_bootstrap']:
            raise ValueError('Unsupported learning-rule variant')
        self.value_guard=OnlineValueGuard(value_guard_rules)
        self.guard_events=[]
        self.guard_runtime_stopped=False
        self.config.update(reward_kind='uniform_mean_R2_improvement',value_guard_version=1,
                           value_guard_rules=copy.deepcopy(self.value_guard.rules))

    def _learn(self,state,action,reward,next_state,next_invalid,done):
        # R2RewardTrainer's actual replay reward adapter, unchanged.
        if state[-2]:
            B=self.ev.n_bands
            reward=r2_cost(self.ev,np.flatnonzero(state[:B]))-r2_cost(self.ev,np.flatnonzero(next_state[:B]))
        else:reward=0.
        self.replay.append(_Transition(state.copy(),self.zero_hint,action,float(reward),
            next_state.copy(),self.zero_hint,next_invalid.copy(),bool(done)))
        self.transitions+=1
        if len(self.replay)>=32:
            indices=self.rng.choice(len(self.replay),32,replace=False)
            if hasattr(self,'_pending_state_gradient_squares'):self._pending_state_gradient_squares={}
            try:
                stats=guarded_online_learning_step(self.online,self.target,self.optimizer,
                    [self.replay[int(i)] for i in indices],self.config,evaluator=self.ev,
                    guard=self.value_guard,update_index=self.updates+1)
            except (ValueQualityStop,NonfiniteTrainingError) as exc:
                self.guard_runtime_stopped=True
                self.guard_events.append(dict(transition=self.transitions,training_fit_count=self.ev.fit_count,
                    phase='add' if state[-2] else 'remove',kind=type(exc).__name__,details=copy.deepcopy(exc.details)))
                raise
            self.updates+=1
            self.td_trace.append(dict(transition=self.transitions,training_fit_count=self.ev.fit_count,
                phase='add' if state[-2] else 'remove',**stats))
            if stats['value_guard_kind']=='value_quality_warning':
                self.guard_events.append(dict(transition=self.transitions,training_fit_count=self.ev.fit_count,
                    phase='add' if state[-2] else 'remove',kind='value_quality_warning',
                    details=copy.deepcopy(self.value_guard.last_decision)))
            # Preserve the existing Leaky trainer's passive gradient audit.
            if hasattr(self,'state_gradient_rows'):
                if set(self._pending_state_gradient_squares)!={'weight','bias'}:
                    raise AssertionError('Missing state-gradient observation')
                norm=math.sqrt(sum(self._pending_state_gradient_squares.values()))
                self.state_gradient_rows.append(dict(update=self.updates,transition=self.transitions,
                    training_fit_count=self.ev.fit_count,phase='add' if state[-2] else 'remove',
                    state_gradient_norm_before_clip=norm))
                if len(self.state_gradient_rows)>=256 and all(r['state_gradient_norm_before_clip']==0.
                        for r in self.state_gradient_rows[-256:]):
                    self.guard_runtime_stopped=True
                    raise ValueQualityStop('State branch has zero gradients for256 consecutive updates',
                        dict(optimizer_step_performed=True,convergence_proven=False))
        if self.transitions%64==0:self.target.load_state_dict(self.online.state_dict())
        if done:self._schedule_restart(self.episode+1)


class GuardedMixedRestartTrainer(_GuardedOnlineMixin,MixedRestartTrainer):
    """Original ReLU mixed-R2 method plus passive pre-step protection."""


class GuardedLeakyStateTrainer(_GuardedOnlineMixin,LeakyStateTrainer):
    """Previously approved Leaky variant with the same protection adapter."""


def restore_guarded_runtime(trainer):
    """Call after a strict loader restores models/optimizer/attributes."""
    if not isinstance(trainer,(GuardedMixedRestartTrainer,GuardedLeakyStateTrainer)):
        raise ValueError('Unexpected guarded trainer type')
    if not isinstance(trainer.value_guard,OnlineValueGuard):raise ValueError('Missing guard state')
    if trainer.config.get('value_guard_version')!=1 or trainer.config.get('value_guard_rules')!=trainer.value_guard.rules:
        raise ValueError('Guard configuration mismatch')
    if trainer.guard_runtime_stopped:
        raise ValueError('Stopped guarded run requires explicit review, not silent continuation')
    if trainer.value_guard.last_update!=trainer.updates:
        raise ValueError('Guard and optimizer clocks disagree; stopped snapshots cannot silently resume')
    if isinstance(trainer,GuardedLeakyStateTrainer):
        trainer._pending_state_gradient_squares={}
        hooks=[bool(param._backward_hooks) for param in trainer.online.state_encoder[0].parameters()]
        if any(hooks) and not all(hooks):raise ValueError('Partial state-gradient hooks')
        if not any(hooks):trainer._attach_state_gradient_hooks()
    return trainer
