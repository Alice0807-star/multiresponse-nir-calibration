"""Uniform mean-R2 improvement rewards with the existing NMSE state unchanged.

The legacy trainer still computes its NMSE bookkeeping internally. This adapter
replaces the reward delivered to replay and exported traces, and recomputes
frozen-policy calibration in the correct R2 units. No network/training rule is
changed. Rewards are differences, not absolute per-step R2 or clipped R2.
"""
from __future__ import annotations
import json
import numpy as np
import torch
from .shared_action import AttentionArchitectureTrainer


def r2_cost(evaluator, subset):
    """1 - uniform-average R2; only already-paid predictions may be consumed."""
    key = tuple(sorted(subset))
    if not hasattr(evaluator, '_r2_costs'):
        variance = np.var(evaluator.y_reward_z,axis=0,ddof=0)
        if not np.isfinite(variance).all() or np.any(variance<=1e-12):
            raise ValueError('R2 reward requires nonconstant validation targets')
        evaluator._r2_variance = variance
        evaluator._r2_costs = {}
    if key not in evaluator._r2_costs:
        prediction = evaluator._prediction_cache[key]  # No regression query here.
        mse = np.mean((evaluator.y_reward_z-prediction)**2,axis=0)
        evaluator._r2_costs[key] = float(np.mean(mse/evaluator._r2_variance))
    return evaluator._r2_costs[key]


class R2RewardTrainer(AttentionArchitectureTrainer):
    def _learn(self, state, action, reward, next_state, next_invalid, done):
        if state[-2]:
            B = self.ev.n_bands
            before,after = np.flatnonzero(state[:B]),np.flatnonzero(next_state[:B])
            reward = r2_cost(self.ev,before)-r2_cost(self.ev,after)
        else:
            reward = 0.
        super()._learn(state,action,reward,next_state,next_invalid,done)

    def advance(self, fit_target):
        first = len(self.training_trace)
        status = super().advance(fit_target)
        for row in self.training_trace[first:]:
            before = json.loads(row['current_subset_json'])
            after = json.loads(row['subset_json'])
            row['nmse_improvement'] = row['reward']
            row['reward'] = r2_cost(self.ev,before)-r2_cost(self.ev,after)
            row['reward_kind'] = 'uniform_mean_R2_improvement'
        return status

    def probe(self, checkpoint):
        first = len(self.probe_trace)
        summary = super().probe(checkpoint)
        rows = self.probe_trace[first:]
        for panel in summary['panels']:
            chosen = [r for r in rows if r['pool_index']==panel['pool_index']]
            start = json.loads(chosen[0]['current_subset_json'])
            end = json.loads(chosen[-1]['subset_json'])
            actual = r2_cost(self.pev,start)-r2_cost(self.pev,end)
            panel['nmse_return'] = panel['actual_return']
            panel['actual_return'] = actual
            panel['calibration_error'] = panel['predicted_initial_Q']-actual
            panel['start_mean_r2'] = 1.-r2_cost(self.pev,start)
            panel['end_mean_r2'] = 1.-r2_cost(self.pev,end)
            violation = 0.
            for step,row in enumerate(chosen):
                before = json.loads(row['current_subset_json'])
                after = json.loads(row['subset_json'])
                bound = r2_cost(self.pev,before)
                row['nmse_improvement'] = row['reward']
                row['reward'] = bound-r2_cost(self.pev,after)
                for pending in [None,int(row['removed_1based'])-1]:
                    state,mask = self.probe_context.state(before,row['before_nmse'],step,pending=pending)
                    with torch.no_grad():
                        q = self.online(torch.from_numpy(state)[None]).squeeze(0).numpy()
                    violation = max(violation,float(np.maximum(q[~mask]-bound,0.).max()))
            panel['max_valid_action_Q_upper_violation'] = violation
        summary['greedy_mean_return'] = float(np.mean([r['actual_return'] for r in summary['panels']]))
        summary['greedy_mean_end_r2'] = float(np.mean([r['end_mean_r2'] for r in summary['panels']]))
        summary['Q_calibration_mae'] = float(np.mean([abs(r['calibration_error']) for r in summary['panels']]))
        summary['max_Q_upper_violation'] = max(r['max_valid_action_Q_upper_violation'] for r in summary['panels'])
        summary['reward_kind'] = 'uniform_mean_R2_improvement'
        return summary
