"""Predeclared engineering plateau and failure rules, not convergence proofs."""
import numpy as np

RULES = dict(minimum_plateau_checkpoint=6400,window=4,window_start_at_least=3200,
    endpoint_r2_range=0.001,best_queried_r2_gain=0.0005,
    calibration_mae_range=0.002,minimum_action_agreement=0.95,
    fatal_q_bound_violation=0.25,
    fatal_policy_drop=0.05,policy_drop_patience=2,
    max_training_fit_budget=12800,
    interpretation='engineering plateau only; a stable bad policy is not a successful method')


def assess(history,rules=RULES):
    last=history[-1]
    for key in ['greedy_mean_end_r2','best_queried_mean_r2','Q_calibration_mae',
                'max_Q_upper_violation','mean_unique_subsets']:
        if not np.isfinite(last[key]):return dict(stop=True,kind='problem',reason=f'nonfinite {key}')
    if last['max_Q_upper_violation']>rules['fatal_q_bound_violation']:
        return dict(stop=True,kind='problem',reason='severe Q upper-bound violation',
            value=last['max_Q_upper_violation'])
    patience=rules['policy_drop_patience']
    late=[r for r in history if r['checkpoint']>=3200]
    initial=history[0]['greedy_mean_end_r2']
    if len(late)>=patience and all(r['greedy_mean_end_r2']<initial-rules['fatal_policy_drop'] for r in late[-patience:]):
        return dict(stop=True,kind='problem',reason='persistent large frozen-policy degradation')
    w=history[-rules['window']:]
    if (len(w)==rules['window'] and last['checkpoint']>=rules['minimum_plateau_checkpoint']
            and w[0]['checkpoint']>=rules['window_start_at_least']):
        endpoint=float(np.ptp([r['greedy_mean_end_r2'] for r in w]))
        gain=w[-1]['best_queried_mean_r2']-w[0]['best_queried_mean_r2']
        calibration=float(np.ptp([r['Q_calibration_mae'] for r in w]))
        agreements=[float(np.mean(np.array(a['bank_actions'])==np.array(b['bank_actions']))) for a,b in zip(w,w[1:])]
        stats=dict(endpoint_range=endpoint,best_queried_r2_gain=gain,
            calibration_range=calibration,minimum_action_agreement=min(agreements))
        if (endpoint<=rules['endpoint_r2_range'] and gain<=rules['best_queried_r2_gain']
                and calibration<=rules['calibration_mae_range']
                and min(agreements)>=rules['minimum_action_agreement']):
            return dict(stop=True,kind='plateau',reason='stable policy, performance, incumbent and calibration',statistics=stats)
    if last['checkpoint']>=rules['max_training_fit_budget']:
        return dict(stop=True,kind='budget_cap',reason='fixed maximum budget reached; convergence not established')
    return dict(stop=False,kind='continue',reason='no plateau/failure rule satisfied')
