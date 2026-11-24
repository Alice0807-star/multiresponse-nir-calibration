"""Conservative confirmation of a provisional plateau, including late recovery.

The in-flight core runner's plateau is only a candidate. If it stops before
confirmation, its complete state must be continued within the original cap.
No reward/network/optimizer parameter is changed by this outer stopping rule.
"""
from .convergence import assess,RULES
import numpy as np

CONFIRMATION = dict(additional_training_fits=3200,minimum_later_checkpoints=2,
    preserve_maximum_budget=12800,
    reset_on='renewed performance/incumbent improvement, action changes or calibration instability',
    meaning='finite confirmation cannot prove that improvement will never recur')


def assess_with_confirmation(history):
    candidate=None;candidate_index=None
    for index in range(len(history)):
        prefix=history[:index+1];current=history[index]
        result=assess(prefix)
        if result['kind']=='problem':return result
        if result['kind']=='plateau':
            if candidate is None:
                candidate=current['checkpoint'];candidate_index=index
            window=history[candidate_index:index+1]
            changed=(np.ptp([r['greedy_mean_end_r2'] for r in window])>RULES['endpoint_r2_range']
                or window[-1]['best_queried_mean_r2']-window[0]['best_queried_mean_r2']>RULES['best_queried_r2_gain']
                or np.ptp([r['Q_calibration_mae'] for r in window])>RULES['calibration_mae_range']
                or any(np.mean(np.array(a['bank_actions'])==np.array(b['bank_actions']))<RULES['minimum_action_agreement']
                    for a,b in zip(window,window[1:])))
            if changed:
                candidate=current['checkpoint'];candidate_index=index
            if (current['checkpoint']-candidate>=CONFIRMATION['additional_training_fits']
                    and index-candidate_index>=CONFIRMATION['minimum_later_checkpoints']):
                return dict(stop=True,kind='confirmed_plateau',candidate_checkpoint=candidate,
                    confirmed_checkpoint=current['checkpoint'],reason='plateau persisted through additional observation',
                    convergence_proven=False)
        else:
            candidate=None;candidate_index=None
    last=history[-1]
    if last['checkpoint']>=CONFIRMATION['preserve_maximum_budget']:
        return dict(stop=True,kind='budget_cap',reason='budget exhausted without confirmed plateau',convergence_proven=False)
    if candidate is not None:
        return dict(stop=False,kind='plateau_candidate',candidate_checkpoint=candidate,
            additional_fits_required=max(0,candidate+CONFIRMATION['additional_training_fits']-last['checkpoint']))
    return dict(stop=False,kind='continue',reason='no persistent plateau')
