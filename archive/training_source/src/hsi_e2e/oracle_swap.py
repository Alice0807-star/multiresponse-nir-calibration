"""Full one-swap diagnostic labels, with frozen historical query exclusions."""
from __future__ import annotations
import json
import numpy as np
from scipy.stats import spearmanr
from .diverse_search import build_ranking,select_batch
from .local_search import swap_subset

def rank_correlation(a,b):
    return float(spearmanr(a,b).statistic) if np.ptp(a)>0 and np.ptp(b)>0 else 0.


def uniform_best_accepted_expectation(gains,k=8):
    values=np.sort(np.asarray(gains,dtype=float))
    n=len(values)
    if n==0:
        return 0.
    k=min(k,n)
    ranks=np.arange(1,n+1,dtype=float)
    weights=(k/n)*np.prod((ranks[:,None]-1-np.arange(k-1))/(n-1-np.arange(k-1)),axis=1) if k>1 else np.full(n,1/n)
    weights[:k-1]=0.
    if not np.isclose(weights.sum(),1.,atol=1e-10):
        raise AssertionError('Invalid order statistic probabilities')
    return float(weights@np.maximum(values,0.))


def freeze_lineage(pool,batches,queries):
    last=batches[-1]
    episode=int(last['episode'])
    first=next(r for r in batches if int(r['episode'])==episode)
    root=tuple(pool[(episode-1)%len(pool)])
    assert tuple(json.loads(first['parent_subset_json']))==root
    panels=[]
    for label,batch in [('initial',first),('late',last)]:
        decision=int(batch['decision'])
        cutoff=int(batch['fit_count'])-int(batch['batch_size'])
        seen=[json.loads(r['subset_json']) for r in queries if int(r['scope_fit_count'])<=cutoff]
        panels.append(dict(panel=label,episode=episode,step=int(batch['step']),decision=decision,
            parent_subset=json.loads(batch['parent_subset_json']),before_nmse=float(batch['accepted_nmse_before']),
            prefix_fit_count=cutoff,seen=seen))
    return panels


def evaluate_panel(evaluator,builder,model,panel,seen):
    parent=tuple(panel['parent_subset'])
    before=float(evaluator.evaluate(parent))
    if abs(before-panel['before_nmse'])>1e-10:
        raise AssertionError('Frozen parent NMSE mismatch')
    ranking=build_ranking(evaluator,parent,before,model,'extra_trees',builder)
    records=[]
    truth=[]
    unseen=[]
    for pred,remove,add in zip(ranking['scores'],ranking['removes'],ranking['adds']):
        child=swap_subset(parent,int(remove),int(add))
        after=float(evaluator.evaluate(child))
        gain=before-after
        available=child not in seen
        records.append(dict(panel=panel['panel'],remove=int(remove),add=int(add),
            predicted_gain=float(pred),true_gain=gain,after_nmse=after,
            unseen=available,child_subset_json=json.dumps(child)))
        truth.append(gain)
        unseen.append(available)
    truth=np.array(truth)
    unseen=np.array(unseen,bool)
    scores=ranking['scores']
    oracle_all=float(max(truth.max(),0.))
    oracle_unseen=float(max(truth[unseen].max(),0.)) if unseen.any() else 0.
    removal_rows=[]
    for remove in parent:
        mask=ranking['removes']==remove
        v=scores[mask]
        g=truth[mask]
        selected=int(np.argmax(v))
        removal_rows.append(dict(panel=panel['panel'],remove=remove,
            within_remove_spearman=rank_correlation(v,g),
            predicted_best_gain=float(v.max()),actual_gain_of_predicted_best=float(g[selected]),
            oracle_best_gain=float(g.max()),predicted_mean_gain=float(v.mean()),actual_mean_gain=float(g.mean())))
    decisions=[]
    for arm in ['ranked_top8','delete_diverse8','joint_diverse8']:
        r={k:v.copy() for k,v in ranking.items()}
        chosen=select_batch(parent,r,seen,arm,builder.correlation,np.random.default_rng(0),batch_size=8,corr_threshold=.95)
        ids=[p['index'] for p in chosen]
        accepted=float(max(truth[ids].max(),0.)) if ids else 0.
        covered=np.isin(ranking['removes'],[p['remove'] for p in chosen])&unseen
        restricted=float(max(truth[covered].max(),0.)) if covered.any() else 0.
        decisions.append(dict(panel=panel['panel'],method=arm,n_proposals=len(ids),
            accepted_gain=accepted,accepted_after_nmse=before-accepted,
            oracle_unseen_regret=oracle_unseen-accepted,
            predicted_mean_gain=float(scores[ids].mean()) if ids else 0.,
            actual_mean_gain=float(truth[ids].mean()) if ids else 0.,
            oracle_over_covered_removals_gain=restricted,
            removal_coverage_regret=oracle_unseen-restricted,
            addition_selection_regret=restricted-accepted,
            chosen_pairs=[[p['remove'],p['add']] for p in chosen],
            max_relaxation=max([p['relaxation'] for p in chosen],default=0)))
    expected=uniform_best_accepted_expectation(truth[unseen])
    decisions.append(dict(panel=panel['panel'],method='uniform8_exact_expectation',n_proposals=min(8,int(unseen.sum())),
        accepted_gain=expected,accepted_after_nmse=before-expected,oracle_unseen_regret=oracle_unseen-expected,
        chosen_pairs=[],max_relaxation=0))
    summary=dict(panel=panel['panel'],before_nmse=before,n_candidates=len(truth),n_unseen=int(unseen.sum()),
        oracle_all_accepted_gain=oracle_all,oracle_unseen_accepted_gain=oracle_unseen,
        oracle_all_after_nmse=before-oracle_all,oracle_unseen_after_nmse=before-oracle_unseen,
        overall_spearman=rank_correlation(scores,truth),
        unseen_spearman=rank_correlation(scores[unseen],truth[unseen]) if unseen.any() else 0.,
        mean_within_remove_spearman=float(np.mean([r['within_remove_spearman'] for r in removal_rows])),
        cross_remove_best_spearman=rank_correlation([r['predicted_best_gain'] for r in removal_rows],[r['oracle_best_gain'] for r in removal_rows]),
        gain_mae=float(np.abs(scores-truth).mean()),prediction_bias=float((scores-truth).mean()))
    return records,removal_rows,decisions,summary
