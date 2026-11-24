"""Frozen-model proposal diversity; blocked candidates remain eligible later."""
from __future__ import annotations
import json,time
import numpy as np
from .local_search import evaluate_pool,swap_subset
from .swap_supervision import candidate_states,predict_scores

ARMS=['ranked_top8','delete_diverse8','joint_diverse8','ranked4_random4']

def build_ranking(evaluator,parent,error,model,mode,feature_builder):
    parent=tuple(parent)
    context=feature_builder.context(evaluator,parent)
    available=np.array([i for i in range(evaluator.n_bands) if i not in parent],dtype=int)
    scores=[]
    removes=[]
    adds=[]
    for remove in parent:
        f=feature_builder.matrix(context,remove,available)
        s=candidate_states(parent,error,remove,evaluator.n_bands)
        values=predict_scores(model,mode,f,s,available)
        scores.extend(values.tolist())
        removes.extend([remove]*len(available))
        adds.extend(available.tolist())
    scores=np.asarray(scores,np.float64)
    removes=np.asarray(removes,np.int64)
    adds=np.asarray(adds,np.int64)
    if not np.isfinite(scores).all():
        raise FloatingPointError('Nonfinite frozen proposal scores')
    order=np.lexsort((adds,removes,-scores))
    return dict(scores=scores[order],removes=removes[order],adds=adds[order],
        banned=np.zeros(len(order),bool))

def select_batch(parent,ranking,seen_keys,arm,correlation,rng,*,batch_size=8,corr_threshold=.95):
    if arm not in ARMS:
        raise ValueError(arm)
    parent=tuple(parent)
    removes,adds,scores=ranking['removes'],ranking['adds'],ranking['scores']
    banned=ranking['banned']
    selected=[]
    level=0
    def candidate(index):
        child=swap_subset(parent,int(removes[index]),int(adds[index]))
        if child in seen_keys:
            banned[index]=True
            return None
        return child
    while len(selected)<batch_size:
        uniform=(arm=='ranked4_random4' and len(selected)>=4)
        draws=0
        if uniform:
            if banned.all():
                break
            while True:
                draws+=1
                index=int(rng.integers(len(scores)))
                if banned[index]:
                    continue
                child=candidate(index)
                if child is not None:
                    break
                if banned.all():
                    return selected
        else:
            feasible=~banned.copy()
            if arm in ['delete_diverse8','joint_diverse8'] and level<2:
                used=[r['remove'] for r in selected]
                feasible &= ~np.isin(removes,used)
            if arm=='joint_diverse8' and level==0 and selected:
                chosen_adds=[r['add'] for r in selected]
                feasible &= (np.abs(correlation[np.ix_(adds,chosen_adds)])<=corr_threshold+1e-12).all(1)
            child=None
            for index in np.flatnonzero(feasible):
                index=int(index)
                child=candidate(index)
                if child is not None:
                    break
            if child is None:
                if arm=='joint_diverse8' and level==0:
                    level=1  # Drop spectral constraint, preserve removal coverage.
                    continue
                if arm in ['delete_diverse8','joint_diverse8'] and level<2:
                    level=2  # Last resort if insufficient unqueried removals.
                    continue
                break
        # Only paid-before or about-to-be-paid candidates are banned forever.
        # Diversity-rejected candidates have not been touched.
        banned[index]=True
        selected.append(dict(index=index,score=float(scores[index]),remove=int(removes[index]),
            add=int(adds[index]),child=child,proposal_source='uniform' if uniform else 'ranked',
            relaxation=int(level),uniform_draws=draws))
    return selected

def run_diverse(evaluator,pool,*,seed,arm,mode,model,feature_builder,horizon=16,
                batch_size=8,corr_threshold=.95,random_seed_offset=7000000):
    started=time.perf_counter()
    scored=evaluate_pool(evaluator,pool)
    best,best_error=min(scored,key=lambda item:item[1])
    trace=[dict(phase='initial_pool',subset_json=json.dumps(s),validation_nmse=e,
        fit_count=i+1,incumbent_nmse=min(x[1] for x in scored[:i+1])) for i,(s,e) in enumerate(scored)]
    rng=np.random.default_rng(seed+random_seed_offset)
    rankings={}
    batches=[]
    episode=decisions=accepted=0
    while evaluator.fit_count<evaluator.fit_budget:
        current,error=scored[episode%len(scored)]
        for step in range(horizon):
            if evaluator.fit_count>=evaluator.fit_budget:
                break
            if current not in rankings:
                rankings[current]=build_ranking(evaluator,current,error,model,mode,feature_builder)
            proposals=select_batch(current,rankings[current],evaluator._cache,arm,
                feature_builder.correlation,rng,batch_size=min(batch_size,evaluator.fit_budget-evaluator.fit_count),
                corr_threshold=corr_threshold)
            if not proposals:
                break
            # Freeze complete batch before any child reward is observed.
            outcomes=[]
            for slot,proposal in enumerate(proposals,1):
                child=proposal['child']
                value=float(evaluator.evaluate(child))
                outcomes.append((child,value))
                if value<best_error:
                    best,best_error=child,value
                trace.append(dict(phase='search',episode=episode+1,step=step+1,decision=decisions+1,
                    slot=slot,current_subset_json=json.dumps(current),subset_json=json.dumps(child),
                    removed=proposal['remove'],added=proposal['add'],before_nmse=error,
                    validation_nmse=value,predicted_gain=proposal['score'],true_gain=error-value,
                    fit_count=evaluator.fit_count,incumbent_nmse=best_error,
                    proposal_source=proposal['proposal_source'],relaxation=proposal['relaxation'],
                    rank_index=proposal['index'],uniform_draws=proposal['uniform_draws']))
            chosen=min(outcomes,key=lambda item:item[1])
            after_error=min(error,chosen[1])
            added=[r['add'] for r in proposals]
            spectral=feature_builder.correlation[np.ix_(added,added)]
            pairs=np.abs(spectral[np.triu_indices(len(added),1)])
            batches.append(dict(decision=decisions+1,episode=episode+1,step=step+1,
                parent_subset_json=json.dumps(current),batch_size=len(proposals),
                unique_removals=len({r['remove'] for r in proposals}),unique_additions=len(set(added)),
                mean_add_abs_corr=float(pairs.mean()) if len(pairs) else 0.,
                max_add_abs_corr=float(pairs.max()) if len(pairs) else 0.,
                relaxation=max(r['relaxation'] for r in proposals),corr_threshold=corr_threshold,
                accepted_nmse_before=error,accepted_nmse_after=after_error,fit_count=evaluator.fit_count))
            if chosen[1]<error:
                current,error=chosen
                accepted+=1
            decisions+=1
        episode+=1
        if episode>10000:
            raise RuntimeError('No progress through available neighborhoods')
    return dict(best_subset=list(best),best_validation_nmse=best_error,
        initial_best_nmse=min(e for _,e in scored),trace=trace,batches=batches,
        search_seconds=time.perf_counter()-started,decisions=decisions,
        accepted_moves=accepted,episodes=episode,evaluator_counters=evaluator.snapshot())
