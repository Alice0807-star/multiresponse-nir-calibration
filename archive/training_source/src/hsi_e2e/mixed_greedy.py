"""Original greedy-swap8 with only the complete-episode restart rule changed."""
from __future__ import annotations
import heapq
import json
import time
import numpy as np
from .local_search import evaluate_pool,random_swap


def run_mixed_greedy(evaluator,pool,*,seed,mix_probability=.5,elite_size=16,horizon=16,batch_size=8):
    if not 0<=mix_probability<=1 or elite_size<1:raise ValueError('Invalid restart mixture')
    started=time.perf_counter();rng=np.random.default_rng(seed+100000)
    restart_rng=np.random.default_rng(seed+910000)
    scored=evaluate_pool(evaluator,pool);best_subset,best_error=min(scored,key=lambda p:p[1])
    initial_best=best_error;trace=[];restarts=[];decisions=[]
    for i,(subset,score) in enumerate(scored):
        trace.append(dict(phase='initial_pool',query=i+1,fit_count=i+1,subset_json=json.dumps(subset),
            validation_nmse=score,incumbent_nmse=min(v for _,v in scored[:i+1]),elapsed_seconds=time.perf_counter()-started))
    episode=decision=proposals=accepted=0
    while evaluator.fit_count<evaluator.fit_budget:
        index=episode%len(scored)
        coin=float(restart_rng.random()) if mix_probability>0 else None
        use_elite=coin is not None and coin<mix_probability;rank=None;elite=[]
        if use_elite:
            elite=heapq.nsmallest(elite_size,evaluator._cache.items(),key=lambda p:(p[1],p[0]))
            rank=int(restart_rng.integers(len(elite)));current,current_error=elite[rank]
        else:current,current_error=scored[index]
        restarts.append(dict(episode=episode+1,source='elite' if use_elite else 'original',coin=coin,
            original_index=index,elite_rank=None if rank is None else rank+1,elite_candidates=len(elite),
            cache_size=len(evaluator._cache),fit_count=evaluator.fit_count,
            subset_json=json.dumps(current),start_nmse=current_error))
        for step in range(horizon):
            if evaluator.fit_count>=evaluator.fit_budget:break
            candidates=[];attempts=0
            while len(candidates)<batch_size and evaluator.fit_count<evaluator.fit_budget:
                attempts+=1
                if attempts>10000:raise RuntimeError('Could not propose an unseen complete subset')
                remove,add,candidate=random_swap(current,evaluator.n_bands,rng)
                if candidate in evaluator._cache:continue
                score=float(evaluator.evaluate(candidate))
                if not np.isfinite(score):raise FloatingPointError('Nonfinite greedy evaluation')
                proposals+=1;candidates.append((candidate,score,remove,add))
                if score<best_error:best_subset,best_error=candidate,score
                trace.append(dict(phase='search',episode=episode+1,step=step+1,decision=decision+1,
                    proposal_in_decision=len(candidates),current_subset_json=json.dumps(current),
                    subset_json=json.dumps(candidate),removed_1based=remove+1,added_1based=add+1,
                    validation_nmse=score,before_nmse=current_error,reward=current_error-score,
                    incumbent_nmse=best_error,fit_count=evaluator.fit_count,elapsed_seconds=time.perf_counter()-started))
            if not candidates:break
            chosen=min(candidates,key=lambda p:p[1]);before=current_error;before_subset=current
            take=chosen[1]<current_error
            if take:current,current_error=chosen[:2];accepted+=1
            decisions.append(dict(episode=episode+1,step=step+1,decision=decision+1,fit_count=evaluator.fit_count,
                before_json=json.dumps(before_subset),chosen_json=json.dumps(chosen[0]),after_json=json.dumps(current),
                before_nmse=before,chosen_nmse=chosen[1],after_nmse=current_error,accepted=take,
                candidates=len(candidates),proposal_attempts=attempts))
            decision+=1
        episode+=1
    return dict(best_subset=list(best_subset),best_validation_nmse=best_error,initial_best_nmse=initial_best,
        search_seconds=time.perf_counter()-started,trace=trace,restarts=restarts,decision_trace=decisions,
        evaluator_counters=evaluator.snapshot(),decisions=decision,proposed_complete_subsets=proposals,
        accepted_moves=accepted,episodes=episode,horizon=horizon,batch_size=batch_size,
        mix_probability=mix_probability,elite_size=elite_size)
