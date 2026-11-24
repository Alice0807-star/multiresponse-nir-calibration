"""Pure provenance and ranking rules for the four-panel controlled check."""
import math

CUTS=(1024,2048,3072,4096)


def prefix_unions(ledgers,expected_final,*,cuts=CUTS,n_bands=641,k=16):
    if len(ledgers)!=3 or not cuts or tuple(sorted(set(cuts)))!=tuple(cuts):
        raise ValueError('Exactly three target ledgers and increasing unique cuts required')
    snapshots=[]
    for ledger in ledgers:
        if len(ledger)!=cuts[-1]:raise ValueError('Exact paid ledger length required')
        seen=set();best=None;best_index=None;prefix={}
        for index,row in enumerate(ledger,1):
            subset=tuple(row['subset']);score=row['mean_r2']
            if row['actual_fits']!=index or len(subset)!=k or tuple(sorted(set(subset)))!=subset or not all(type(v) is int and 0<=v<n_bands for v in subset):
                raise ValueError('Invalid canonical paid ledger')
            if subset in seen or not math.isfinite(score) or len(row['r2'])!=1 or row['r2'][0]!=score:
                raise ValueError('Repeated paid subset or invalid single-target score')
            seen.add(subset)
            if best is None or score>best['mean_r2']:best=row;best_index=index
            if index in cuts:prefix[index]=dict(subset=list(best['subset']),paid_best_index=best_index,reward_r2=best['mean_r2'])
        snapshots.append(prefix)
    panels=[]
    for i,cut in enumerate(cuts):
        provenance=[a[cut] for a in snapshots]
        subset=sorted(set().union(*(set(a['subset']) for a in provenance)))
        if not k<=len(subset)<=3*k:raise AssertionError('Union input cap')
        panels.append(dict(id='U'+str(i),cut_per_target=cut,source_queries=3*cut,subset=subset,private_provenance=provenance))
    if panels[-1]['subset']!=list(expected_final):raise ValueError('Final union differs from original selected panel')
    if len({tuple(p['subset']) for p in panels})!=len(cuts):
        raise ValueError('Duplicate four-panel pool;stop instead of silently replacing candidates')
    return panels


def choose_panel(panel_ids,teacher_scores,neural_scores):
    if len(set(panel_ids))!=len(panel_ids) or not panel_ids:raise ValueError('Distinct fixed candidates required')
    if set(teacher_scores)!=set(panel_ids) or set(neural_scores)!=set(panel_ids):raise ValueError('Incomplete ranking table')
    for panel in panel_ids:
        if len(neural_scores[panel])!=3 or not all(math.isfinite(x) for x in [teacher_scores[panel],*neural_scores[panel]]):
            raise ValueError('Finite three-target neural and teacher scores required')
    # max keeps the first member on exact ties:chronological prefix order.
    pls=max(panel_ids,key=lambda p:teacher_scores[p])
    mlp=max(panel_ids,key=lambda p:sum(neural_scores[p])/3)
    return dict(PLS=pls,MLP=mlp)


def strict_contrast(rows,method,reference,splits,seeds):
    pairs=[]
    for split in splits:
        for seed in seeds:
            a=next(r for r in rows if (r['split'],r['seed'],r['method'])==(split,seed,method))
            b=next(r for r in rows if (r['split'],r['seed'],r['method'])==(split,seed,reference))
            delta=[x-y for x,y in zip(a['audit']['r2'],b['audit']['r2'])]
            pairs.append(dict(split=split,seed=seed,target_delta=delta,mean_delta=sum(delta)/3))
    target=[sum(p['target_delta'][t] for p in pairs)/len(pairs) for t in range(3)]
    parts={str(s):sum(p['mean_delta'] for p in pairs if p['split']==s)/len(seeds) for s in splits}
    wins=sum(p['mean_delta']>0 for p in pairs)
    return dict(mean_delta=sum(target)/3,target_delta=target,partition_deltas=parts,wins=wins,pairs=pairs,
        descriptive_support=bool(sum(target)>0 and min(target)>=0 and min(parts.values())>0 and wins>=math.ceil(2*len(pairs)/3)))
