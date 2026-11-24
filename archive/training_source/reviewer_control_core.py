"""Pure design/aggregation helpers for the frozen reviewer controls."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/reviewer_control_v1'
READY=ROOT/'results/reviewer_control_readiness_001'
DOC=ROOT/'docs/审稿补强实验协议与终端说明_2026-11-21.md'
SPEC=dict(version='reviewer_control_v1',declared_date='2026-11-21',
    original_selector_seeds=[17,29,43],new_outer_seeds=[31001,31002],new_selector_seed=17,
    random_panel_seeds=list(range(40101,40111)),random_replicates=10,search_components=8,
    search_budget=3072,wide_cap=15,quota=dict(cgl=12,corn=11),
    target_count=4,primary_contrasts=['B_minus_D_low','C_minus_D_wide'],
    retained_contrasts=['B_minus_A','C_minus_B'],
    secondary_contrast='C_minus_B_minus_mean_D_wide_minus_D_low',
    scope='Post-specified same-source extension;not independent blind validation',
    D_budget_scope='Per-target input count matched;global union,correlation,search cost not matched',
    rng_rule='PCG64 SeedSequence[panel_seed,source_code,outer_seed,fold_number,selector_seed,target]',
    duplicate_draw_rule='Independent replicate streams;coincidental identical panels allowed,no redraw',
    feasibility='wide grid1..min(15,input_columns,fit_rows-1);limited grid1..quota',
    numerical_failure='Stop on PLS fitting warnings,nonfinite values or isolation/identity failure;no score-driven retry',
    evaluation='Pool each seed and replicate OOF before metrics;mean metrics,never average predictions',
    stop_rule='Exactly10D replicates and2newCorn5fold partitions;no outcome-driven additions')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_value(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,Path): return str(value)
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')
def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=json_value)+'\n',encoding='utf-8')
    temp.replace(path)
def check_hashes(values):
    for path,digest in values.items():
        if sha(path)!=digest: raise RuntimeError('Frozen file changed: '+path)
def cell_id(source,cohort,fold,seed): return f'{source}__{cohort}__{fold}__s{seed}'
def entry_id(cell,kind,target,replicate=None):
    return f'{cell["id"]}__{kind}__t{target}'+('' if replicate is None else f'__r{replicate:02d}')


def validate_panel(own,union,bands,quota):
    if own!=sorted(set(own)) or union!=sorted(set(union)):
        raise ValueError('Column lists must be sorted and unique')
    if len(own)!=quota or not set(own)<=set(union) or not union or min(union)<0 or max(union)>=bands:
        raise ValueError('Invalid own/union panel')


def random_panel(cell,own,union,target,replicate,spec=SPEC):
    validate_panel(own,union,cell['bands'],cell['quota'])
    if not 1<=replicate<=spec['random_replicates']: raise ValueError('Replicate outside frozen range')
    source_code={'cgl':1,'corn':2}[cell['source']]
    fold_number=0 if cell['fold']=='official' else int(cell['fold'].split('_')[-1])
    entropy=[spec['random_panel_seeds'][replicate-1],source_code,cell['outer_seed'],fold_number,cell['selector_seed'],target]
    rng=np.random.default_rng(np.random.SeedSequence(entropy))
    outside=np.array([i for i in range(cell['bands']) if i not in set(own)])
    extra=rng.choice(outside,len(union)-len(own),replace=False)
    panel=sorted(own+extra.tolist())
    if len(panel)!=len(union) or len(set(panel))!=len(panel) or not set(own)<=set(panel):
        raise AssertionError('Random extension dimension/containment failed')
    return panel,entropy


def choose(trials,cap):
    rows=[r for r in trials if r['components']<=cap]
    if [r['components'] for r in rows]!=list(range(1,cap+1)):
        raise ValueError('Missing,duplicate or unordered component candidate')
    if not all(np.isfinite(r['mean_r2']) for r in rows): raise ValueError('Nonfinite validation score')
    return max(rows,key=lambda r:r['mean_r2'])


def make_partitions(n,seed,folds=5):
    if n%folds: raise ValueError('Equal held-out folds required')
    groups=np.array_split(np.random.default_rng(seed).permutation(n),folds); result=[]
    for i,test in enumerate(groups):
        train=np.array(sorted(set(range(n))-set(test)),dtype=int)
        order=np.random.default_rng(seed+100+i).permutation(len(train))
        cut=3*len(train)//4
        result.append(dict(fold=f'fold_{i+1}',fit_ids=train[order[:cut]].tolist(),
                           validation_ids=train[order[cut:]].tolist(),test_ids=test.tolist(),
                           inner_seed=seed+100+i))
    flat=[i for r in result for i in r['test_ids']]
    if sorted(flat)!=list(range(n)): raise AssertionError('Outer folds do not cover unique samples')
    return result


def split_signature(records):
    return sorted(tuple(sorted(r['test_ids'])) for r in records)


def metrics(y,pred,training_sd=None):
    y,pred=np.asarray(y),np.asarray(pred)
    if y.shape!=pred.shape or y.ndim!=2 or not np.isfinite(y).all() or not np.isfinite(pred).all(): raise ValueError('Invalid prediction arrays')
    den=((y-y.mean(0))**2).sum(0)
    if min(den)<=1e-12: raise ValueError('Constant evaluation response')
    err=pred-y; r2=1-(err**2).sum(0)/den
    value=dict(r2=r2.tolist(),mean_r2=float(r2.mean()),rmse=np.sqrt((err**2).mean(0)).tolist(),mae=abs(err).mean(0).tolist())
    if training_sd is not None:
        sd=np.asarray(training_sd)
        if sd.shape!=y.shape or not np.isfinite(sd).all() or min(sd.ravel())<=0: raise ValueError('Invalid training SD alignment')
        value['normalized_rmse']=np.sqrt(((err/sd)**2).mean(0)).tolist()
    return value


def pool(rows,expected_ids):
    ids=[]; yy=[]; pp=[]; sds=[]
    for row in rows:
        with np.load(row['prediction_file'],allow_pickle=False) as z:
            ids.append(z['ids'].copy()); yy.append(z['target'].copy()); pp.append(z['prediction'].copy())
        sds.append(np.broadcast_to(np.asarray(row['outer_training_sd']),yy[-1].shape))
    ids=np.concatenate(ids); order=np.argsort(ids)
    if len(ids)!=len(set(ids.tolist())) or sorted(ids.tolist())!=sorted(expected_ids):
        raise ValueError('OOF identity/coverage mismatch')
    y,p,sd=[np.vstack(a)[order] for a in (yy,pp,sds)]
    return dict(ids=ids[order],target=y,prediction=p,score=metrics(y,p,sd))


def mean_metrics(values):
    return {k:float(np.mean([v[k] for v in values])) if k=='mean_r2' else np.mean([v[k] for v in values],axis=0).tolist()
            for k in ('r2','mean_r2','rmse','mae','normalized_rmse')}


def summarize(rows,cells,spec=SPEC):
    cohorts=defaultdict(list)
    for cell in cells: cohorts[(cell['source'],cell['cohort'])].append(cell)
    result={}; raw=[]
    for (source,cohort),cs in sorted(cohorts.items()):
        seeds=sorted({c['selector_seed'] for c in cs}); expected=sorted({i for c in cs for i in c['test_ids']})
        per_seed={}; reference=None
        for seed in seeds:
            per_seed[str(seed)]={}
            for arm in ('A','B','C','D_low','D_wide'):
                reps=[None] if arm in ('A','B','C') else list(range(1,spec['random_replicates']+1))
                for rep in reps:
                    selected=[r for r in rows if (r['source'],r['cohort'],r['selector_seed'],r['arm'],r['replicate'])==(source,cohort,seed,arm,rep)]
                    if len(selected)!=len([c for c in cs if c['selector_seed']==seed]): raise ValueError('Missing prediction cell')
                    v=pool(selected,expected)
                    if reference is None: reference=v
                    if not np.array_equal(reference['ids'],v['ids']) or not np.array_equal(reference['target'],v['target']):
                        raise ValueError('Paired targets/IDs differ')
                    key=arm if rep is None else f'{arm}:{rep}'
                    per_seed[str(seed)][key]=v['score']
                    raw.append(dict(source=source,cohort=cohort,selector_seed=seed,arm=arm,replicate=rep,rows=len(expected),**v['score']))
        abc={a:mean_metrics([per_seed[str(s)][a] for s in seeds]) for a in ('A','B','C')}
        draw={a:[dict(replicate=r,**mean_metrics([per_seed[str(s)][f'{a}:{r}'] for s in seeds]))
                 for r in range(1,spec['random_replicates']+1)] for a in ('D_low','D_wide')}
        methods={**abc,**{a:mean_metrics(v) for a,v in draw.items()}}
        contrasts={}
        for c,a in [('B','A'),('C','B'),('B','D_low'),('C','D_wide')]:
            contrasts[c+'_minus_'+a]=dict(mean_r2=methods[c]['mean_r2']-methods[a]['mean_r2'],
                target_r2=(np.array(methods[c]['r2'])-methods[a]['r2']).tolist(),
                rmse_delta=(np.array(methods[c]['rmse'])-methods[a]['rmse']).tolist(),
                mae_delta=(np.array(methods[c]['mae'])-methods[a]['mae']).tolist(),
                normalized_rmse_delta=(np.array(methods[c]['normalized_rmse'])-methods[a]['normalized_rmse']).tolist())
        contrasts['range_difference']=contrasts['C_minus_B']['mean_r2']-(methods['D_wide']['mean_r2']-methods['D_low']['mean_r2'])
        ranks={a:dict(B_or_C_better_count=sum(abc['B' if a=='D_low' else 'C']['mean_r2']>v['mean_r2'] for v in vv),
                     draws=len(vv),min=float(min(v['mean_r2'] for v in vv)),max=float(max(v['mean_r2'] for v in vv)),
                     description='Randomization distribution description;not a p-value or independent-data confidence interval') for a,vv in draw.items()}
        result[source+'/'+cohort]=dict(methods=methods,per_seed=per_seed,D_draws=draw,contrasts=contrasts,random_reference_position=ranks,
            sample_count_per_seed=len(expected),algorithm_seeds=seeds,
            D_global_union_counts=[r['global_input_union_count'] for r in rows if r['source']==source and r['cohort']==cohort and r['arm']=='D_low'])
    sensitivity=[]
    for key,v in result.items():
        if key.startswith('corn/') and str(spec['new_selector_seed']) in v['per_seed']:
            ss=v['per_seed'][str(spec['new_selector_seed'])]
            sensitivity.append(dict(cohort=key.split('/')[1],selector_seed=spec['new_selector_seed'],
                A=ss['A']['mean_r2'],B=ss['B']['mean_r2'],C=ss['C']['mean_r2'],
                D_low=float(np.mean([ss[f'D_low:{r}']['mean_r2'] for r in range(1,spec['random_replicates']+1)])),
                D_wide=float(np.mean([ss[f'D_wide:{r}']['mean_r2'] for r in range(1,spec['random_replicates']+1)])),
                B_minus_A=ss['B']['mean_r2']-ss['A']['mean_r2'],C_minus_B=ss['C']['mean_r2']-ss['B']['mean_r2']))
    return dict(cohorts=result,split_sensitivity_fixed_seed=sensitivity,
                scope='Each partition has its own OOF metrics;no concatenation into240 independent samples'),raw
