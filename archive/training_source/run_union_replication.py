"""Complete same-union ST controls and two prespecified fixed-budget replications.

No historical code/results are edited. All new choices are sealed globally
before this runner opens any audit arrays. Historical exposure is disclosed.
"""
from pathlib import Path
from datetime import datetime,timezone
import sys,signal,time,warnings,traceback
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from threadpoolctl import threadpool_limits,threadpool_info
import run_empirical_budget_pilot as pilot
from empirical_budget_core import greedy,covsel
from hsi_e2e.framework_screen import metrics,nn_predict,pls_fit,pls_predict
from hsi_e2e.formal_runtime import read,write,sha,torch_atomic

OUT=ROOT/'results/union_replication_001'
OLD=pilot.base.OLD
SPLITS=(7117,8111,9103);NEW_SPLITS=(8111,9103);SEEDS=(17,29,43)
NN_METHODS=('union_mt','union_st_width','union_st_capacity','shared_mt','individual_st','covsel_mt','full_mt')
PLS_METHODS=('shared_pls','covsel_pls','full_pls')
METHODS=NN_METHODS+PLS_METHODS
DURATION=dict(max_epochs=2400,min_epochs=1200,patience=600)
PRIMARY_REFERENCES=('shared_mt','union_st_width','union_st_capacity')


def tasks(method):return [0,1,2] if method in ('union_st_width','union_st_capacity','individual_st') else [None]
def widths(method):return (24,23) if method=='union_st_capacity' else (64,32)
def reused(split,method):return split==7117 and method not in ('union_st_width','union_st_capacity')
def folder_for(split,seed,method,task):
    parent=pilot.OUT if reused(split,method) else OUT/str(split)
    return parent/'models'/str(seed)/method/('joint' if task is None else str(task))


def counts():
    return dict(new_nn=sum(len(tasks(m)) for s in SPLITS for _ in SEEDS for m in NN_METHODS if not reused(s,m)),
        reused_nn=sum(len(tasks(m)) for s in SPLITS for _ in SEEDS for m in NN_METHODS if reused(s,m)),
        new_pls=sum(not reused(s,m) for s in SPLITS for _ in SEEDS for m in PLS_METHODS),
        reused_pls=sum(reused(s,m) for s in SPLITS for _ in SEEDS for m in PLS_METHODS),pipeline_cases=len(SPLITS)*len(SEEDS)*len(METHODS))


def prepare():
    if OUT.exists():raise RuntimeError('Existing output;refuse automatic restart')
    pilot.lock();pilot.seal();pilot.prior.lock();pilot.base.lock()
    if not read(pilot.OUT/'verification.json')['passed'] or not read(pilot.OUT/'selected_endpoint_verification.json')['passed']:
        raise RuntimeError('Previous pilot not verified')
    previous=read(pilot.OUT/'delivery_manifest.json')['immutable_files']
    for file,digest in previous.items():
        if sha(file)!=digest:raise RuntimeError('Previous scientific delivery changed:'+file)
    sources=[Path(__file__),ROOT/'verify_union_replication.py',ROOT/'tests/test_union_replication.py',
        ROOT/'empirical_budget_core.py',ROOT/'run_empirical_training_control.py']+sorted((ROOT/'src/hsi_e2e').glob('*.py'))
    inputs=[ROOT/'docs/并集同谱对照与三划分复核预先协议_2026-11-18.md',
        ROOT/'docs/线程实际设置核查_2026-11-18.md',pilot.PREFIX/'subsets.json',pilot.PREFIX/'protocol.json',pilot.OUT/'protocol.json',
        pilot.OUT/'panels.json',pilot.OUT/'covsel_selection.json',pilot.OUT/'metrics.json',pilot.OUT/'summary.json',
        pilot.OUT/'choices_locked.json',pilot.OUT/'verification.json',pilot.OUT/'delivery_manifest.json']
    inputs += [OLD/str(s)/name for s in SPLITS for name in ['selection_data.npz','audit_data.npz']]
    inputs += [Path(file) for file in previous if ('/models/' in file.replace('\\','/') or '/predictions/' in file.replace('\\','/'))]
    OUT.mkdir(parents=True)
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),partitions=list(SPLITS),new_partitions=list(NEW_SPLITS),seeds=list(SEEDS),
        nn_methods=list(NN_METHODS),pls_methods=list(PLS_METHODS),counts=counts(),duration=DURATION,
        questions=['Same union panel:does joint training improve over both fixed independent network recipes?',
                   'Does the union-panel candidate retain its accuracy/tradeoffs on all3historical partitions?'],
        selection=dict(total_query_cap=12288,target_query_cap=4096,target_k=16,shared_k=48,components=8),
        final_pls_candidates=[4,8,12,16],nn_hyperparameters='Unchanged AdamWlr.001,decay.001,batch128,dropout.05,clip5,eval10;legacy requestTorch4 then threadpool_limits1;observed effective Torch1/BLAS1/inter-op1',
        capacity_control='Fixed24/23,widths not tuned;report actual parameter discrepancy at each union size;not equal total compute',
        primary_references=list(PRIMARY_REFERENCES),
        descriptive_support_rule='unionMT-minus-each-reference:overallmean>0,all3partitionmeans>0,>=6/9cellwins,all3targetoverallmeans>=0;descriptive only',
        secondary='Full,CovSel,individualST and tunedPLS target/cost tradeoffs;unionSTwidth-minus-individualST separates task-visible inputs at same acquired union',
        reuse='7117fiveNNrecipes and3PLSrecipes byte-preserved;only its2unionST controls are newly trained;older sharedSTpilot scores retained as historical appendix',
        seal='All96newNN+18newPLS choices plus all30reused choices globally hashed before current audit arrays',
        exposure='7117pilot already known;all3partitions reuse historically adapted1680samples;overlapping audits;not independent validation or independent hypothesis tests',
        stopping='Any exception,warning,nonfinite or integrity error stops;normal2400cap recorded,not extended;no new K,LR,width,seed,split or reweighting',
        scope_exclusions='No RL/attention modification,no new-source test evaluation,no hardware savings or fully-converged superiority claim',
        source_sha256={str(f):sha(f) for f in sources},input_sha256={str(f):sha(f) for f in dict.fromkeys(inputs)})
    write(OUT/'protocol.json',p);write(OUT/'status.json',dict(state='prepared',current_audit_opened=False))
    return p


def lock():
    p=read(OUT/'protocol.json')
    for key in ['source_sha256','input_sha256']:
        for file,h in p[key].items():
            if sha(file)!=h:raise RuntimeError('Locked artifact changed:'+file)
    return p


def emit(msg):
    line=datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+msg
    print(line,flush=True)
    with (OUT/'run.log').open('a',encoding='utf-8') as stream:stream.write(line+'\n')


def selection_arrays(split):
    with np.load(OLD/str(split)/'selection_data.npz',allow_pickle=False) as z:
        arrays=[z[k].copy() for k in ['X_fit','y_fit','X_reward','y_reward']]
        pilot.prior.assert_partition(z['fit_ids'],z['reward_ids'])
    if [a.shape for a in arrays]!=[(1008,641),(1008,3),(336,641),(336,3)]:raise ValueError('Unexpected sample/target shapes')
    return arrays


def panels_all():
    panels={'7117':read(pilot.OUT/'panels.json')}
    folder=OUT/'7117';folder.mkdir();write(folder/'covsel_selection.json',read(pilot.OUT/'covsel_selection.json'))
    prefixes=read(pilot.PREFIX/'subsets.json')
    for split in NEW_SPLITS:
        x,y,xv,yv=selection_arrays(split);fold={};tick=time.perf_counter();c=covsel(x,y,48)
        c['selection_seconds']=time.perf_counter()-tick
        folder=OUT/str(split);folder.mkdir();write(folder/'covsel_selection.json',c)
        for seed in SEEDS:
            shared=next(p for p in prefixes if p['split']==split and p['seed']==seed)
            if shared['budget']!=12288 or shared['best_paid_index']>12288:raise AssertionError('Bad prefix')
            own=[]
            for target in range(3):
                state=greedy(x,y[:,target:target+1],xv,yv[:,target:target+1],seed=seed,k=16,budget=4096,emit=emit)
                path=folder/'search'/str(seed)/str(target);path.mkdir(parents=True)
                write(path/'ledger.json',state.ev.ledger);write(path/'decisions.json',state.decisions);write(path/'summary.json',state.summary())
                own.append(list(state.ev.best_subset));emit(f'PANEL split={split} seed={seed} target={target} exact4096 complete')
            union=sorted(set().union(*map(set,own)))
            if not 16<=len(union)<=48:raise AssertionError('Union acquisition cap')
            fold[str(seed)]=dict(shared=shared['subset'],individual=own,union=union,covsel=c['subset'],full=list(range(641)),actual_union_size=len(union))
        panels[str(split)]=fold
    write(OUT/'panels.json',panels);return panels


def subset_for(panels,split,seed,method,task):
    row=panels[str(split)][str(seed)]
    if method.startswith('union_'):return row['union']
    if method=='individual_st':return row['individual'][task]
    if method.startswith('shared_'):return row['shared']
    if method.startswith('covsel_'):return row['covsel']
    if method.startswith('full_'):return row['full']
    raise ValueError(method)


def select_all(panels):
    for split in SPLITS:
        x,y,xv,yv=selection_arrays(split)
        for seed in SEEDS:
            for method in NN_METHODS:
                if reused(split,method):continue
                for target in tasks(method):
                    cols=slice(None) if target is None else slice(target,target+1)
                    subset=subset_for(panels,split,seed,method,target)
                    t=pilot.base.train(x,y[:,cols],xv,yv[:,cols],subset,seed,widths(method),**DURATION)
                    folder=folder_for(split,seed,method,target);folder.mkdir(parents=True)
                    torch_atomic(folder/'selected_model.pt',dict(state=t['model'].state_dict(),norm=t['norm'],subset=subset))
                    write(folder/'history.json',t['history']);fields={k:t[k] for k in ['epoch','epochs_run','selection','updates','seconds']}
                    write(folder/'selected.json',dict(split=split,seed=seed,method=method,task=target,subset=subset,widths=widths(method),
                        parameters=sum(v.numel() for v in t['model'].parameters()),**fields))
                    emit(f'NN SELECT split={split} seed={seed} method={method} target={target} best={t["epoch"]} run={t["epochs_run"]}')
            for method in PLS_METHODS:
                if reused(split,method):continue
                subset=subset_for(panels,split,seed,method,None);trials=[];best=None;tick=time.perf_counter()
                for component in [4,8,12,16]:
                    model=pls_fit(x,y,subset,component);score=metrics(yv,pls_predict(model,xv));trials.append(dict(components=component,**score))
                    if best is None or score['mean_r2']>best[1]['mean_r2']:best=(model,score)
                folder=folder_for(split,seed,method,None);folder.mkdir(parents=True)
                torch_atomic(folder/'selected_model.pt',best[0]);write(folder/'trials.json',trials)
                write(folder/'selected.json',dict(split=split,seed=seed,method=method,subset=subset,components=best[0]['components'],
                    selection=best[1],tuning_fits=4,seconds=time.perf_counter()-tick))
    seal();lock();emit('ALL96NEW NN+18NEW PLS choices sealed;30reused choices also sealed;before current audit')


def choice_files():
    files=[OUT/'panels.json']+[OUT/str(s)/'covsel_selection.json' for s in SPLITS]
    for split in SPLITS:
        for seed in SEEDS:
            for method in METHODS:
                for target in tasks(method):
                    folder=folder_for(split,seed,method,target)
                    files += [folder/'selected.json',folder/'selected_model.pt',folder/('history.json' if method in NN_METHODS else 'trials.json')]
    return files


def seal():
    files={str(f):sha(f) for f in choice_files()};marker=OUT/'choices_locked.json'
    if marker.exists():
        if read(marker)['files']!=files:raise RuntimeError('Sealed choices changed')
    else:write(marker,dict(utc=datetime.now(timezone.utc).isoformat(),new_nn=96,new_pls=18,reused_nn=21,reused_pls=9,files=files))


def load_audit(split):
    if not (OUT/'choices_locked.json').exists():raise RuntimeError('Audit blocked before global choice seal')
    seal();lock()
    with np.load(OLD/str(split)/'audit_data.npz',allow_pickle=False) as z:return tuple(z[k].copy() for k in ['X_audit','y_audit','audit_ids'])


def paired(rows,a,b):
    pairs=[]
    for split in SPLITS:
        for seed in SEEDS:
            x=next(r for r in rows if (r['split'],r['seed'],r['method'])==(split,seed,a))
            y=next(r for r in rows if (r['split'],r['seed'],r['method'])==(split,seed,b))
            pairs.append(dict(split=split,seed=seed,mean_delta=x['audit']['mean_r2']-y['audit']['mean_r2'],
                target_delta=(np.array(x['audit']['r2'])-y['audit']['r2']).tolist()))
    target=np.mean([r['target_delta'] for r in pairs],axis=0)
    parts={str(s):float(np.mean([r['mean_delta'] for r in pairs if r['split']==s])) for s in SPLITS}
    wins=sum(r['mean_delta']>0 for r in pairs);delta=float(np.mean([r['mean_delta'] for r in pairs]))
    return dict(direction=a+' minus '+b,mean_delta=delta,target_delta=target.tolist(),partition_deltas=parts,
        partition_wins=sum(v>0 for v in parts.values()),cell_wins=wins,pairs=pairs,
        descriptive_support=bool(delta>0 and min(parts.values())>0 and wins>=6 and min(target)>=0))


def summarize(rows):
    if len(rows)!=90 or len({(r['split'],r['seed'],r['method']) for r in rows})!=90:raise ValueError('Missing/duplicate primary cases')
    methods={}
    for m in METHODS:
        rr=[r for r in rows if r['method']==m]
        methods[m]=dict(mean_r2=float(np.mean([r['audit']['mean_r2'] for r in rr])),
            r2=np.mean([r['audit']['r2'] for r in rr],axis=0).tolist(),rmse=np.mean([r['audit']['rmse'] for r in rr],axis=0).tolist(),
            partition_means={str(s):float(np.mean([r['audit']['mean_r2'] for r in rr if r['split']==s])) for s in SPLITS},
            actual_input_unions=[r['actual_input_union'] for r in rr],total_nn_caps=sum(r['nn_epoch_caps'] for r in rr))
    contrasts={m:paired(rows,'union_mt',m) for m in METHODS if m!='union_mt'}
    return dict(methods=methods,union_mt_minus_reference=contrasts,
        input_visibility_control=paired(rows,'union_st_width','individual_st'),
        primary_support_passed=all(contrasts[m]['descriptive_support'] for m in PRIMARY_REFERENCES),
        independent_generalization_claimed=False,configuration_promoted=False,
        scope='3overlapping historically adapted partitions x3algorithmseeds;not9independent samples/sources')


def evaluate_all(panels):
    seal();lock();write(OUT/'status.json',dict(state='refitting_and_evaluating',current_audit_opened=True))
    old_rows=read(pilot.OUT/'metrics.json');rows=[]
    for split in SPLITS:
        xf,yf,xr,yr=selection_arrays(split);x,y=np.vstack([xf,xr]),np.vstack([yf,yr]);xa,ya,ids=load_audit(split)
        with np.load(OLD/str(split)/'selection_data.npz',allow_pickle=False) as z:pilot.prior.assert_partition(z['fit_ids'],z['reward_ids'],ids)
        for seed in SEEDS:
            for method in METHODS:
                if reused(split,method):
                    previous=next(r for r in old_rows if r['seed']==seed and r['method']==method)
                    path=pilot.OUT/'predictions'/str(seed)/(method+'.npz')
                    with np.load(path,allow_pickle=False) as z:
                        if not np.array_equal(z['target'],ya) or not np.array_equal(z['ids'],ids):raise AssertionError('Reused audit identity')
                        score=metrics(ya,z['prediction']);score['mae']=abs(ya-z['prediction']).mean(0).tolist()
                    if score!=previous['audit']:raise AssertionError('Reused score differs')
                    row=dict(**previous,split=split,reused=True,prediction_file=str(path),nn_parameters=previous['parameters'])
                else:
                    preds=[];params=0;caps=0;selection_time=0.;refit_time=0.
                    for target in tasks(method):
                        folder=folder_for(split,seed,method,target);chosen=read(folder/'selected.json');subset=chosen['subset'];tick=time.perf_counter()
                        if method in NN_METHODS:
                            cols=slice(None) if target is None else slice(target,target+1)
                            t=pilot.base.train(x,y[:,cols],None,None,subset,seed,widths(method),fixed_epochs=chosen['epoch'])
                            pred=nn_predict(t['model'],t['norm'],xa,subset);saved=dict(state=t['model'].state_dict(),norm=t['norm'],subset=subset)
                            updates=t['updates'];params+=chosen['parameters'];caps+=chosen['epochs_run']==2400
                        else:
                            saved=pls_fit(x,y,subset,chosen['components']);pred=pls_predict(saved,xa);updates=0
                        seconds=time.perf_counter()-tick;refit_time+=seconds;selection_time+=chosen['seconds'];preds.append(pred)
                        torch_atomic(folder/'refit_model.pt',saved);write(folder/'refit.json',dict(seconds=seconds,updates=updates,
                            epoch=chosen.get('epoch'),components=chosen.get('components')))
                    pred=np.hstack(preds);score=metrics(ya,pred);score['mae']=abs(ya-pred).mean(0).tolist()
                    path=OUT/str(split)/'predictions'/str(seed)/(method+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,prediction=pred,target=ya,ids=ids)
                    k=len(panels[str(split)][str(seed)]['union']) if method.startswith('union_') or method=='individual_st' else (641 if method.startswith('full_') else 48)
                    row=dict(split=split,seed=seed,method=method,audit=score,reused=False,prediction_file=str(path),
                        actual_input_union=k,per_target_inputs=[16]*3 if method=='individual_st' else [k]*3,
                        logical_selection_pls_fits=0 if method.startswith(('covsel_','full_')) else 12288,
                        nn_parameters=params,nn_epoch_caps=caps,selection_nn_or_pls_tuning_seconds=selection_time,refit_seconds=refit_time)
                if method in PLS_METHODS:row['pls_regression_coefficients_including_intercept']=3*row['actual_input_union']+3
                target=OUT/str(split)/'cases'/str(seed)/(method+'.json');target.parent.mkdir(parents=True,exist_ok=True);write(target,row);rows.append(row)
                emit(f'CASE split={split} seed={seed} method={method} R2={row["audit"]["mean_r2"]:.7f} reused={row["reused"]}')
    write(OUT/'metrics.json',rows);write(OUT/'summary.json',summarize(rows));seal();lock();pilot.lock();pilot.seal()
    write(OUT/'complete.json',dict(utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(OUT/'protocol.json'),
        metrics_sha256=sha(OUT/'metrics.json'),summary_sha256=sha(OUT/'summary.json')))
    write(OUT/'status.json',dict(state='complete',current_audit_opened=True,new_selection_pls_fits=73728,
        new_nn_selections=96,new_nn_refits=96,new_pls_tuning_fits=72,new_pls_refits=18,primary_cases=90))
    emit('COMPLETE all90cases;all same-union controls and2partition replications finished;no promotion')


def main():
    signal.signal(signal.SIGINT,lambda *_:setattr(pilot.base,'STOP',True));prepare()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    try:
        with warnings.catch_warnings(),threadpool_limits(limits=1):
            warnings.simplefilter('error')
            if torch.get_num_threads()!=1:raise RuntimeError('Unexpected effective legacy Torch thread count')
            write(OUT/'execution_environment.json',dict(python_executable=sys.executable,torch_version=torch.__version__,numpy_version=np.__version__,
                requested_torch_threads_before_context=4,actual_torch_threads_inside_context=torch.get_num_threads(),
                actual_interop_threads=torch.get_num_interop_threads(),threadpools=threadpool_info(),
                correction='Historical4-thread labels were requests;legacy global context caps actual Torch at1;retained for exact comparability'))
            write(OUT/'status.json',dict(state='selecting',current_audit_opened=False))
            panels=panels_all();select_all(panels);evaluate_all(panels)
    except BaseException as e:
        previous=read(OUT/'status.json')
        write(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),utc=datetime.now(timezone.utc).isoformat()))
        write(OUT/'status.json',dict(state='stopped_on_problem',current_audit_opened=previous.get('current_audit_opened',False)))
        emit('STOPPED on problem;all artifacts retained;no automatic retries or changes');raise

if __name__=='__main__':main()
