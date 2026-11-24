"""Prespecified single-partition fair input-cap/selection-query pilot.

Training uses a common bounded duration rule fixed before convergence results.
Historical development only. All39NN and9PLS choices precede current audit.
"""
from pathlib import Path
from datetime import datetime,timezone
import sys,signal,time,traceback,warnings
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run_empirical_training_control as base
import run_empirical_training_replication as prior
from empirical_budget_core import greedy,covsel
from hsi_e2e.framework_screen import metrics,nn_predict,pls_fit,pls_predict
from hsi_e2e.formal_runtime import read,write,sha,torch_atomic

OUT=ROOT/'results/empirical_budget_pilot_001'
PREFIX=ROOT/'results/empirical_budget_prefix_audit_001'
DATA=base.OLD/'7117'
SEEDS=(17,29,43)
NN_METHODS=('shared_mt','shared_st_width','shared_st_capacity','individual_st','union_mt','covsel_mt','full_mt')
PLS_METHODS=('shared_pls','covsel_pls','full_pls')


def tasks(method):return [0,1,2] if method in ['shared_st_width','shared_st_capacity','individual_st'] else [None]
def widths(method):return (24,23) if method=='shared_st_capacity' else (64,32)
def folder_for(seed,method,task):return OUT/'models'/str(seed)/method/('joint' if task is None else str(task))


def lock():
    p=read(OUT/'protocol.json')
    for key in ['source_sha256','input_sha256']:
        for file,h in p[key].items():
            if sha(file)!=h:raise RuntimeError('Locked artifact changed:'+file)
    return p


def prepare():
    if OUT.exists():raise RuntimeError('Refuse automatic rerun')
    prior.lock();prior.seal();base.lock()
    convergence=ROOT/'results/empirical_convergence_check_001'
    if not read(convergence/'verification.json')['passed']:raise ValueError('Duration check not verified')
    paths=[Path(__file__),ROOT/'empirical_budget_core.py',ROOT/'run_empirical_training_control.py',
        ROOT/'verify_empirical_budget_pilot.py',ROOT/'tests/test_empirical_budget_core.py',
        ROOT/'tests/test_empirical_budget_pilot.py']+sorted((ROOT/'src/hsi_e2e').glob('*.py'))
    inputs=[DATA/'selection_data.npz',DATA/'audit_data.npz',PREFIX/'subsets.json',PREFIX/'protocol.json',
        convergence/'protocol.json',convergence/'complete.json',convergence/'summary.json',
        ROOT/'results/empirical_public_baseline_review_001/implementation_checks.json',
        ROOT/'results/empirical_public_baseline_review_001/covsel_author_R.txt']
    OUT.mkdir(parents=True)
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),split=7117,seeds=list(SEEDS),
        scope='Posthoc historically adapted development single partition;not independent validation',
        nn_methods=list(NN_METHODS),pls_methods=list(PLS_METHODS),new_nn_selections=39,new_nn_refits=39,
        selection_budget_cap=12288,shared_k=48,individual_k_per_target=16,individual_budget_per_target=4096,
        compute_scope='Selection-query cap only;end-to-end compute differs;PLS1/PLS2 costs differ;tuning/refits counted separately',
        neural_duration=dict(max_epochs=2400,min_epochs=1200,patience=600),
        duration_selection='Fixed in reward-only check protocol before results;no reward/audit-driven per-method changes',
        neural_other_parameters='Unchanged AdamWlr.001/decay.001,batch128,dropout.05,clip5,eval10;Torch4/BLAS1/inter-op1',
        final_pls_candidates=[4,8,12,16],final_pls_rule='Lowest complexity on exact reward meanR2 tie;choose on fit/reward,then refit1344',
        covsel='X center only,Y fit sd ddof0,equal row weights;selection itself0PLSfits;mathematical Python implementation checked against author projection rule',
        individual_rule='Equal16per-target quota,not optimal budget allocation;actual union<=48must be reported',
        evaluation='Seal39NN plus9PLS choices and selection provenance before audit loader',
        stopping='Any warning,nonfinite,integrity failure or exception stops;no automatic repair,rerun,extra seeds or tuning',
        interpretation='Report all targets and all3seeds;no promotion from this one split;no fully converged or hardware-cost claim',
        source_sha256={str(f):sha(f) for f in paths},input_sha256={str(f):sha(f) for f in inputs})
    write(OUT/'protocol.json',p);write(OUT/'status.json',dict(state='prepared',audit_opened=False))
    return p


def emit(s):
    line=datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+s
    print(line,flush=True)
    with (OUT/'run.log').open('a',encoding='utf-8') as f:f.write(line+'\n')


def selection_arrays():
    with np.load(DATA/'selection_data.npz',allow_pickle=False) as z:
        arrays=[z[k].copy() for k in ['X_fit','y_fit','X_reward','y_reward']]
        prior.assert_partition(z['fit_ids'],z['reward_ids'])
    return arrays


def select_panels(arrays):
    xf,yf,xr,yr=arrays
    common=covsel(xf,yf,48);write(OUT/'covsel_selection.json',common)
    prefixes=read(PREFIX/'subsets.json');panels={}
    for seed in SEEDS:
        shared=next(r for r in prefixes if r['split']==7117 and r['seed']==seed)
        if shared['budget']!=12288 or shared['best_paid_index']>12288:raise ValueError('Invalid paid prefix')
        own=[]
        for task in range(3):
            state=greedy(xf,yf[:,task:task+1],xr,yr[:,task:task+1],seed=seed,k=16,budget=4096,emit=emit)
            folder=OUT/'search'/str(seed)/str(task);folder.mkdir(parents=True)
            write(folder/'ledger.json',state.ev.ledger);write(folder/'decisions.json',state.decisions)
            write(folder/'summary.json',state.summary());own.append(list(state.ev.best_subset))
            emit(f'PANEL seed={seed} target={task} fits={state.ev.fit_count} complete')
        union=sorted(set().union(*map(set,own)))
        if not len(union)<=48 or len(shared['subset'])!=48:raise AssertionError('Shared input cap')
        panels[str(seed)]=dict(shared=shared['subset'],individual=own,union=union,covsel=common['subset'],
                              full=list(range(641)),actual_union_size=len(union))
    write(OUT/'panels.json',panels)
    return panels


def subset_for(panels,seed,method,task):
    row=panels[str(seed)]
    if method=='individual_st':return row['individual'][task]
    if method.startswith('shared_'):return row['shared']
    if method.startswith('covsel_'):return row['covsel']
    if method.startswith('full_'):return row['full']
    if method=='union_mt':return row['union']
    raise ValueError(method)


def select_models(arrays,panels,p):
    xf,yf,xr,yr=arrays
    for seed in SEEDS:
        for method in NN_METHODS:
            for task in tasks(method):
                cols=slice(None) if task is None else slice(task,task+1)
                subset=subset_for(panels,seed,method,task)
                trained=base.train(xf,yf[:,cols],xr,yr[:,cols],subset,seed,widths(method),**p['neural_duration'])
                folder=folder_for(seed,method,task);folder.mkdir(parents=True)
                torch_atomic(folder/'selected_model.pt',dict(state=trained['model'].state_dict(),norm=trained['norm'],subset=subset))
                write(folder/'history.json',trained['history'])
                fields={k:trained[k] for k in ['epoch','epochs_run','selection','updates','seconds']}
                write(folder/'selected.json',dict(seed=seed,method=method,task=task,subset=subset,widths=widths(method),
                    parameters=sum(v.numel() for v in trained['model'].parameters()),**fields))
                emit(f'NN SELECT seed={seed} method={method} task={task} epoch={trained["epoch"]}/{trained["epochs_run"]}')
        for method in PLS_METHODS:
            subset=subset_for(panels,seed,method,None);trials=[];best=None;tick=time.perf_counter()
            for n in p['final_pls_candidates']:
                model=pls_fit(xf,yf,subset,n);score=metrics(yr,pls_predict(model,xr));trials.append(dict(components=n,**score))
                if best is None or score['mean_r2']>best[1]['mean_r2']:best=(model,score)
            folder=folder_for(seed,method,None);folder.mkdir(parents=True)
            torch_atomic(folder/'selected_model.pt',best[0]);write(folder/'trials.json',trials)
            write(folder/'selected.json',dict(seed=seed,method=method,subset=subset,components=best[0]['components'],
                selection=best[1],tuning_fits=4,seconds=time.perf_counter()-tick))
    seal();lock();emit('ALL39NN+9PLS CHOICES SEALED before current audit')


def seal():
    files=[OUT/'panels.json',OUT/'covsel_selection.json']
    for seed in SEEDS:
        for method in [*NN_METHODS,*PLS_METHODS]:
            for task in tasks(method):
                folder=folder_for(seed,method,task)
                files += [folder/'selected.json',folder/'selected_model.pt',
                          folder/('trials.json' if method in PLS_METHODS else 'history.json')]
    signatures={str(f):sha(f) for f in files};marker=OUT/'choices_locked.json'
    if marker.exists():
        if read(marker)['files']!=signatures:raise RuntimeError('Sealed choices changed')
    else:write(marker,dict(utc=datetime.now(timezone.utc).isoformat(),nn_models=39,pls_models=9,files=signatures))


def load_audit():
    if not (OUT/'choices_locked.json').exists():raise RuntimeError('Unsealed audit access')
    seal();lock()
    with np.load(DATA/'audit_data.npz',allow_pickle=False) as z:return tuple(z[k].copy() for k in ['X_audit','y_audit','audit_ids'])


def evaluate(arrays,panels):
    xa,ya,ids=load_audit();xf,yf,xr,yr=arrays;x,y=np.vstack([xf,xr]),np.vstack([yf,yr])
    with np.load(DATA/'selection_data.npz',allow_pickle=False) as z:prior.assert_partition(z['fit_ids'],z['reward_ids'],ids)
    write(OUT/'status.json',dict(state='refitting',audit_opened=True));rows=[]
    for seed in SEEDS:
        for method in [*NN_METHODS,*PLS_METHODS]:
            preds=[];selection_seconds=0.;refit_seconds=0.;parameters=0;caps=0
            for task in tasks(method):
                folder=folder_for(seed,method,task);selected=read(folder/'selected.json');subset=selected['subset']
                selection_seconds+=selected['seconds'];tick=time.perf_counter()
                if method in NN_METHODS:
                    cols=slice(None) if task is None else slice(task,task+1)
                    t=base.train(x,y[:,cols],None,None,subset,seed,widths(method),fixed_epochs=selected['epoch'])
                    pred=nn_predict(t['model'],t['norm'],xa,subset)
                    saved=dict(state=t['model'].state_dict(),norm=t['norm'],subset=subset)
                    updates=t['updates'];parameters+=selected['parameters'];caps+=selected['epochs_run']==2400
                else:
                    saved=pls_fit(x,y,subset,selected['components']);pred=pls_predict(saved,xa);updates=0
                elapsed=time.perf_counter()-tick;refit_seconds+=elapsed;preds.append(pred)
                torch_atomic(folder/'refit_model.pt',saved);write(folder/'refit.json',dict(seconds=elapsed,updates=updates,
                    epoch=selected.get('epoch'),components=selected.get('components')))
            prediction=np.hstack(preds);score=metrics(ya,prediction);score['mae']=abs(ya-prediction).mean(0).tolist()
            folder=OUT/'predictions'/str(seed);folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/(method+'.npz'),prediction=prediction,target=ya,ids=ids)
            individual=method in ['individual_st','union_mt']
            input_count=panels[str(seed)]['actual_union_size'] if individual else (641 if method.startswith('full') else 48)
            query_count=0 if method.startswith(('covsel','full')) else 12288
            rows.append(dict(seed=seed,method=method,audit=score,actual_input_union=input_count,
                per_target_inputs=[16]*3 if method=='individual_st' else [input_count]*3,
                logical_selection_pls_fits=query_count,parameters=parameters,
                nn_epoch_caps=caps,selection_nn_or_pls_tuning_seconds=selection_seconds,refit_seconds=refit_seconds))
            emit(f'AUDIT seed={seed} method={method} meanR2={score["mean_r2"]:.7f} input_union={input_count}')
    write(OUT/'metrics.json',rows)
    summary={m:dict(mean_r2=float(np.mean([r['audit']['mean_r2'] for r in rows if r['method']==m])),
        r2=np.mean([r['audit']['r2'] for r in rows if r['method']==m],axis=0).tolist(),
        rmse=np.mean([r['audit']['rmse'] for r in rows if r['method']==m],axis=0).tolist(),
        actual_unions=[r['actual_input_union'] for r in rows if r['method']==m],
        total_nn_caps=sum(r['nn_epoch_caps'] for r in rows if r['method']==m)) for m in [*NN_METHODS,*PLS_METHODS]}
    write(OUT/'summary.json',dict(methods=summary,scope='Single historical development partition7117 x3seeds;not independent validation',configuration_promoted=False))
    seal();lock();prior.lock();base.lock()
    write(OUT/'complete.json',dict(protocol_sha256=sha(OUT/'protocol.json'),metrics_sha256=sha(OUT/'metrics.json'),summary_sha256=sha(OUT/'summary.json')))
    write(OUT/'status.json',dict(state='complete',audit_opened=True,new_search_fits=36864,
        new_nn_selections=39,new_nn_refits=39,final_pls_tuning_fits=36,final_pls_refits=9))
    emit('COMPLETE all30pipeline cases;no new-source test scores or protocol promotion')


def main():
    signal.signal(signal.SIGINT,lambda *_:setattr(base,'STOP',True))
    p=prepare();torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    try:
        with warnings.catch_warnings(),threadpool_limits(limits=1):
            warnings.simplefilter('error');arrays=selection_arrays()
            write(OUT/'status.json',dict(state='selecting',audit_opened=False))
            panels=select_panels(arrays);select_models(arrays,panels,p);evaluate(arrays,panels)
    except BaseException as e:
        write(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));emit('STOPPED on problem;no automatic changes');raise

if __name__=='__main__':main()
