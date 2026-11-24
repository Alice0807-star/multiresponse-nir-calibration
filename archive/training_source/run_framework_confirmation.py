"""Locked multi-split replication of the selected shared-band MLP pipelines.

Only historically adapted development data are used. No outer formal/external
labels are loaded. All 45 NN selections must be locked before any audit opens.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, json, shutil, signal, sys, time, traceback

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from hsi_e2e.framework_screen import PolicySearch, train_nn, nn_predict, metrics
from hsi_e2e.component_evaluator import ComponentEvaluator
from hsi_e2e.variable_k import VariableEnsemble
from hsi_e2e.formal_runtime import (read,sha,write,archive,completed,save_active,
    restore_active,drive,GracefulStop,torch_atomic)

OUT=ROOT/'results/framework_confirmation_001'
DATA=ROOT/'data/development_v1/development_only.npz'
METHODS=['adaptive','frozen','diverse','greedy']
PREDICTORS=METHODS+['full']
STOP=False


def emit(message):
    line=datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+message
    print(line,flush=True)
    if OUT.exists():
        with (OUT/'run.log').open('a',encoding='utf-8') as f:f.write(line+'\n')


def split_indices(n,seed):
    if n!=1680:raise ValueError('Unexpected development sample count')
    order=np.random.default_rng(seed).permutation(n)
    return tuple(np.sort(s) for s in np.split(order,[1008,1344]))


def assert_partition(fit,reward,audit):
    groups=[set(a) for a in [fit,reward,audit]]
    if [len(a) for a in groups]!=[1008,336,336]:raise ValueError('Partition sizes')
    if any(groups[i]&groups[j] for i,j in [(0,1),(0,2),(1,2)]):raise ValueError('Partition overlap')
    if len(set.union(*groups))!=1680:raise ValueError('Partition coverage')


def prepare():
    if (OUT/'protocol.json').exists():return lock()
    if OUT.exists():raise RuntimeError('Output exists without a locked protocol')
    OUT.mkdir(parents=True)
    c=dict(partitions=[7117,8111,9103],seeds=[17,29,43],k=48,components=8,
        budget=12800,warm=800,warm_updates=512,torch_threads=4,blas_threads=1)
    sources=[Path(__file__).resolve(),ROOT/'tests/test_framework_confirmation.py']
    sources+=sorted((ROOT/'src/hsi_e2e').glob('*.py'))
    snapshot=OUT/'code_snapshot';snapshot.mkdir()
    for path in sources:shutil.copy2(path,snapshot/path.name)
    preserved=[]
    for folder,names in [('formal_v3_2026-11-06',['outer_metrics.json','report.md','evaluation_complete.json']),
        ('framework_serial_001',['audit_metrics.json','summary.json','protocol.json']),
        ('framework_matched_mlp_001',['metrics.json','summary.json','protocol.json'])]:
        preserved.extend(ROOT/'results'/folder/name for name in names)
    with np.load(DATA,allow_pickle=False) as z:
        x=np.vstack([z['X_fit'],z['X_reward']]);y=np.vstack([z['y_fit'],z['y_reward']])
        ids=np.r_[z['fit_ids'],z['reward_ids']];names=z['target_names']
    if x.shape!=(1680,641) or y.shape!=(1680,3) or len(set(ids))!=1680:raise ValueError('Unexpected development data')
    inputs={str(DATA):sha(DATA)}
    for split in c['partitions']:
        fit,reward,audit=split_indices(len(x),split);assert_partition(ids[fit],ids[reward],ids[audit])
        folder=OUT/str(split);folder.mkdir()
        np.savez_compressed(folder/'selection_data.npz',X_fit=x[fit],y_fit=y[fit],
            X_reward=x[reward],y_reward=y[reward],fit_ids=ids[fit],reward_ids=ids[reward],target_names=names)
        np.savez_compressed(folder/'audit_data.npz',X_audit=x[audit],y_audit=y[audit],
            audit_ids=ids[audit],target_names=names)
        for name in ['selection_data.npz','audit_data.npz']:inputs[str(folder/name)]=sha(folder/name)
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),config=c,methods=METHODS,
        scope='Three new random partitions of historically adapted development1680;not independent data;overlapping audit rows across partitions',
        predictor='Unchanged shared64/GELU/dropout.05/32/GELU with3linearheads;fit-only scaling;normalizedMSE;AdamWlr.001/decay.001;batch128;clip5;max1200/min400/patience300/evalevery10;fit+reward refit for locked best epoch',
        selection='Unchanged original attention/frozen/diverse/greedy;PLS8 meanR2 reward;K48;actual12800 unique queries;warm800 shared physically and charged to each nongreedy;no new K/model/hyperparameter selection',
        gate='Descriptive replication:adaptive vs each frozen/diverse/greedy;overall mean delta>0,all3 partition mean deltas>0,>=6/9 paired-cell wins,each overall target mean delta>=0;no statistical independence or significance claim',
        stopping='No performance earlystop for fixed12800 search;unchanged broad NN patience;any exception/warning/nonfinite stops entire run and preserves diagnostics;no dropped cells',
        evaluation='All45 NN checkpoints/epochs/subsets locked globally before any audit loader;MLP and paired PLS8 refits use1344 rows;audit never drives stopping or choices',
        cost='36 logical searches460800 queries;warm reused once per partition/seed;physical446400 queries;NN epochs/updates and pipeline time reported separately',
        source_sha256={str(f):sha(f) for f in sources},input_sha256=inputs,
        preserved_sha256={str(f):sha(f) for f in preserved})
    write(OUT/'protocol.json',p);write(OUT/'status.json',dict(state='prepared',audit_opened=False))
    emit('PREPARED3newpartitions x3pairedseeds x4selectors;all MLPhyperparameters fixed')
    return p


def lock():
    p=read(OUT/'protocol.json')
    for group in ['source_sha256','input_sha256','preserved_sha256']:
        for path,value in p[group].items():
            if sha(path)!=value:raise RuntimeError('Locked file changed:'+path)
    return p


def paths(p):
    return [OUT/str(split)/'models'/str(seed)/method/'selected.json'
        for split in p['config']['partitions'] for seed in p['config']['seeds'] for method in PREDICTORS]


def seal_choices(p):
    selected=paths(p)
    if any(not path.exists() for path in selected):raise RuntimeError('Missing selected pipeline;refuse audit')
    files={str(path):sha(path) for path in selected}
    files.update({str(path.parent/'selected_model.pt'):sha(path.parent/'selected_model.pt') for path in selected})
    marker=OUT/'choices_locked.json'
    if marker.exists():
        if read(marker)['files']!=files:raise RuntimeError('Sealed choices changed')
    else:write(marker,dict(utc=datetime.now(timezone.utc).isoformat(),files=files,count=len(selected)))


def audit_data(split):
    if not (OUT/'choices_locked.json').exists():raise RuntimeError('Refuse audit before global choice seal')
    for path,value in read(OUT/'choices_locked.json')['files'].items():
        if sha(path)!=value:raise RuntimeError('Choice seal mismatch')
    with np.load(OUT/str(split)/'audit_data.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def select(p):
    c=p['config']
    for split in c['partitions']:
        with np.load(OUT/str(split)/'selection_data.npz',allow_pickle=False) as z:arrays=[z[k] for k in ['X_fit','y_fit','X_reward','y_reward']]
        for seed in c['seeds']:
            warm=None
            for method in METHODS:
                if STOP:raise GracefulStop('Operator requested stop')
                folder=OUT/str(split)/'search'/str(seed)/method
                if (folder/'selected.json').exists():continue
                identity=dict(stage='replication',unit=str(split),seed=seed,k=48,variant=method,
                    budget=c['budget'],protocol_sha256=sha(OUT/'protocol.json'))
                if (folder/'complete.json').exists():summary=completed(folder,identity)
                else:
                    if (folder/'active_manifest.json').exists():state=restore_active(folder,identity)
                    elif method=='greedy':
                        start=time.perf_counter();state=PolicySearch(ComponentEvaluator(*arrays,c['budget'],k=48,components=8),seed,'greedy')
                        state.initialize();state.elapsed_seconds=time.perf_counter()-start;save_active(folder,state,identity)
                    else:
                        if warm is None:
                            wf=OUT/str(split)/'search'/str(seed)/'warm';wi={**identity,'variant':'warm','budget':800}
                            if (wf/'active_manifest.json').exists():warm=restore_active(wf,wi)
                            else:
                                start=time.perf_counter();warm=PolicySearch(ComponentEvaluator(*arrays,800,k=48,components=8),seed,'warm')
                                warm.initialize();warm.elapsed_seconds=time.perf_counter()-start;save_active(wf,warm,wi)
                            drive(warm,800,wf,wi,lambda:STOP,emit);save_active(wf,warm,wi)
                        start=time.perf_counter();state=copy.deepcopy(warm);state.arm=method;state.ev.budget=c['budget']
                        state.warm_groups=[]
                        if method!='diverse':
                            state.learner=VariableEnsemble(warm.warm_groups,k=48,adaptive=method=='adaptive')
                            for _ in range(c['warm_updates']):state.learner.update()
                        state.elapsed_seconds+=time.perf_counter()-start;save_active(folder,state,identity)
                    emit(f'SEARCH START split={split} seed={seed} method={method}')
                    drive(state,c['budget'],folder,identity,lambda:STOP,emit)
                    if state.ev.fit_count!=12800 or state.ev.cache_hits:raise RuntimeError('Paid-query identity')
                    summary=archive(folder,state,identity,OUT);del state
                row=dict(split=split,seed=seed,method=method,subset=summary['best']['subset'],
                    selection={k:summary['best'][k] for k in ['r2','mean_r2','rmse']},
                    fits=12800,seconds=summary['seconds'])
                write(folder/'selected.json',row);emit(f'SEARCH COMPLETE split={split} seed={seed} method={method}')
            del warm
            for method in PREDICTORS:
                folder=OUT/str(split)/'models'/str(seed)/method
                if (folder/'selected.json').exists():continue
                if STOP:raise GracefulStop('Operator requested stop')
                selector=None if method=='full' else read(OUT/str(split)/'search'/str(seed)/method/'selected.json')
                subset=list(range(641)) if selector is None else selector['subset']
                emit(f'NN START split={split} seed={seed} method={method}')
                trained=train_nn(*arrays,subset,seed)
                folder.mkdir(parents=True,exist_ok=True)
                torch_atomic(folder/'selected_model.pt',dict(state=trained['model'].state_dict(),norm=trained['norm'],subset=subset))
                write(folder/'history.json',trained['history'])
                write(folder/'selected.json',dict(split=split,seed=seed,method=method,subset=subset,
                    epoch=trained['epoch'],epochs_run=trained['epochs_run'],updates=trained['updates'],
                    nn_seconds=trained['seconds'],selector_seconds=0 if selector is None else selector['seconds'],selection=trained['selection']))
                emit(f'NN COMPLETE split={split} seed={seed} method={method} bestepoch={trained["epoch"]} stop={trained["epochs_run"]}')
    lock();seal_choices(p)


def evaluate(p):
    from hsi_e2e.framework_screen import pls_fit,pls_predict
    if not (OUT/'choices_locked.json').exists():raise RuntimeError('Choices are not sealed')
    write(OUT/'status.json',dict(state='evaluating',audit_opened=True));rows=[]
    for split in p['config']['partitions']:
        with np.load(OUT/str(split)/'selection_data.npz',allow_pickle=False) as z:
            x=np.vstack([z['X_fit'],z['X_reward']]);y=np.vstack([z['y_fit'],z['y_reward']])
        audit=audit_data(split);xa,ya=audit['X_audit'],audit['y_audit']
        for seed in p['config']['seeds']:
            for method in PREDICTORS:
                folder=OUT/str(split)/'models'/str(seed)/method
                if (folder/'audit.json').exists():rows.append(read(folder/'audit.json'));continue
                if STOP:raise GracefulStop('Operator requested stop')
                chosen=read(folder/'selected.json');start=time.perf_counter()
                trained=train_nn(x,y,None,None,chosen['subset'],seed,fixed_epochs=chosen['epoch'])
                pred=nn_predict(trained['model'],trained['norm'],xa,chosen['subset'])
                row=dict(**chosen,audit=metrics(ya,pred),refit_seconds=time.perf_counter()-start,refit_updates=trained['updates'])
                torch_atomic(folder/'refit_model.pt',dict(state=trained['model'].state_dict(),norm=trained['norm'],subset=chosen['subset']))
                np.savez_compressed(folder/'predictions.npz',prediction=pred,target=ya,ids=audit['audit_ids'])
                if method!='full':
                    fitted=pls_fit(x,y,chosen['subset'],8);pp=pls_predict(fitted,xa)
                    row['pls8_audit']=metrics(ya,pp);np.savez_compressed(folder/'pls8_predictions.npz',prediction=pp,target=ya)
                write(folder/'audit.json',row);rows.append(row)
                emit(f'AUDIT split={split} seed={seed} method={method} mean={row["audit"]["mean_r2"]:.7f}')
    write(OUT/'metrics.json',rows);lock();seal_choices(p)
    write(OUT/'evaluation_complete.json',dict(utc=datetime.now(timezone.utc).isoformat(),cases=len(rows),metrics_sha256=sha(OUT/'metrics.json')))
    write(OUT/'status.json',dict(state='complete',audit_opened=True,cases=len(rows)))
    emit('ALL3PARTITIONS COMPLETE45MLPcases;no unplanned method or tuning')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    signal.signal(signal.SIGINT,lambda *_:globals().__setitem__('STOP',True))
    p=prepare()
    if args.prepare_only:return
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    try:
        with threadpool_limits(limits=1):
            write(OUT/'status.json',dict(state='selecting',audit_opened=False));select(p);evaluate(p)
    except GracefulStop as exc:
        write(OUT/'stopped.json',dict(reason=str(exc),utc=datetime.now(timezone.utc).isoformat()));emit('STOPPED safely;preserved complete-batch checkpoints');raise
    except BaseException as exc:
        write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),utc=datetime.now(timezone.utc).isoformat()));emit('FAILED;do not rerun blindly');raise


if __name__=='__main__':main()
