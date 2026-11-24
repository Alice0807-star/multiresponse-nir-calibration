"""Complete two prespecified same-band joint/ST development partitions.

The original trainer and all historical artifacts stay frozen. No selection,
hyperparameter sweep, external data, or audit-driven extension is performed.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import signal
import sys
import time
import traceback
import warnings

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run_empirical_training_control as base
from hsi_e2e.framework_screen import metrics,nn_predict
from hsi_e2e.formal_runtime import read,write,sha,torch_atomic

OUT=ROOT/'results/empirical_training_replication_001'
OLD=base.OLD
FIRST=base.OUT
NEW_SPLITS=(8111,9103)
ALL_SPLITS=(7117,8111,9103)
SEEDS=base.SEEDS
ARMS=base.ARMS
STOP=False


def assert_partition(fit,reward,audit=None):
    values=[np.asarray(fit),np.asarray(reward)]
    expected=[1008,336]
    if audit is not None:values.append(np.asarray(audit));expected.append(336)
    sets=[set(a.tolist()) for a in values]
    if [len(a) for a in values]!=expected or [len(a) for a in sets]!=expected:
        raise ValueError('Unexpected sizes or duplicate IDs')
    if any(sets[i]&sets[j] for i in range(len(sets)) for j in range(i+1,len(sets))):
        raise ValueError('Within-partition sample overlap')


def sources():
    return [Path(__file__),ROOT/'verify_empirical_training_replication.py',
            ROOT/'tests/test_empirical_training_replication.py',
            ROOT/'run_empirical_training_control.py']+sorted((ROOT/'src/hsi_e2e').glob('*.py'))


def inputs():
    files=[FIRST/name for name in ['protocol.json','metrics.json','summary.json',
          'verification.json','complete.json','choices_locked.json']]
    files += [OLD/name for name in ['protocol.json','metrics.json','summary.json','verification.json']]
    for split in ALL_SPLITS:
        files += [OLD/str(split)/name for name in ['selection_data.npz','audit_data.npz']]
        for seed in SEEDS:
            folder=OLD/str(split)/'models'/str(seed)/'greedy'
            files += [folder/name for name in ['selected.json','selected_model.pt','refit_model.pt',
                       'predictions.npz','audit.json']]
            if split==7117:
                for arm in ARMS:
                    files += [FIRST/str(seed)/arm/name for name in ['predictions.npz','audit.json']]
                    for task in range(3):
                        files += [FIRST/str(seed)/arm/str(task)/name for name in
                            ['selected.json','selected_model.pt','history.json','refit_model.pt','refit.json']]
    return files


def prepare():
    if (OUT/'protocol.json').exists():return lock()
    if OUT.exists():raise RuntimeError('Output exists without protocol')
    base.lock();base.seal()
    import run_framework_confirmation as old
    import run_shared_warm_freeze as previous
    old.lock();previous.lock()
    if not read(FIRST/'verification.json')['passed']:raise ValueError('First control not verified')
    OUT.mkdir(parents=True)
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),new_partitions=list(NEW_SPLITS),
        all_partitions=list(ALL_SPLITS),seeds=list(SEEDS),k=48,arms=ARMS,
        config=dict(max_epochs=1200,min_epochs=400,patience=300,eval_every=10,
                    lr=.001,weight_decay=.001,batch_size=128,dropout=.05,clip_grad=5.,
                    torch_threads=4,torch_interop_threads=1,blas_threads=1),
        training_implementation='Direct call to frozen run_empirical_training_control.train',
        selection='Each target own reward R2 checkpoint;all36new choices sealed globally before any new audit loader',
        reuse='7117 independent results and all9joint references reused byte-for-byte,no repeated selection search',
        sample_scope='Historically adapted1680samples;3overlapping partitions,3algorithmseeds;not9independent datasets',
        decision_rule='Descriptive fixed-budget support vs each ST control:mean delta>0,all3partition means>0,>=6of9cell wins,all3overall target means>=0;not statistical significance or automatic promotion',
        preregistration_limit='7117control already known;this completion protocol is fixed before reading new8111/9103ST audit scores',
        convergence='Record1200epoch caps,best epochs,and late reward improvements;no extension based on audit scores or claim of full convergence',
        stopping='Any exception,warning,nonfinite or integrity failure stops full run;no automatic rerun,case dropping,or parameter repair',
        new_selection_models=36,new_fixed_epoch_refits=36,new_band_search_fits=0,
        source_sha256={str(f):sha(f) for f in sources()},input_sha256={str(f):sha(f) for f in inputs()})
    write(OUT/'protocol.json',p)
    write(OUT/'status.json',dict(state='prepared',current_audit_opened=False))
    return p


def lock():
    p=read(OUT/'protocol.json')
    for key in ['source_sha256','input_sha256']:
        for path,digest in p[key].items():
            if sha(path)!=digest:raise RuntimeError('Locked artifact changed:'+path)
    return p


def selected_paths():
    return [OUT/str(split)/str(seed)/arm/str(task)/'selected.json'
            for split in NEW_SPLITS for seed in SEEDS for arm in ARMS for task in range(3)]


def seal():
    paths=selected_paths()
    if any(not p.exists() for p in paths):raise RuntimeError('Missing model selections;refuse audit')
    files={str(p):sha(p) for p in paths}
    files.update({str(p.parent/'selected_model.pt'):sha(p.parent/'selected_model.pt') for p in paths})
    files.update({str(p.parent/'history.json'):sha(p.parent/'history.json') for p in paths})
    marker=OUT/'choices_locked.json'
    if marker.exists():
        if read(marker)['files']!=files:raise RuntimeError('Sealed selections changed')
    else:write(marker,dict(utc=datetime.now(timezone.utc).isoformat(),count=36,files=files))


def load_audit(split):
    if not (OUT/'choices_locked.json').exists():raise RuntimeError('Audit requires global choices seal')
    seal();lock()
    with np.load(OLD/str(split)/'audit_data.npz',allow_pickle=False) as z:
        return tuple(z[k].copy() for k in ['X_audit','y_audit','audit_ids'])


def emit(message):
    line=datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+message
    print(line,flush=True)
    with (OUT/'run.log').open('a',encoding='utf-8') as stream:stream.write(line+'\n')


def select_all():
    write(OUT/'status.json',dict(state='selecting',current_audit_opened=False))
    for split in NEW_SPLITS:
        with np.load(OLD/str(split)/'selection_data.npz',allow_pickle=False) as z:
            xf,yf,xr,yr=[z[k].copy() for k in ['X_fit','y_fit','X_reward','y_reward']]
            assert_partition(z['fit_ids'],z['reward_ids'])
        if xf.shape!=(1008,641) or yf.shape!=(1008,3) or xr.shape!=(336,641) or yr.shape!=(336,3):
            raise ValueError('Unexpected development array shapes')
        if any(not np.isfinite(a).all() for a in [xf,yf,xr,yr]):raise ValueError('Nonfinite input')
        for seed in SEEDS:
            subset=read(OLD/str(split)/'models'/str(seed)/'greedy/selected.json')['subset']
            if len(subset)!=48 or len(set(subset))!=48:raise ValueError('Band identity')
            for arm,widths in ARMS.items():
                for task in range(3):
                    if STOP:raise InterruptedError('Operator requested stop')
                    folder=OUT/str(split)/str(seed)/arm/str(task)
                    if folder.exists():raise RuntimeError('Existing model directory;review before any restart')
                    trained=base.train(xf,yf[:,task:task+1],xr,yr[:,task:task+1],subset,seed,widths)
                    folder.mkdir(parents=True)
                    torch_atomic(folder/'selected_model.pt',dict(state=trained['model'].state_dict(),
                                 norm=trained['norm'],subset=subset))
                    write(folder/'history.json',trained['history'])
                    fields={k:trained[k] for k in ['epoch','epochs_run','selection','updates','seconds']}
                    write(folder/'selected.json',dict(split=split,seed=seed,arm=arm,task=task,
                          subset=subset,widths=widths,**fields))
                    emit(f'SELECT split={split} seed={seed} arm={arm} target={task} best={trained["epoch"]} run={trained["epochs_run"]}')
    seal();lock()
    emit('ALL36SELECTIONS SEALED before current audit load')


def summarize(rows):
    methods={};contrasts={}
    for arm in ['joint_reference',*ARMS]:
        cases=[r for r in rows if r['arm']==arm]
        if len(cases)!=9 or len({(r['split'],r['seed']) for r in cases})!=9:
            raise ValueError('Missing/duplicate paired cases')
        methods[arm]=dict(mean_r2=float(np.mean([r['audit']['mean_r2'] for r in cases])),
            r2=np.mean([r['audit']['r2'] for r in cases],axis=0).tolist(),
            rmse=np.mean([r['audit']['rmse'] for r in cases],axis=0).tolist(),
            partition_means={str(s):float(np.mean([r['audit']['mean_r2'] for r in cases if r['split']==s])) for s in ALL_SPLITS})
    for arm in ARMS:
        target=np.array(methods['joint_reference']['r2'])-methods[arm]['r2']
        partitions={str(s):methods['joint_reference']['partition_means'][str(s)]-methods[arm]['partition_means'][str(s)] for s in ALL_SPLITS}
        pairs=[]
        for split in ALL_SPLITS:
            for seed in SEEDS:
                a=next(r for r in rows if (r['split'],r['seed'],r['arm'])==(split,seed,'joint_reference'))
                b=next(r for r in rows if (r['split'],r['seed'],r['arm'])==(split,seed,arm))
                pairs.append(dict(split=split,seed=seed,mean_delta=a['audit']['mean_r2']-b['audit']['mean_r2'],
                                  target_delta=(np.array(a['audit']['r2'])-b['audit']['r2']).tolist()))
        wins=sum(p['mean_delta']>0 for p in pairs)
        mean_delta=float(target.mean())
        contrasts[arm]=dict(direction='joint minus independent',mean_delta=mean_delta,
            target_delta=target.tolist(),partition_deltas=partitions,partition_wins=sum(v>0 for v in partitions.values()),
            cell_wins=wins,pairs=pairs,
            descriptive_fixed_budget_support=bool(mean_delta>0 and min(partitions.values())>0 and wins>=6 and min(target)>=0))
    return dict(methods=methods,joint_minus_st=contrasts,configuration_promoted=False,
        independent_generalization_claimed=False,
        scope='Three overlapping historically adapted development partitions x3algorithmseeds;no independent-source evaluation')


def evaluate_all():
    seal();lock()
    write(OUT/'status.json',dict(state='refitting_and_evaluating',current_audit_opened=True))
    rows=[dict(**row,split=7117,reused=True) for row in read(FIRST/'metrics.json')]
    for split in NEW_SPLITS:
        with np.load(OLD/str(split)/'selection_data.npz',allow_pickle=False) as z:
            xf,yf,xr,yr=[z[k].copy() for k in ['X_fit','y_fit','X_reward','y_reward']]
            fit_ids,reward_ids=z['fit_ids'].copy(),z['reward_ids'].copy()
        xa,ya,ids=load_audit(split);assert_partition(fit_ids,reward_ids,ids)
        x,y=np.vstack([xf,xr]),np.vstack([yf,yr])
        for seed in SEEDS:
            reference=OLD/str(split)/'models'/str(seed)/'greedy'
            with np.load(reference/'predictions.npz',allow_pickle=False) as z:
                if not np.array_equal(z['target'],ya) or not np.array_equal(z['ids'],ids):raise ValueError('Reference IDs/targets')
                reference_score=metrics(ya,z['prediction'])
            if not np.allclose(reference_score['r2'],read(reference/'audit.json')['audit']['r2'],rtol=0,atol=1e-14):
                raise ValueError('Reference score')
            rows.append(dict(split=split,seed=seed,arm='joint_reference',audit=reference_score,reused=True))
            for arm,widths in ARMS.items():
                predictions=[];costs=[]
                for task in range(3):
                    if STOP:raise InterruptedError('Operator requested stop')
                    folder=OUT/str(split)/str(seed)/arm/str(task);selected=read(folder/'selected.json')
                    trained=base.train(x,y[:,task:task+1],None,None,selected['subset'],seed,widths,
                                       fixed_epochs=selected['epoch'])
                    predictions.append(nn_predict(trained['model'],trained['norm'],xa,selected['subset']))
                    torch_atomic(folder/'refit_model.pt',dict(state=trained['model'].state_dict(),
                                 norm=trained['norm'],subset=selected['subset']))
                    write(folder/'refit.json',dict(epoch=trained['epoch'],updates=trained['updates'],
                          seconds=trained['seconds'],selection_seconds=selected['seconds']))
                    costs.append(dict(selection_seconds=selected['seconds'],refit_seconds=trained['seconds']))
                prediction=np.hstack(predictions);folder=OUT/str(split)/str(seed)/arm
                np.savez_compressed(folder/'predictions.npz',prediction=prediction,target=ya,ids=ids)
                row=dict(split=split,seed=seed,arm=arm,audit=metrics(ya,prediction),reused=False,training_costs=costs)
                write(folder/'audit.json',row);rows.append(row)
                emit(f'AUDIT split={split} seed={seed} arm={arm} meanR2={row["audit"]["mean_r2"]:.7f}')
    write(OUT/'metrics.json',rows)
    write(OUT/'summary.json',summarize(rows))
    lock();seal()
    write(OUT/'complete.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(OUT/'protocol.json'),metrics_sha256=sha(OUT/'metrics.json'),summary_sha256=sha(OUT/'summary.json')))
    write(OUT/'status.json',dict(state='complete',current_audit_opened=True,
        new_selected_models=36,new_refits=36,reused_first_split_models=18,reused_joint_cases=9))
    emit('COMPLETE 3partitions x3seeds all27cases retained;no new band searches or tuning')


def main():
    def stop(*_):
        global STOP
        STOP=True;base.STOP=True
    signal.signal(signal.SIGINT,stop)
    if OUT.exists() and (OUT/'status.json').exists():
        raise RuntimeError('Refuse automatic rerun of existing experiment;inspect its status first')
    prepare()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    try:
        with warnings.catch_warnings(),threadpool_limits(limits=1):
            warnings.simplefilter('error')
            select_all();evaluate_all()
    except BaseException as exc:
        write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),utc=datetime.now(timezone.utc).isoformat()))
        write(OUT/'status.json',dict(state='stopped_on_problem',error=repr(exc),
              current_audit_opened=(OUT/'choices_locked.json').exists()))
        emit('STOPPED on problem;preserved all artifacts;no automatic retry')
        raise


if __name__=='__main__':main()
