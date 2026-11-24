"""Single-factor online attention adaptation diagnostic on fixed development split."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,shutil,signal,sys,time,traceback,json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch,sklearn
from threadpoolctl import threadpool_limits
from hsi_e2e.variable_k import VariableEnsemble
from hsi_e2e.framework_screen import train_nn,nn_predict,metrics,pls_fit,pls_predict
from hsi_e2e.formal_runtime import (read,sha,write,equal_tree,torch_atomic,restore_active,
    save_active,drive,archive,completed,GracefulStop)
from shared_warm_freeze import fork_after_warm
import run_framework_confirmation as historical

BASE=ROOT/'results/framework_confirmation_001'
OUT=ROOT/'results/shared_warm_freeze_003'
CONDITIONS=['online_full','head_only']
STOP=False

_torch_atomic_existing_parent=torch_atomic


def torch_atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    _torch_atomic_existing_parent(path,value)


def archive_equivalent(runtime,stored):
    # Archives encode tuples as JSON lists;do not round any numeric value.
    from hsi_e2e.formal_runtime import jsonable
    return json.loads(json.dumps(runtime,default=jsonable,allow_nan=False))==stored


def emit(message):
    line=datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+message
    print(line,flush=True)
    if OUT.exists():
        with (OUT/'run.log').open('a',encoding='utf-8') as f:f.write(line+'\n')


def prepare():
    if (OUT/'protocol.json').exists():return lock()
    historical.lock()
    if OUT.exists():raise RuntimeError('Refuse existing output without protocol')
    oldenv=read(BASE/'execution_environment.json')
    for name,value in dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,sklearn=sklearn.__version__).items():
        if oldenv[name]!=value:raise RuntimeError('Historical environment version changed:'+name)
    OUT.mkdir()
    sources=[Path(__file__).resolve(),ROOT/'shared_warm_freeze.py',ROOT/'tests/test_shared_warm_freeze.py']+sorted((ROOT/'src/hsi_e2e').glob('*.py'))
    inputs=[BASE/'protocol.json',BASE/'7117/selection_data.npz',BASE/'7117/audit_data.npz',BASE/'metrics.json']
    for seed in [17,29,43]:
        for method,names in [('warm',['active_state.pt','active_manifest.json']),
            ('adaptive',['completed_state.pt','ledger.json','decisions.json','restarts.json','selected.json','complete.json'])]:
            inputs.extend(BASE/'7117/search'/str(seed)/method/name for name in names)
    snapshot=OUT/'code_snapshot';snapshot.mkdir()
    for path in sources:shutil.copy2(path,snapshot/path.name)
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),split=7117,seeds=[17,29,43],conditions=CONDITIONS,
        previous_startup='001 checkpointparentmissing;002 tuple-vs-JSONlist check falsealarm at800 historicalwarmfits;bothstopped before any newPLSquery;allfailure/protocol/snapshots retained;two512updatewarmcompute attempts repeated',
        k=48,components=8,budget=12800,warm_queries=800,warm_updates=512,
        torch_threads=4,blas_threads=1,
        intervention='After identical full warm training,freeze attention only in head_only;all warm weights,Adam groups/moments,norm,replay,RNG unchanged;online head continues botharms',
        freeze_implementation='Keeprequires_grad and forward/backward kernels identical;discard attentiongrad beforeclip/Adam;attentionweights/moments fixed;timingnot optimizedfrozenbackprop',
        controls='Original legacy128pool/4guided4probe/diversity/restart16/R2reward/Adam.001/update1;no new K,objective,learning rate,NNhyperparameters',
        warm_cost='Historical800warmPLS reused;logical12800perarm includes800;72000 newphysicalsearchfits,76800logicalfits across6arms;warmcompute512 perseed sharedbotharms',
        positive_control='online_full rerun must exactly reproduce historicaladaptive paidledger/actions/predictions/restarts/weights/Adam/RNG;first discrepancy stops before audit',
        predictor='Unchanged shared64/32/3heads MLP;normalizedMSE;AdamW.001/decay.001;max1200/min400/patience300;fit1008/reward336;fixedbestepochrefit1344',
        gate='Headonly diagnostic improvement:mean delta>0,>=2/3 pairedseedwins,all3 targetmean deltas>=0;otherwise no stable benefit;no new independent validation',
        boundary='All6subsets/NNepochs/checkpoints sealed before current audit loads;historicalaudit previously inspected;posthoc development singlefactor study',
        stop='Stop entire experiment on exception/numerical/fee/reproduction failure;preserve full failurestate;no dropped seeds',
        source_sha256={str(path):sha(path) for path in sources},input_sha256={str(path):sha(path) for path in inputs})
    write(OUT/'protocol.json',p);write(OUT/'status.json',dict(state='prepared',current_audit_opened=False))
    emit('PREPARED7117 x3pairedseeds;sharedtrainedwarm;onlyonlineattentiontrainabilitychanges')
    return p


def lock():
    historical.lock();p=read(OUT/'protocol.json')
    for group in ['source_sha256','input_sha256']:
        for path,value in p[group].items():
            if sha(path)!=value:raise RuntimeError('Locked file changed:'+path)
    return p


def restore_warm(seed):
    folder=BASE/'7117/search'/str(seed)/'warm';manifest=read(folder/'active_manifest.json')
    warm=restore_active(folder,manifest['identity'])
    if warm.ev.fit_count!=800 or warm.learner is not None:raise RuntimeError('Unexpected historical warm state')
    return warm


def shared_warm(seed):
    path=OUT/str(seed)/'shared_warm.pt'
    if path.exists():
        marker=read(path.with_suffix('.json'));assert sha(path)==marker['sha256']
        saved=torch.load(path,weights_only=False,map_location='cpu');torch.set_rng_state(saved['torch_rng']);return saved['search']
    warm=restore_warm(seed);start=time.perf_counter();state=copy.deepcopy(warm)
    state.arm='adaptive';state.ev.budget=12800;state.warm_groups=[]
    state.learner=VariableEnsemble(warm.warm_groups,k=48,adaptive=True)
    for _ in range(512):state.learner.update()
    state.elapsed_seconds+=time.perf_counter()-start
    torch_atomic(path,dict(search=state,torch_rng=torch.get_rng_state()))
    write(path.with_suffix('.json'),dict(sha256=sha(path),seed=seed,updates=state.learner.updates))
    return state


def check_initial(prepared,state):
    a,b=prepared.learner,state.learner
    assert equal_tree(a.norm,b.norm) and equal_tree(list(a.groups),list(b.groups))
    assert a.rng.bit_generator.state==b.rng.bit_generator.state
    assert equal_tree(prepared.rng.bit_generator.state,state.rng.bit_generator.state)
    assert equal_tree(prepared.restart_rng.bit_generator.state,state.restart_rng.bit_generator.state)
    assert prepared.ev.ledger==state.ev.ledger and prepared.current==state.current
    for ma,mb,oa,ob in zip(a.models,b.models,a.optimizers,b.optimizers):
        assert equal_tree(ma.state_dict(),mb.state_dict()) and ma.training==mb.training
        assert equal_tree(oa.state_dict(),ob.state_dict())
    return dict(passed=True,updates=512,initial_weights_optimizer_norm_replay_sampler_searchRNG_equal=True,
        frozen_attention_only=state.arm=='frozen')


def historical_reproduction(folder,seed):
    old=BASE/'7117/search'/str(seed)/'adaptive'
    for name in ['ledger.json','decisions.json','restarts.json']:
        if read(folder/name)!=read(old/name):raise RuntimeError('Historical positivecontrol differs:'+name)
    a=torch.load(folder/'completed_state.pt',weights_only=False,map_location='cpu')
    b=torch.load(old/'completed_state.pt',weights_only=False,map_location='cpu')
    for key in ['best_record','torch_rng','search_rng','restart_rng','roots','current','episode','step_in_episode']:
        assert equal_tree(a[key],b[key]),key
    for x,y,ox,oy in zip(a['learner'].models,b['learner'].models,a['learner'].optimizers,b['learner'].optimizers):
        assert equal_tree(x.state_dict(),y.state_dict()) and equal_tree(ox.state_dict(),oy.state_dict())
    for key in ['norm','updates']:
        assert equal_tree(getattr(a['learner'],key),getattr(b['learner'],key))
    assert equal_tree(list(a['learner'].groups),list(b['learner'].groups))
    assert a['learner'].rng.bit_generator.state==b['learner'].rng.bit_generator.state
    write(folder/'reproduction.json',dict(passed=True,paid_queries=12800,exact_actions_predictions_rewards_weights_optimizer_replay_RNG=True))


def select(p):
    with np.load(BASE/'7117/selection_data.npz',allow_pickle=False) as z:arrays=[z[k] for k in ['X_fit','y_fit','X_reward','y_reward']]
    for seed in p['seeds']:
        prepared=shared_warm(seed);shared_torch=torch.get_rng_state()
        for condition in CONDITIONS:
            folder=OUT/str(seed)/'search'/condition
            if (folder/'selected.json').exists():continue
            variant='adaptive' if condition=='online_full' else 'frozen'
            identity=dict(stage='sharedwarm',unit='7117',seed=seed,k=48,variant=variant,condition=condition,
                budget=12800,protocol_sha256=sha(OUT/'protocol.json'))
            if (folder/'complete.json').exists():summary=completed(folder,identity)
            else:
                if (folder/'active_manifest.json').exists():state=restore_active(folder,identity)
                else:
                    torch.set_rng_state(shared_torch);state=fork_after_warm(prepared,condition)
                    write(folder/'initial_equivalence.json',check_initial(prepared,state));save_active(folder,state,identity)
                emit(f'SEARCH START seed={seed} condition={condition}')
                expected_ledger=read(BASE/'7117/search'/str(seed)/'adaptive'/'ledger.json') if condition=='online_full' else None
                expected_decisions=read(BASE/'7117/search'/str(seed)/'adaptive'/'decisions.json') if condition=='online_full' else None
                checked_fits,checked_decisions=0,0
                def stop_or_check():
                    nonlocal checked_fits,checked_decisions
                    if expected_ledger is not None:
                        if (not archive_equivalent(state.ev.ledger[checked_fits:],expected_ledger[checked_fits:state.ev.fit_count]) or
                            not archive_equivalent(state.decisions[checked_decisions:],expected_decisions[checked_decisions:len(state.decisions)])):
                            torch_atomic(folder/'failure_state.pt',dict(search=state,torch_rng=torch.get_rng_state()))
                            raise RuntimeError('Historical reproduction firstdivergence at fits='+str(state.ev.fit_count))
                        checked_fits=state.ev.fit_count;checked_decisions=len(state.decisions)
                    return STOP
                drive(state,12800,folder,identity,stop_or_check,emit)
                assert state.ev.fit_count==12800 and state.ev.cache_hits==0
                summary=archive(folder,state,identity,OUT);del state
            if condition=='online_full':historical_reproduction(folder,seed)
            else:
                saved=torch.load(folder/'completed_state.pt',weights_only=False,map_location='cpu')['learner']
                for ma,mb in zip(prepared.learner.models,saved.models):
                    assert equal_tree(ma.attention.state_dict(),mb.attention.state_dict())
                    assert not equal_tree(ma.head.state_dict(),mb.head.state_dict())
                saved.audit()
            write(folder/'selected.json',dict(seed=seed,method=condition,subset=summary['best']['subset'],
                selection={k:summary['best'][k] for k in ['r2','mean_r2','rmse']},fits=12800,seconds=summary['seconds']))
            emit(f'SEARCH COMPLETE seed={seed} condition={condition}')
        del prepared
    for seed in p['seeds']:
        for condition in CONDITIONS:
            folder=OUT/str(seed)/'models'/condition
            if (folder/'selected.json').exists():continue
            if STOP:raise GracefulStop('Operator stop')
            choice=read(OUT/str(seed)/'search'/condition/'selected.json')
            emit(f'NN START seed={seed} condition={condition}')
            trained=train_nn(*arrays,choice['subset'],seed)
            torch_atomic(folder/'selected_model.pt',dict(state=trained['model'].state_dict(),norm=trained['norm'],subset=choice['subset']))
            write(folder/'history.json',trained['history'])
            write(folder/'selected.json',dict(seed=seed,method=condition,subset=choice['subset'],epoch=trained['epoch'],
                epochs_run=trained['epochs_run'],updates=trained['updates'],selection=trained['selection'],
                nn_seconds=trained['seconds'],selector_seconds=choice['seconds']))
            emit(f'NN COMPLETE seed={seed} condition={condition}')
    paths=[OUT/str(seed)/'models'/c/'selected.json' for seed in p['seeds'] for c in CONDITIONS]
    files={str(path):sha(path) for path in paths}
    files.update({str(path.parent/'selected_model.pt'):sha(path.parent/'selected_model.pt') for path in paths})
    marker=OUT/'choices_locked.json'
    if marker.exists():assert read(marker)['files']==files
    else:write(marker,dict(utc=datetime.now(timezone.utc).isoformat(),count=6,files=files))


def audit_data():
    marker=OUT/'choices_locked.json'
    if not marker.exists():raise RuntimeError('Refuse current audit before6choice seal')
    for path,value in read(marker)['files'].items():assert sha(path)==value
    with np.load(BASE/'7117/audit_data.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def evaluate(p):
    audit=audit_data();xa,ya=audit['X_audit'],audit['y_audit']
    with np.load(BASE/'7117/selection_data.npz',allow_pickle=False) as z:
        xr,yr=np.vstack([z['X_fit'],z['X_reward']]),np.vstack([z['y_fit'],z['y_reward']])
    rows=[]
    for seed in p['seeds']:
        for condition in CONDITIONS:
            folder=OUT/str(seed)/'models'/condition
            if (folder/'audit.json').exists():rows.append(read(folder/'audit.json'));continue
            chosen=read(folder/'selected.json');trained=train_nn(xr,yr,None,None,chosen['subset'],seed,fixed_epochs=chosen['epoch'])
            pred=nn_predict(trained['model'],trained['norm'],xa,chosen['subset'])
            model=pls_fit(xr,yr,chosen['subset'],8);pp=pls_predict(model,xa)
            row=dict(**chosen,audit=metrics(ya,pred),pls8_audit=metrics(ya,pp),refit_updates=trained['updates'],refit_seconds=trained['seconds'])
            torch_atomic(folder/'refit_model.pt',dict(state=trained['model'].state_dict(),norm=trained['norm'],subset=chosen['subset']))
            np.savez_compressed(folder/'predictions.npz',prediction=pred,pls8_prediction=pp,target=ya,ids=audit['audit_ids'])
            if condition=='online_full':
                old=read(BASE/'7117/models'/str(seed)/'adaptive'/'audit.json')
                assert row['epoch']==old['epoch'] and row['audit']==old['audit']
                with np.load(BASE/'7117/models'/str(seed)/'adaptive'/'predictions.npz',allow_pickle=False) as z:np.testing.assert_array_equal(pred,z['prediction'])
            write(folder/'audit.json',row);rows.append(row)
            emit(f'AUDIT seed={seed} condition={condition} mean={row["audit"]["mean_r2"]:.7f}')
    write(OUT/'metrics.json',rows);lock()
    write(OUT/'complete.json',dict(utc=datetime.now(timezone.utc).isoformat(),cases=6,metrics_sha256=sha(OUT/'metrics.json')))
    write(OUT/'status.json',dict(state='complete',cases=6));emit('SHAREDWARM SINGLEFACTOR COMPLETE')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    signal.signal(signal.SIGINT,lambda *_:globals().__setitem__('STOP',True))
    p=prepare()
    if args.prepare_only:return
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    try:
        with threadpool_limits(limits=1):
            write(OUT/'status.json',dict(state='selecting',current_audit_opened=False));select(p);evaluate(p)
    except BaseException as exc:
        write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));emit('STOPPED;preserve state and diagnose before retry');raise


if __name__=='__main__':main()
