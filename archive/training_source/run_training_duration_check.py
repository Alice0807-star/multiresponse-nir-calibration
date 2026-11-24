"""One bounded SSC/pH duration check. Operator starts real training explicitly.

Unchanged subsets, widths, optimizer, learning rate, loss and seed streams.
Only duration changes to min2400/max4800/patience1200 for all three recipes.
No refit, audit-array loading, external data, or subset search in this entry.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import math
import platform
import signal
import sys
import traceback
import warnings

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import sklearn
from threadpoolctl import threadpool_limits
import run_matched_predictor_check as previous
from hsi_e2e.framework_screen import metrics,nn_predict,advance_patience
from hsi_e2e.formal_runtime import read,write,sha,torch_atomic

base=previous.prior.pilot.base
OUT=ROOT/'results/training_duration_4800_001'
SPLITS=(7117,8111,9103);SEEDS=(17,29,43)
METHODS=('union','covsel','full');TARGETS=(0,1)
WIDTHS=(24,23);DURATION=dict(min_epochs=2400,max_epochs=4800,patience=1200)
EXPECTED_SHAPES=((1008,641),(1008,3),(336,641),(336,3))


def cases():
    return [(s,seed,m,t) for s in SPLITS for seed in SEEDS for m in METHODS for t in TARGETS]


def versions():
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__)


def sources():
    return list(dict.fromkeys(previous.source_files()+[Path(__file__),ROOT/'tests/test_training_duration_check.py']))


def selection_arrays(split):
    # Deliberately not the prior audit loader; only the selection-only archive.
    with np.load(previous.prior.OLD/str(split)/'selection_data.npz',allow_pickle=False) as z:
        arrays=tuple(z[k].copy() for k in ('X_fit','y_fit','X_reward','y_reward'))
        ids=[z[k].copy() for k in ('fit_ids','reward_ids')]
    if tuple(a.shape for a in arrays)!=EXPECTED_SHAPES:raise ValueError('Unexpected fixed selection shapes')
    if len(set(ids[0]))!=len(ids[0]) or len(set(ids[1]))!=len(ids[1]) or set(ids[0])&set(ids[1]):
        raise ValueError('Duplicate/overlapping selection IDs')
    return arrays


def old_folder(case):return previous.folder_for(*case)
def new_folder(case):
    s,seed,method,target=case
    return OUT/str(s)/str(seed)/method/str(target)


def lock():
    p=read(OUT/'protocol.json')
    for key in ('source_sha256','input_sha256'):
        for file,digest in p[key].items():
            if sha(file)!=digest:raise RuntimeError('Locked file changed:'+file)
    if p['library_versions']!=versions():raise RuntimeError('Environment versions changed after preparation')
    previous.lock();previous.seal()
    return p


def prepare():
    if OUT.exists():raise RuntimeError('Existing duration check;do not overwrite or re-prepare')
    previous.lock();previous.seal()
    proof=read(previous.OUT/'verification.json')
    if not proof['passed'] or read(previous.OUT/'status.json')['state']!='complete_verified':
        raise RuntimeError('Completed matched-predictor comparison required')
    for file,digest in read(previous.OUT/'delivery_manifest.json')['files'].items():
        if sha(file)!=digest:raise RuntimeError('Previous result changed:'+file)
    tests=read(ROOT/'results/training_duration_readiness_001/verification.json')
    if not tests['passed']:raise RuntimeError('Duration entry tests required')
    for file,digest in tests['source_sha256'].items():
        if sha(file)!=digest:raise RuntimeError('Code changed since duration tests:'+file)
    panel=previous.panels();inputs=[previous.OUT/'protocol.json',previous.OUT/'delivery_manifest.json',
        previous.OUT/'panels.json',ROOT/'results/training_duration_readiness_001/verification.json']
    rates=[]
    for split in SPLITS:
        arrays=selection_arrays(split)
        inputs.append(previous.prior.OLD/str(split)/'selection_data.npz')
        for seed in SEEDS:
            for method in METHODS:
                previous.validate_selection(*arrays,previous.subset_for(panel,split,seed,method))
                # Firmness is never trained here;its existing reward metadata only.
                inputs.append(previous.folder_for(split,seed,method,2)/'selected.json')
    for case in cases():
        folder=old_folder(case);chosen=read(folder/'selected.json');history=read(folder/'history.json')
        if chosen['widths']!=list(WIDTHS) or chosen['epochs_run']>DURATION['min_epochs']:
            raise ValueError('Old/new duration or architecture mismatch')
        if [h['epoch'] for h in history]!=list(range(10,chosen['epochs_run']+1,10)):
            raise ValueError('Incomplete original history')
        inputs += [folder/name for name in ('selected.json','selected_model.pt','history.json')]
        rates.append(chosen['seconds']/chosen['epochs_run'])
    OUT.mkdir(parents=True)
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),purpose='Bounded reward-only training-duration sensitivity,not new model tuning',
        partitions=list(SPLITS),seeds=list(SEEDS),methods=list(METHODS),targets=list(TARGETS),models=len(cases()),widths=list(WIDTHS),duration=DURATION,
        unchanged='Exact panels,AdamWlr.001/decay.001,dropout.05,batchcap128,clip5,eval10,loss,normalization,seeds and random streams',
        prefix='Replay from original seed;compare every original reward checkpoint and original best weights/scales before extending past original stop',
        stopping='Any prefix mismatch,warning,nonfinite,integrity failure or operator interrupt stops;no automatic restart,extraepoch beyond4800,or case deletion',
        excluded='No audit arrays,no refits,no new-source data,no subset search,no LR/width/K/reweighting change',
        firmness='Existing reward metadata reused only;no new firmness training;cannot infer its global convergence',
        reporting='Keep all54cases;report best reward gains and relative gaps;best-over-longer-history gains are not independent generalization evidence',
        decision='Record remaining4800caps and late gains;no automatic protocol promotion or further extension;never claim global optimum',
        history_exposure='Same repeatedly used development fit/reward samples;posthoc bounded sensitivity,not blind validation',
        runtime='CPU actualTorch4/BLAS1/interop1',library_versions=versions(),
        estimate_minutes=dict(min_epochs=sum(r*DURATION['min_epochs'] for r in rates)/60,
            max_epochs=sum(r*DURATION['max_epochs'] for r in rates)/60,scope='Historical selection-time extrapolation,not guaranteed wall clock'),
        source_sha256={str(f):sha(f) for f in sources()},input_sha256={str(f):sha(f) for f in dict.fromkeys(inputs)})
    write(OUT/'protocol.json',p);write(OUT/'status.json',dict(state='prepared_not_started',models=len(cases()),completed=0,audit_arrays_loaded=False))
    lock()
    print(dict(prepared_only=True,models=len(cases()),real_training_started=False,estimate=p['estimate_minutes']),flush=True)


class PrefixObserver:
    """Read-only observers around the frozen trainer's evaluation callbacks."""
    def __init__(self,chosen,history,payload,emit=lambda message:None):
        self.chosen=chosen;self.history=history;self.payload=payload;self.emit=emit
        self.points=0;self.weights_exact=False;self.prefix_verified=False;self.best=-float('inf')

    def __enter__(self):
        self.predict_original=base.nn_predict;self.metrics_original=base.metrics
        base.nn_predict=self.predict;base.metrics=self.score
        return self

    def __exit__(self,*args):
        base.nn_predict=self.predict_original;base.metrics=self.metrics_original

    def predict(self,model,norm,x,subset):
        epoch=(self.points+1)*10
        if epoch==self.chosen['epoch']:
            if list(subset)!=self.payload['subset']:raise AssertionError('Original best columns differ')
            if any(not np.array_equal(norm[k],v) for k,v in self.payload['norm'].items()):raise AssertionError('Original normalization differs')
            if model.state_dict().keys()!=self.payload['state'].keys() or any(
                not torch.equal(model.state_dict()[k],v) for k,v in self.payload['state'].items()):
                raise AssertionError('Original best weights differ')
            self.weights_exact=True
        return self.predict_original(model,norm,x,subset)

    def score(self,y,prediction):
        score=self.metrics_original(y,prediction);epoch=(self.points+1)*10
        if self.points<len(self.history) and dict(epoch=epoch,**score)!=self.history[self.points]:
            raise AssertionError('Original reward history prefix differs at epoch'+str(epoch))
        self.points+=1;self.best=max(self.best,score['mean_r2'])
        if self.points==len(self.history):
            if not self.weights_exact:raise AssertionError('Best-weight prefix was not checked')
            self.prefix_verified=True;self.emit(f'PREFIX VERIFIED through epoch={epoch},old best weights/scales exact')
        if epoch%200==0:self.emit(f'EPOCH {epoch} best_reward_R2={self.best:.7f}')
        return score


def emit(message):
    line=datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+message
    print(line,flush=True)
    with (OUT/'run.log').open('a',encoding='utf-8') as f:f.write(line+'\n')


def train_case(case,arrays):
    split,seed,method,target=case;xf,yf,xr,yr=arrays
    old=old_folder(case);chosen=read(old/'selected.json');history=read(old/'history.json')
    payload=torch.load(old/'selected_model.pt',map_location='cpu',weights_only=False)
    subset=previous.subset_for(previous.panels(),split,seed,method)
    if chosen['subset']!=subset:raise AssertionError('Panel changed')
    emit(f'START split={split} seed={seed} method={method} target={target}')
    with PrefixObserver(chosen,history,payload,emit) as observer:
        trained=base.train(xf,yf[:,target:target+1],xr,yr[:,target:target+1],subset,seed,WIDTHS,**DURATION)
    if not observer.prefix_verified:raise AssertionError('Training ended before original prefix was verified')
    if trained['history'][:len(history)]!=history:raise AssertionError('Post-training prefix mismatch')
    if trained['selection']['mean_r2']<chosen['selection']['mean_r2']:raise AssertionError('Old best checkpoint lost')
    folder=new_folder(case);folder.mkdir(parents=True)
    torch_atomic(folder/'selected_model.pt',dict(state=trained['model'].state_dict(),norm=trained['norm'],subset=subset))
    write(folder/'history.json',trained['history'])
    row=dict(split=split,seed=seed,method=method,target=target,subset=subset,widths=list(WIDTHS),
        old_epoch=chosen['epoch'],old_epochs_run=chosen['epochs_run'],old_reward=chosen['selection'],
        epoch=trained['epoch'],epochs_run=trained['epochs_run'],updates=trained['updates'],seconds=trained['seconds'],reward=trained['selection'],
        reward_gain=trained['selection']['mean_r2']-chosen['selection']['mean_r2'],
        cap=trained['epochs_run']==DURATION['max_epochs'],
        history_prefix_exact=True,old_best_weights_exact=True,old_scales_exact=True,
        late1200_gain=trained['selection']['mean_r2']-max(h['mean_r2'] for h in trained['history'] if h['epoch']<=min(DURATION['max_epochs']-DURATION['patience'],trained['epochs_run'])))
    write(folder/'selected.json',row)
    emit(f'MODEL COMPLETE split={split} seed={seed} method={method} target={target} best={row["epoch"]} run={row["epochs_run"]} reward_gain={row["reward_gain"]:+.7f}')
    return row


def summarize(rows):
    if len(rows)!=len(cases()) or {(a['split'],a['seed'],a['method'],a['target']) for a in rows}!=set(cases()):
        raise AssertionError('Incomplete/duplicate duration cases')
    methods={}
    for method in METHODS:
        g=[a for a in rows if a['method']==method]
        old=[];new=[];caps=[]
        for target in TARGETS:
            tr=[a for a in g if a['target']==target]
            old.append(float(np.mean([a['old_reward']['mean_r2'] for a in tr])))
            new.append(float(np.mean([a['reward']['mean_r2'] for a in tr])))
            caps.append(sum(a['cap'] for a in tr))
        firmness=float(np.mean([read(previous.folder_for(s,seed,method,2)/'selected.json')['selection']['mean_r2'] for s in SPLITS for seed in SEEDS]))
        methods[method]=dict(old_reward=old+[firmness],new_reward=new+[firmness],trained_target_caps=caps,
            old_mean_reward=float(np.mean(old+[firmness])),new_mean_reward=float(np.mean(new+[firmness])))
    comparison={}
    for ref in METHODS[1:]:
        a,b=methods['union'],methods[ref]
        comparison[ref]=dict(old_gap=(np.array(a['old_reward'])-b['old_reward']).tolist(),new_gap=(np.array(a['new_reward'])-b['new_reward']).tolist())
    return dict(methods=methods,union_minus_reference=comparison,remaining_caps=sum(a['cap'] for a in rows),
        bounded_duration_rule=DURATION,configuration_promoted=False,independent_validation=False,global_convergence_claimed=False,
        interpretation='Best reward after more checkpoints is training sensitivity,not independent performance improvement;firmness unchanged')


def verify():
    lock();rows=read(OUT/'metrics.json');checks=0;maximum=0.
    for case in cases():
        f=new_folder(case);row=read(f/'selected.json');history=read(f/'history.json');chosen=read(old_folder(case)/'selected.json')
        if next(a for a in rows if (a['split'],a['seed'],a['method'],a['target'])==case)!=row:
            raise AssertionError('Per-model and global metadata mismatch')
        if (row['split'],row['seed'],row['method'],row['target'])!=case or row['widths']!=list(WIDTHS):
            raise AssertionError('Model identity/width mismatch')
        if row['subset']!=previous.subset_for(previous.panels(),case[0],case[1],case[2]):
            raise AssertionError('Frozen panel mismatch')
        if not all(row[k] for k in ('history_prefix_exact','old_best_weights_exact','old_scales_exact')):
            raise AssertionError('Missing prefix proof')
        old_history=read(old_folder(case)/'history.json')
        if history[:len(old_history)]!=old_history:raise AssertionError('Stored prefix mismatch')
        if row['epoch']!=max(history,key=lambda a:a['mean_r2'])['epoch']:raise AssertionError('Reward-only best epoch')
        if not DURATION['min_epochs']<=row['epochs_run']<=DURATION['max_epochs']:raise AssertionError('Duration bounds')
        if [h['epoch'] for h in history]!=list(range(10,row['epochs_run']+1,10)):raise AssertionError('Incomplete history')
        arrays=selection_arrays(case[0]);x,y,xv,yv=arrays;target=case[3];subset=row['subset']
        if row['updates']!=math.ceil(len(x)/128)*row['epochs_run']:raise AssertionError('Update accounting')
        anchor,last,best=-float('inf'),0,-float('inf')
        for h in history:
            best=max(best,h['mean_r2']);anchor,last=advance_patience(best,anchor,last,h['epoch'])
        if row['epochs_run']!=DURATION['max_epochs'] and row['epochs_run']-last<DURATION['patience']:raise AssertionError('Patience mismatch')
        payload=torch.load(f/'selected_model.pt',map_location='cpu',weights_only=False)
        expected=dict(xm=x[:,subset].mean(0),xs=x[:,subset].std(0),ym=y[:,target:target+1].mean(0),ys=y[:,target:target+1].std(0))
        if payload['subset']!=subset or any(not np.array_equal(payload['norm'][k],v) for k,v in expected.items()):raise AssertionError('Training-only scales/columns')
        model=base.ControlMLP(len(subset),1,WIDTHS);model.load_state_dict(payload['state']);model.eval()
        actual=metrics(yv[:,target:target+1],nn_predict(model,payload['norm'],xv,subset))
        error=float(np.max(abs(np.array(actual['r2'])-row['reward']['r2'])));maximum=max(maximum,error)
        if actual!=row['reward'] or actual['mean_r2']<chosen['selection']['mean_r2']:raise AssertionError('Reward reload/old best mismatch')
        checks+=len(old_history)
    summary=summarize(rows)
    if summary!=read(OUT/'summary.json'):raise AssertionError('Duration summary mismatch')
    lock()
    proof=dict(passed=True,models=len(rows),original_prefix_points=checks,max_reward_reload_error=maximum,
        new_model_fits=len(rows),new_searches=0,new_refits=0,audit_arrays_loaded=False,external_data_loaded=False,
        original_files_unchanged=True,global_convergence_claimed=False)
    write(OUT/'verification.json',proof)
    return proof


def report(summary,proof):
    lines=['# 固定训练时长检查结果','',
        '只检查SSC与pH；并集、CovSel、全谱均用最少2400／最多4800／耐心1200。所有其他规则与波段固定；硬度不新训练。',
        '本轮只使用历史拟合／奖励数据，没有重拟合或打开复核数组。奖励最大值在更多检查点中选择，不能当作独立泛化提高。','',
        '|方案|旧奖励SSC|新奖励SSC|旧奖励pH|新奖励pH|SSC触顶|pH触顶|','|---|---:|---:|---:|---:|---:|---:|']
    for method,a in summary['methods'].items():
        lines.append(f'|{method}|{a["old_reward"][0]:.6f}|{a["new_reward"][0]:.6f}|{a["old_reward"][1]:.6f}|{a["new_reward"][1]:.6f}|{a["trained_target_caps"][0]}|{a["trained_target_caps"][1]}|')
    lines+=['','|并集减参照|旧SSC差|新SSC差|旧pH差|新pH差|','|---|---:|---:|---:|---:|']
    for ref,a in summary['union_minus_reference'].items():
        lines.append(f'|{ref}|{a["old_gap"][0]:+.6f}|{a["new_gap"][0]:+.6f}|{a["old_gap"][1]:+.6f}|{a["new_gap"][1]:+.6f}|')
    lines+=['',f'仍触及4800轮：{summary["remaining_caps"]}个模型。触顶是预算限制，不是性能上限。',
        f'全部{proof["models"]}模型的原始曲线前缀和原最佳权重／尺度已逐一核对；保存模型奖励预测复算误差{proof["max_reward_reload_error"]:.1g}。',
        '本次最多执行到4800，不自动加轮数、改变学习率或晋级正式配置。需结合每个划分／种子和目标的差值判断，不挑选单次最好结果。',
        '若之后进行正式泛化评价，必须先冻结统一训练规则及完整评价协议，不能把本轮奖励增益直接补写成复核增益。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run():
    lock()
    if read(OUT/'status.json')['state']!='prepared_not_started':raise RuntimeError('Already started;inspect results/failure,do not retry automatically')
    with (OUT/'operator_start.txt').open('x',encoding='utf-8') as f:f.write(datetime.now(timezone.utc).isoformat()+'\n')
    try:
        signal.signal(signal.SIGINT,lambda *_:setattr(base,'STOP',True))
        torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        with warnings.catch_warnings(),threadpool_limits(limits=1):
            warnings.simplefilter('error');runtime=previous.version.pin_and_record()
            if runtime['actual_interop_threads']!=1:raise RuntimeError('Interop1 required')
            runtime['python_executable']=sys.executable;write(OUT/'environment.json',runtime)
            write(OUT/'status.json',dict(state='training_reward_only',models=len(cases()),completed=0,audit_arrays_loaded=False))
            arrays={s:selection_arrays(s) for s in SPLITS};rows=[]
            for case in cases():
                rows.append(train_case(case,arrays[case[0]]));previous.version.pin_and_record()
                write(OUT/'progress_metrics.json',rows)
                write(OUT/'status.json',dict(state='training_reward_only',models=len(cases()),completed=len(rows),audit_arrays_loaded=False))
            write(OUT/'metrics.json',rows);summary=summarize(rows);write(OUT/'summary.json',summary)
            proof=verify();report(summary,proof)
        write(OUT/'status.json',dict(state='complete_verified',models=len(cases()),completed=len(rows),audit_arrays_loaded=False))
        emit('COMPLETE duration sensitivity verified;no audit/refit/external experiment or further extension started')
        files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json']
        write(OUT/'delivery_manifest.json',dict(files={str(p):sha(p) for p in files}))
    except BaseException as error:
        write(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        write(OUT/'status.json',dict(state='stopped_on_problem',audit_arrays_loaded=False))
        emit('STOPPED;preserve files and diagnose;no automatic retry or changed parameters')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run','status']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='run':run()
    else:print(read(OUT/'status.json'))


if __name__=='__main__':main()
