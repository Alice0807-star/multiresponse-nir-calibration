"""Fixed final predictor-matching check. Preparation never trains a model.

This is historical development evidence, not independent-source validation.
The operator explicitly starts `run` from their terminal after preparation.
Frozen historical scientific files are only read. No parameters are searched.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import math
import signal
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run_union_replication_v2 as version
from hsi_e2e.framework_screen import metrics, nn_predict, advance_patience
from hsi_e2e.formal_runtime import read, write, sha, torch_atomic

prior = version.run
OUT = ROOT/'results/matched_predictor_final_001'
AUDIT = ROOT/'results/preformal_code_audit_001'
SPLITS = (7117, 8111, 9103)
SEEDS = (17, 29, 43)
METHODS = ('union', 'shared', 'covsel', 'full')
WIDTHS = (24, 23)
DURATION = dict(min_epochs=1200, max_epochs=2400, patience=600)


def source_files():
    old = [Path(p) for p in prior.lock()['source_sha256']]
    return list(dict.fromkeys(old+[Path(__file__), ROOT/'tests/test_matched_predictor_check.py',
        ROOT/'audit_preformal_code.py', ROOT/'tests/test_preformal_effects.py']))


def validate_selection(x, y, xv, yv, subset):
    values = [np.asarray(v) for v in (x, y, xv, yv)]
    if any(v.ndim != 2 or not np.isfinite(v).all() for v in values):
        raise ValueError('Finite two-dimensional arrays required')
    if len(x) != len(y) or len(xv) != len(yv) or x.shape[1] != xv.shape[1] or y.shape[1] != yv.shape[1]:
        raise ValueError('Input/response alignment')
    raw = list(subset)
    if not raw or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in raw):
        raise ValueError('Integer selected columns required')
    if len(set(raw)) != len(raw) or min(raw) < 0 or max(raw) >= x.shape[1]:
        raise ValueError('Duplicate/out-of-range selected columns')
    if np.any(x[:, raw].std(0) <= 1e-12) or np.any(y.std(0) <= 1e-12) or np.any(yv.std(0) <= 1e-12):
        raise ValueError('Constant fit input or fit/reward response')


def counts():
    cells=len(SPLITS)*len(SEEDS);new=cells*(len(METHODS)-1)*3;reused=cells*3
    return dict(pipeline_cases=cells*len(METHODS), underlying_nn_models=new+reused, new_nn_selections=new,
                new_nn_refits=new, reused_nn_selections=reused, reused_nn_refits=reused, new_band_search_fits=0)


def panels():
    return read(OUT/'panels.json')


def subset_for(panel, split, seed, method):
    if method not in METHODS:
        raise ValueError('Unknown fixed method')
    return panel[str(split)][str(seed)][method]


def folder_for(split, seed, method, target):
    if method == 'union':
        return prior.folder_for(split, seed, 'union_st_capacity', target)
    return OUT/str(split)/'models'/str(seed)/method/str(target)


def selection_arrays(split):
    if split not in SPLITS:
        raise ValueError('This entry only permits the three declared grape development partitions')
    arrays = prior.selection_arrays(split)
    if arrays[1].shape[1] != 3:
        raise ValueError('This bounded grape check requires exactly three targets')
    return arrays


def lock():
    p = read(OUT/'protocol.json')
    for field in ('source_sha256', 'input_sha256'):
        for path, digest in p[field].items():
            if sha(path) != digest:
                raise RuntimeError('Locked file changed: '+path)
    prior.lock(); prior.seal()
    return p


def choice_files():
    files = [OUT/'panels.json']
    for split in SPLITS:
        for seed in SEEDS:
            for method in METHODS:
                for target in range(3):
                    f = folder_for(split, seed, method, target)
                    files += [f/'selected.json', f/'selected_model.pt', f/'history.json']
    return files


def seal(create=False):
    files = {str(path): sha(path) for path in choice_files()}
    marker = OUT/'choices_locked.json'
    if marker.exists():
        if read(marker)['files'] != files:
            raise RuntimeError('Selected choices changed')
    elif create:
        write(marker, dict(utc=datetime.now(timezone.utc).isoformat(), files=files, counts=counts()))
    else:
        raise RuntimeError('Global selection seal is missing')


def load_audit(split):
    if not (OUT/'choices_locked.json').exists():
        raise RuntimeError('Audit access blocked before global selection seal')
    seal(); lock()
    with np.load(prior.OLD/str(split)/'audit_data.npz', allow_pickle=False) as data:
        return tuple(data[k].copy() for k in ('X_audit', 'y_audit', 'audit_ids'))


def prepare():
    if OUT.exists():
        raise RuntimeError('Output already exists; inspect status instead of re-preparing or overwriting')
    proof = read(AUDIT/'verification.json')
    if not proof['passed'] or not proof['frozen_union_reverification']['passed']:
        raise RuntimeError('Completed code/structure audit required')
    for path, digest in proof['source_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('Code changed after audit: '+path)
    for path, digest in proof['input_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('Audit input changed: '+path)
    prior.lock(); prior.seal()
    if not read(prior.OUT/'verification.json')['passed']:
        raise RuntimeError('Verified union results required')
    panel = read(prior.OUT/'panels.json')
    for split in SPLITS:
        arrays = selection_arrays(split)
        for seed in SEEDS:
            for method in METHODS:
                validate_selection(*arrays, subset_for(panel, split, seed, method))
                if method != 'full' and len(subset_for(panel, split, seed, method)) > 48:
                    raise ValueError('Shared input cap')
    inputs = [prior.OUT/'protocol.json', prior.OUT/'delivery_manifest.json', prior.OUT/'panels.json',
              prior.OUT/'summary.json', AUDIT/'verification.json']
    inputs += [prior.OLD/str(split)/name for split in SPLITS for name in ('selection_data.npz', 'audit_data.npz')]
    # Include all already-fitted union choices and predictions in the input lock.
    for split in SPLITS:
        for seed in SEEDS:
            inputs.append(prior.OUT/str(split)/'predictions'/str(seed)/'union_st_capacity.npz')
            for target in range(3):
                f = folder_for(split, seed, 'union', target)
                inputs += [f/name for name in ('selected.json', 'selected_model.pt', 'history.json', 'refit_model.pt', 'refit.json')]
    OUT.mkdir(parents=True)
    write(OUT/'panels.json', panel)
    p = dict(created_utc=datetime.now(timezone.utc).isoformat(), purpose='Complete predictor-matched references with no new development tuning',
        partitions=list(SPLITS), seeds=list(SEEDS), methods=list(METHODS), counts=counts(),
        widths=list(WIDTHS), duration=DURATION,
        training='Unchanged frozen generic trainer:AdamWlr.001/decay.001,batchcap128,dropout.05,clip5,eval10',
        input='Use exact already-paid union/shared/CovSel/full panels;all three independent heads see their full method panel',
        choice='Each target chooses its own reward-R2 epoch;seal all108choices before any current audit access;refit fit+reward at fixed epochs',
        descriptive_rule='union minus each reference:overallmean>0,all3partitionmeans>0,at least6/9cellwins,all3targetmeans>=0;no independent significance claim',
        reuse='All27union24/23 models and their fixed-epoch refits/predictions reused byte-exactly;81new selections+81newrefits for three references',
        runtime='ActualTorch4 inside global limits1,BLAS1,interop1;CPU only',
        cost='Report new/reused and logical model counts;historical cached selection costs preclude a strict end-to-end speedup claim',
        exposure='All3grape partitions historically adapted and overlapping;not independent validation or a new algorithm',
        stopping='Any warning/nonfinite/integrity error/operator stop stops;no parameter changes,extraepochs,case dropping,automatic retry or promotion',
        next_stage='Independent-source protocol must be separately sealed after this bounded check;this command never starts it',
        source_sha256={str(path): sha(path) for path in source_files()},
        input_sha256={str(path): sha(path) for path in inputs})
    write(OUT/'protocol.json', p)
    write(OUT/'status.json', dict(state='prepared_not_started', audit_opened=False, counts=counts()))
    lock()
    print('PREPARED ONLY:81new selections+81refits;no model trained;operator must explicitly run the run command.', flush=True)


def emit(message):
    line = datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+message
    print(line, flush=True)
    with (OUT/'run.log').open('a', encoding='utf-8') as stream:
        stream.write(line+'\n')


def select_all(panel):
    for split in SPLITS:
        x, y, xv, yv = selection_arrays(split)
        for seed in SEEDS:
            for method in METHODS[1:]:
                subset = subset_for(panel, split, seed, method)
                for target in range(3):
                    trained = prior.pilot.base.train(x, y[:, target:target+1], xv, yv[:, target:target+1],
                        subset, seed, WIDTHS, **DURATION)
                    folder = folder_for(split, seed, method, target)
                    folder.mkdir(parents=True)
                    torch_atomic(folder/'selected_model.pt', dict(state=trained['model'].state_dict(), norm=trained['norm'], subset=subset))
                    write(folder/'history.json', trained['history'])
                    values = {k: trained[k] for k in ('epoch', 'epochs_run', 'updates', 'seconds', 'selection')}
                    write(folder/'selected.json', dict(split=split, seed=seed, method=method, task=target, subset=subset,
                        widths=list(WIDTHS), parameters=sum(p.numel() for p in trained['model'].parameters()), **values))
                    emit(f'SELECT split={split} seed={seed} method={method} target={target} best={trained["epoch"]} run={trained["epochs_run"]}')
    seal(create=True); lock()
    emit('ALL108CHOICES SEALED;no current audit arrays opened during selection')


def contrast(rows, reference):
    pairs=[]
    for split in SPLITS:
        for seed in SEEDS:
            a=next(r for r in rows if (r['split'],r['seed'],r['method'])==(split,seed,'union'))
            b=next(r for r in rows if (r['split'],r['seed'],r['method'])==(split,seed,reference))
            target=(np.array(a['audit']['r2'])-b['audit']['r2']).tolist()
            pairs.append(dict(split=split,seed=seed,target_delta=target,mean_delta=float(np.mean(target))))
    target=np.mean([p['target_delta'] for p in pairs],axis=0)
    parts={str(s):float(np.mean([p['mean_delta'] for p in pairs if p['split']==s])) for s in SPLITS}
    wins=sum(p['mean_delta']>0 for p in pairs); mean=float(target.mean())
    return dict(mean_delta=mean,target_delta=target.tolist(),partition_deltas=parts,cell_wins=wins,pairs=pairs,
        descriptive_support=bool(mean>0 and min(parts.values())>0 and wins>=6 and min(target)>=0))


def summarize(rows):
    expected={(s,seed,m) for s in SPLITS for seed in SEEDS for m in METHODS}
    if len(rows)!=len(expected) or {(r['split'],r['seed'],r['method']) for r in rows}!=expected:
        raise ValueError('Complete unique method/partition/seed grid required')
    methods={}
    for method in METHODS:
        group=[r for r in rows if r['method']==method]
        methods[method]=dict(mean_r2=float(np.mean([r['audit']['mean_r2'] for r in group])),
            r2=np.mean([r['audit']['r2'] for r in group],axis=0).tolist(),
            rmse=np.mean([r['audit']['rmse'] for r in group],axis=0).tolist(),
            partition_means={str(s):float(np.mean([r['audit']['mean_r2'] for r in group if r['split']==s])) for s in SPLITS})
    comparisons={method:contrast(rows,method) for method in METHODS[1:]}
    return dict(methods=methods,union_minus_reference=comparisons,
        all_descriptive_conditions=all(c['descriptive_support'] for c in comparisons.values()),
        independent_validation=False,configuration_promoted=False,no_further_tuning=True)


def evaluate_all(panel):
    seal(); lock()
    write(OUT/'status.json',dict(state='refitting_and_evaluating',audit_opened=True,counts=counts()))
    rows=[]
    for split in SPLITS:
        xf,yf,xr,yr=selection_arrays(split);x,y=np.vstack([xf,xr]),np.vstack([yf,yr])
        xa,ya,ids=load_audit(split)
        with np.load(prior.OLD/str(split)/'selection_data.npz',allow_pickle=False) as d:
            prior.pilot.prior.assert_partition(d['fit_ids'],d['reward_ids'],ids)
        for seed in SEEDS:
            for method in METHODS:
                subset=subset_for(panel,split,seed,method)
                parameters=0;caps=0;updates=0;seconds=0.;selected_seconds=0.
                for target in range(3):
                    chosen=read(folder_for(split,seed,method,target)/'selected.json')
                    parameters+=chosen['parameters'];caps+=chosen['epochs_run']==2400
                    selected_seconds+=chosen['seconds']
                if method=='union':
                    path=prior.OUT/str(split)/'predictions'/str(seed)/'union_st_capacity.npz'
                    with np.load(path,allow_pickle=False) as z:
                        if not np.array_equal(z['target'],ya) or not np.array_equal(z['ids'],ids):
                            raise ValueError('Reused union audit identity')
                        prediction=z['prediction'].copy()
                    for target in range(3):
                        saved=read(folder_for(split,seed,method,target)/'refit.json')
                        seconds+=saved['seconds'];updates+=saved['updates']
                else:
                    output=[]
                    for target in range(3):
                        folder=folder_for(split,seed,method,target);chosen=read(folder/'selected.json')
                        trained=prior.pilot.base.train(x,y[:,target:target+1],None,None,subset,seed,WIDTHS,fixed_epochs=chosen['epoch'])
                        prediction=nn_predict(trained['model'],trained['norm'],xa,subset);output.append(prediction)
                        seconds+=trained['seconds'];updates+=trained['updates']
                        torch_atomic(folder/'refit_model.pt',dict(state=trained['model'].state_dict(),norm=trained['norm'],subset=subset))
                        write(folder/'refit.json',dict(epoch=trained['epoch'],updates=trained['updates'],seconds=trained['seconds']))
                    prediction=np.hstack(output)
                    path=OUT/str(split)/'predictions'/str(seed)/(method+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,prediction=prediction,target=ya,ids=ids)
                score=metrics(ya,prediction);score['mae']=abs(ya-prediction).mean(0).tolist()
                row=dict(split=split,seed=seed,method=method,audit=score,actual_input_union=len(subset),
                    per_target_inputs=[len(subset)]*3,nn_parameters=parameters,nn_epoch_caps=caps,
                    reused=method=='union',prediction_file=str(path),refit_updates=updates,
                    selection_seconds=selected_seconds,refit_seconds=seconds,
                    times_are_historical=method=='union',logical_selection_queries=12288 if method in ('union','shared') else 0)
                path=OUT/str(split)/'cases'/str(seed)/(method+'.json');write(path,row);rows.append(row)
                emit(f'CASE split={split} seed={seed} method={method} meanR2={score["mean_r2"]:.7f} reused={method=="union"}')
    write(OUT/'metrics.json',rows);write(OUT/'summary.json',summarize(rows));seal();lock()
    write(OUT/'complete.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        files={name:sha(OUT/name) for name in ('protocol.json','metrics.json','summary.json','choices_locked.json')}))
    write(OUT/'status.json',dict(state='complete_unverified',audit_opened=True,counts=counts()))


def verify():
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    p=lock();seal()
    for name,digest in read(OUT/'complete.json')['files'].items():
        if sha(OUT/name)!=digest:raise RuntimeError('Completion hash mismatch')
    panel=panels();rows=read(OUT/'metrics.json'); maximum=0.;checked=0
    with threadpool_limits(limits=1):
        runtime=version.pin_and_record()
        for split in SPLITS:
            xf,yf,xr,yr=selection_arrays(split);x,y=np.vstack([xf,xr]),np.vstack([yf,yr]);xa,ya,ids=load_audit(split)
            for seed in SEEDS:
                for method in METHODS:
                    subset=subset_for(panel,split,seed,method);predictions=[];parameter_count=0
                    for target in range(3):
                        folder=folder_for(split,seed,method,target);chosen=read(folder/'selected.json');history=read(folder/'history.json')
                        if chosen['subset']!=subset or chosen['widths']!=list(WIDTHS):raise AssertionError('Panel/width mismatch')
                        best=max(history,key=lambda z:z['mean_r2'])
                        if chosen['epoch']!=best['epoch'] or chosen['selection']!={k:best[k] for k in ('r2','mean_r2','rmse')}:raise AssertionError('Reward checkpoint mismatch')
                        duration=p['duration'];fit_batches=math.ceil(len(xf)/128);refit_batches=math.ceil(len(x)/128)
                        if not duration['min_epochs']<=chosen['epochs_run']<=duration['max_epochs'] or chosen['updates']!=fit_batches*chosen['epochs_run']:raise AssertionError('Selection duration/updates')
                        if [h['epoch'] for h in history]!=list(range(10,chosen['epochs_run']+1,10)):raise AssertionError('Incomplete reward history')
                        anchor,last,score=-float('inf'),0,-float('inf')
                        for h in history:
                            score=max(score,h['mean_r2']);anchor,last=advance_patience(score,anchor,last,h['epoch'])
                        if chosen['epochs_run']!=duration['max_epochs'] and chosen['epochs_run']-last<duration['patience']:raise AssertionError('Patience rule')
                        for phase,xx,yy in [('selected',xf,yf),('refit',x,y)]:
                            saved=torch.load(folder/(phase+'_model.pt'),map_location='cpu',weights_only=False)
                            if saved['subset']!=subset:raise AssertionError('Checkpoint columns')
                            expected=dict(xm=xx[:,subset].mean(0),xs=xx[:,subset].std(0),ym=yy[:,target:target+1].mean(0),ys=yy[:,target:target+1].std(0))
                            if any(not np.array_equal(saved['norm'][k],value) for k,value in expected.items()):raise AssertionError('Training-only scaler')
                            model=prior.pilot.base.ControlMLP(len(subset),1,WIDTHS);model.load_state_dict(saved['state']);model.eval()
                            if sum(v.numel() for v in model.parameters())!=chosen['parameters']:raise AssertionError('Parameter count')
                            if phase=='selected':
                                score=r2_score(yr[:,target:target+1],nn_predict(model,saved['norm'],xr,subset))
                                if abs(score-chosen['selection']['mean_r2'])>1e-13:raise AssertionError('Reward model reload')
                            else:predictions.append(nn_predict(model,saved['norm'],xa,subset))
                        refit=read(folder/'refit.json')
                        if refit['epoch']!=chosen['epoch'] or refit['updates']!=refit_batches*chosen['epoch']:raise AssertionError('Fixed-epoch refit')
                        parameter_count+=chosen['parameters'];checked+=1
                    prediction=np.hstack(predictions)
                    row=next(r for r in rows if (r['split'],r['seed'],r['method'])==(split,seed,method))
                    if row['nn_parameters']!=parameter_count or row['actual_input_union']!=len(subset):raise AssertionError('Group count/input budget')
                    with np.load(row['prediction_file'],allow_pickle=False) as z:
                        if not np.array_equal(z['target'],ya) or not np.array_equal(z['ids'],ids):raise AssertionError('Audit row/target identity')
                        error=float(np.max(abs(prediction-z['prediction'])));maximum=max(maximum,error)
                        if error!=0:raise AssertionError('Prediction reload mismatch')
                    expected=[r2_score(ya,prediction,multioutput='raw_values'),np.sqrt(mean_squared_error(ya,prediction,multioutput='raw_values')),mean_absolute_error(ya,prediction,multioutput='raw_values')]
                    for key,value in zip(('r2','rmse','mae'),expected):
                        if not np.allclose(value,row['audit'][key],atol=1e-12,rtol=0):raise AssertionError('Independent metric mismatch')
        if summarize(rows)!=read(OUT/'summary.json'):raise AssertionError('Summary mismatch')
    lock();seal()
    result=dict(passed=True,models=checked,pipeline_cases=len(rows),max_checkpoint_prediction_error=maximum,
        independent_validation=False,configuration_promoted=False,counts=counts())
    write(OUT/'verification.json',result)
    write(OUT/'status.json',dict(state='complete_verified',audit_opened=True,counts=counts()))
    report=['# 最终预测器匹配检查','',
        '三个历史开发划分×三个种子。四种面板均配三个独立24/23网络，不新增选谱或调参。',
        '本轮是历史开发证据；不是新来源独立验证。所有单元保留，不自动晋级或增加训练。','',
        '|面板|平均R²|SSC|pH|硬度|','|---|---:|---:|---:|---:|']
    summary=read(OUT/'summary.json')
    for m in METHODS:
        row=summary['methods'][m];report.append('|'+m+'|'+'|'.join(f'{z:.6f}' for z in [row['mean_r2'],*row['r2']])+'|')
    report+=['','|并集减参照|平均差|SSC差|pH差|硬度差|划分差(7117/8111/9103)|单元胜出|预定描述性条件|',
        '|---|---:|---:|---:|---:|---|---:|---|']
    for m,d in summary['union_minus_reference'].items():
        report.append('|'+m+'|'+'|'.join(f'{z:+.6f}' for z in [d['mean_delta'],*d['target_delta']])+'|'+
            '/'.join(f'{z:+.6f}' for z in d['partition_deltas'].values())+f'|{d["cell_wins"]}/9|'+('满足' if d['descriptive_support'] else '未满足')+'|')
    report+=['','规则：平均差>0、三个划分平均差>0、至少6/9单元胜出、三个目标总体均值差≥0。',
        '部分网络可能触及2400轮上限；本轮不证明充分收敛。并集模型与历史选谱复用，不能据本轮混合耗时宣称严格端到端加速。',
        '下一步先根据全部比较冻结候选与适用范围，再准备新来源验证协议；本入口不会启动外部实验。']
    (OUT/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(result,flush=True)


def run():
    lock()
    if read(OUT/'status.json')['state']!='prepared_not_started':
        raise RuntimeError('Run already started or complete;no automatic restart. Inspect status/failure first.')
    # Exclusive claim prevents two terminals from paying for the same model.
    with (OUT/'operator_start.json').open('x',encoding='utf-8') as stream:
        stream.write(datetime.now(timezone.utc).isoformat()+'\n')
    try:
        signal.signal(signal.SIGINT,lambda *_:setattr(prior.pilot.base,'STOP',True))
        torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        with warnings.catch_warnings(),threadpool_limits(limits=1):
            warnings.simplefilter('error');runtime=version.pin_and_record()
            if runtime['actual_interop_threads']!=1:raise RuntimeError('Actual interop1 required')
            runtime['python_executable']=sys.executable
            write(OUT/'execution_environment.json',runtime)
            write(OUT/'status.json',dict(state='selecting',audit_opened=False,counts=counts()))
            panel=panels();select_all(panel);version.pin_and_record();evaluate_all(panel);version.pin_and_record()
            verify()
        emit('COMPLETE:36cases verified;no external experiment or parameter change started')
        names=[path for path in OUT.rglob('*') if path.is_file() and path.name!='delivery_manifest.json']
        write(OUT/'delivery_manifest.json',dict(files={str(path):sha(path) for path in names}))
    except BaseException as exc:
        previous=read(OUT/'status.json')
        write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        write(OUT/'status.json',dict(state='stopped_on_problem',audit_opened=previous.get('audit_opened',False),counts=counts()))
        emit('STOPPED;keep all files;do not retry or change parameters before diagnosis')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','status'])
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='run':run()
    else:print(read(OUT/'status.json'))


if __name__=='__main__':main()
