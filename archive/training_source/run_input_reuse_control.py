"""Post-specified PLS input-visibility control; historical results stay read-only.

Compare each target's own selected inputs against the complete selected union.
Both arms use the SAME latent-component candidate range. All subsets and source
partitions are reused. Preparation never fits scientific models. Only `run`
fits models, and it is intended to be started by the user in their terminal.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from contextlib import ExitStack
from unittest.mock import patch
import argparse
import json
import sys
import time
import traceback
import warnings
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from threadpoolctl import threadpool_limits
import run_public_validation as old
from hsi_e2e.formal_runtime import read, write, sha, torch_atomic

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/input_reuse_control_001'
READY = ROOT / 'results/input_reuse_readiness_001'
DOC = ROOT / 'docs/输入复用对照协议与终端说明_2026-11-20.md'
COMPARISON = ROOT / 'docs/近邻论文逐项比较与贡献定位_2026-11-20.md'
SOURCES = ('cgl', 'corn')
ARMS = ('own', 'union')
SEEDS = (17, 29, 43)


def choose_trial(trials, grid):
    candidates = [r for r in trials if r['components'] in grid]
    if [r['components'] for r in candidates] != list(grid):
        raise ValueError('Exactly one ascending trial for each allowed component is required')
    if any(not np.isfinite(r['mean_r2']) for r in candidates):
        raise ValueError('Non-finite validation score')
    return max(candidates, key=lambda r: r['mean_r2'])  # ascending order breaks exact ties toward smaller models


def validate_panel(own, union, k, bands):
    if own != sorted(set(own)) or union != sorted(set(union)):
        raise ValueError('Subsets must be canonical and unique')
    if len(own) != k or not set(own) <= set(union) or not union or min(union) < 0 or max(union) >= bands:
        raise ValueError('Own panel is not contained in the original selected union')


def score(y, pred):
    value = old.scored(y, pred)
    if not all(np.isfinite(value[k]).all() for k in ('r2', 'rmse', 'mae')):
        raise ValueError('Non-finite prediction metric')
    return value


def aggregate(rows, targets):
    result = {}
    for source in targets:
        methods, seeds = {}, {}
        for arm in ARMS:
            group = []
            for seed in SEEDS:
                parts = []
                for row in rows:
                    if (row['source'], row['seed'], row['arm']) == (source, seed, arm):
                        with np.load(row['prediction_file'], allow_pickle=False) as z:
                            parts.append(tuple(z[k].copy() for k in ('ids', 'target', 'prediction')))
                if not parts: raise ValueError('Missing source/seed/arm prediction group')
                ids = np.concatenate([p[0] for p in parts]); order = np.argsort(ids)
                if len(ids) != len(set(ids.tolist())): raise ValueError('Repeated OOF sample ID')
                y, pred = [np.vstack([p[i] for p in parts])[order] for i in (1, 2)]
                s = score(y, pred); s.update(seed=seed, sample_count=len(ids)); group.append(s)
            seeds[arm] = group
            methods[arm] = dict(mean_r2=float(np.mean([s['mean_r2'] for s in group])),
                r2=np.mean([s['r2'] for s in group], axis=0).tolist(),
                rmse=np.mean([s['rmse'] for s in group], axis=0).tolist(),
                mae=np.mean([s['mae'] for s in group], axis=0).tolist(), sample_count_per_seed=group[0]['sample_count'])
        pairs = [dict(seed=a['seed'], mean_r2_delta=b['mean_r2']-a['mean_r2'],
            target_r2_delta=(np.array(b['r2'])-a['r2']).tolist(),
            target_rmse_delta=(np.array(b['rmse'])-a['rmse']).tolist())
            for a,b in zip(seeds['own'],seeds['union'])]
        result[source] = dict(targets=targets[source], methods=methods, seed_metrics=seeds,
            union_minus_own=dict(mean_r2_delta=methods['union']['mean_r2']-methods['own']['mean_r2'],
                target_r2_delta=(np.array(methods['union']['r2'])-methods['own']['r2']).tolist(),
                target_rmse_percent_change=(100*(np.array(methods['union']['rmse'])/methods['own']['rmse']-1)).tolist(),
                seed_wins=sum(p['mean_r2_delta']>0 for p in pairs), pairs=pairs),
            inference='Descriptive, post-specified control; no independent-fold/seed significance or universal benefit claim')
    return result


def add_original_context(summary, providers):
    """Retain the original 1..15 union result as context, without refitting it."""
    for source,e in providers.items():
        original=read(e.out/'summary.json')['methods']['union_pls']
        v=summary[source]; a=v['methods']['own']; b=v['methods']['union']
        visibility=np.array(b['r2'])-a['r2']
        component_range=np.array(original['r2'])-b['r2']
        total=np.array(original['r2'])-a['r2']
        if not np.allclose(visibility+component_range,total,atol=1e-14,rtol=0):
            raise AssertionError('Operational decomposition identity failed')
        v['original_union_1to15_context']=original
        v['operational_decomposition']=dict(
            visibility_B_minus_A=dict(mean_r2=float(visibility.mean()),target_r2=visibility.tolist()),
            component_range_C_minus_B=dict(mean_r2=float(component_range.mean()),target_r2=component_range.tolist()),
            total_C_minus_A=dict(mean_r2=float(total.mean()),target_r2=total.tolist()),
            algebraic_identity_only=True,unique_causal_mechanism_claimed=False)
    return summary


class Control:
    def __init__(self, out=None, providers=None):
        self.out = Path(out) if out is not None else OUT
        self.providers = providers if providers is not None else {s: old.Experiment(s) for s in SOURCES}
        self.selection_cache = {}

    def lock(self):
        p = read(self.out/'protocol.json')
        old.check_files(p['source_sha256']); old.check_files(p['original_sha256'])
        if old.frozen.duration.versions() != p['versions']:
            raise RuntimeError('Pinned scientific environment changed')
        if not read(READY/'verification.json')['passed']:
            raise RuntimeError('Successful readiness checks required')
        return p

    def emit(self, message):
        line = datetime.now(timezone.utc).isoformat(timespec='seconds') + ' ' + message
        print(line, flush=True)
        with (self.out/'run.log').open('a', encoding='utf-8') as f: f.write(line+'\n')

    def folder(self, entry):
        return self.out/'models'/entry['source']/entry['fold']/str(entry['seed'])/entry['arm']/str(entry['target'])

    def selection(self, source, fold):
        key = (source,fold)
        if key not in self.selection_cache:
            arrays = self.providers[source].selection(fold)
            xf,yf,xr,yr,fi,ri = arrays
            if set(fi)&set(ri) or xf.shape[1]!=xr.shape[1] or yf.shape[1]!=yr.shape[1]:
                raise ValueError('Selection identity or dimensionality mismatch')
            if any(not np.isfinite(a).all() for a in (xf,yf,xr,yr)):
                raise ValueError('Non-finite selection array')
            self.selection_cache[key] = arrays
        return self.selection_cache[key]

    def test(self, source, fold):
        self.seal()
        x,y,ids = self.providers[source].test(fold)
        *_,fi,ri = self.selection(source,fold)
        if (set(fi)|set(ri))&set(ids): raise ValueError('Training/test sample overlap')
        if len(ids)!=len(set(ids.tolist())) or len(ids)!=len(x) or len(ids)!=len(y):
            raise ValueError('Test identity mismatch')
        return x,y,ids

    def seal(self, create=False):
        paths = [self.out/'plan.json'] + sorted((self.out/'models').rglob('selected.json'))
        paths += sorted((self.out/'models').rglob('trials.json')) + sorted((self.out/'models').rglob('trial_*.pt'))
        plan = read(self.out/'plan.json')
        if len(list((self.out/'models').rglob('selected.json'))) != len(plan):
            raise RuntimeError('Global choices incomplete; test access blocked')
        expected = {str(p):sha(p) for p in paths}
        path = self.out/'choices_locked.json'
        if path.exists():
            if read(path)['files'] != expected: raise RuntimeError('New sealed choices changed')
        elif create:
            write(path,dict(created_utc=datetime.now(timezone.utc).isoformat(),files=expected))
        else: raise RuntimeError('Global new-choice seal missing; test access blocked')

    def select_all(self):
        plan = read(self.out/'plan.json'); fits = 0
        for i,e in enumerate(plan,1):
            xf,yf,xr,yr,*_ = self.selection(e['source'],e['fold']); t=e['target']; f=self.folder(e)
            f.mkdir(parents=True,exist_ok=True)
            if e['arm']=='own':
                trials=[]
                for c in e['grid']:
                    m=old.pls_fit(xf,yf[:,t:t+1],e['subset'],c); fits+=1
                    s=score(yr[:,t:t+1],old.pls_predict(m,xr))
                    trials.append(dict(components=c,**s)); torch_atomic(f/f'trial_{c}.pt',m)
                write(f/'trials.json',trials); best=choose_trial(trials,e['grid'])
                model_path=f/f'trial_{best["components"]}.pt'
                refit_reuse=None
            else:
                trials=read(e['cached_trials']); best=choose_trial(trials,e['grid'])
                model_path=Path(e['cached_folder'])/f'trial_{best["components"]}.pt'
                if best['components']!=e['cached_best_components']:
                    raise AssertionError('Prepared bounded choice changed')
                refit_reuse=e['reuse_refit']
            m=torch.load(model_path,map_location='cpu',weights_only=False)
            self.check_model(m,xf,yf[:,t:t+1],e['subset'],best['components'])
            actual=score(yr[:,t:t+1],old.pls_predict(m,xr))
            if abs(actual['mean_r2']-best['mean_r2'])>1e-12:
                raise AssertionError('Selected validation score reload mismatch')
            selected=dict(e)
            selected.update(components=best['components'],selection=actual,
                selected_model_path=str(model_path),reuse_refit=refit_reuse)
            write(f/'selected.json',selected)
            if i%24==0 or i==len(plan): self.emit(f'SELECTION {i}/{len(plan)};new_PLS_trials={fits}')
        self.seal(create=True); self.lock()
        self.emit('ALL NEW CHOICES SEALED;no new-control test prediction computed yet')
        return fits

    @staticmethod
    def check_model(m,x,y,subset,c):
        if m['subset']!=subset or m['components']!=c:
            raise AssertionError('Saved model identity mismatch')
        for name,expected in (('target_mean',y.mean(0)),('target_std',y.std(0)),('xmean',x[:,subset].mean(0)),('xstd',x[:,subset].std(0,ddof=1))):
            if not np.allclose(m[name],expected,atol=1e-13,rtol=1e-12):
                raise AssertionError('Saved model training-only scales mismatch')

    def evaluate_all(self,trial_fits):
        self.seal(); rows=[]; refits=0; reuse=0
        plan=read(self.out/'plan.json'); groups=defaultdict(list)
        for e in plan: groups[(e['source'],e['fold'],e['seed'],e['arm'])].append(e)
        tests={}
        for i,((source,fold,seed,arm),entries) in enumerate(groups.items(),1):
            xf,yf,xr,yr,*_=self.selection(source,fold); x,y=np.vstack([xf,xr]),np.vstack([yf,yr])
            if (source,fold) not in tests: tests[(source,fold)]=self.test(source,fold)
            xt,yt,ids=tests[(source,fold)]; predictions=[]; refs=[]
            for e in sorted(entries,key=lambda v:v['target']):
                f=self.folder(e); a=read(f/'selected.json'); t=e['target']
                if a['reuse_refit'] is not None:
                    path=Path(a['reuse_refit']); m=torch.load(path,map_location='cpu',weights_only=False); reused=True; reuse+=1
                else:
                    m=old.pls_fit(x,y[:,t:t+1],e['subset'],a['components']); refits+=1
                    path=f/'refit_model.pt'; torch_atomic(path,m); reused=False
                self.check_model(m,x,y[:,t:t+1],e['subset'],a['components'])
                write(f/'refit.json',dict(model_path=str(path),selected_folder=str(f),reused_original_refit=reused,
                    subset=e['subset'],target=t,components=a['components']))
                predictions.append(old.pls_predict(m,xt)); refs.append(str(f/'refit.json'))
            prediction=np.hstack(predictions); s=score(yt,prediction)
            path=self.out/'predictions'/source/fold/str(seed)/(arm+'.npz'); path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,ids=ids,target=yt,prediction=prediction)
            rows.append(dict(source=source,fold=fold,seed=seed,arm=arm,score=s,prediction_file=str(path),refit_entries=refs))
            self.emit(f'CASE {i}/{len(groups)} source={source} fold={fold} seed={seed} arm={arm} meanR2={s["mean_r2"]:.7f}')
        write(self.out/'metrics.json',rows)
        write(self.out/'summary.json',add_original_context(aggregate(rows,{s:e.targets for s,e in self.providers.items()}),self.providers))
        cost=dict(new_selection_pls_fits=trial_fits,new_refit_pls_fits=refits,
            reused_original_union_selection_models=sum(e['arm']=='union' for e in plan),
            reused_original_union_refit_models=reuse,new_search_fits=0,new_nn_models=0,
            logical_target_models=len(plan),verification_new_fits=0)
        write(self.out/'costs.json',cost)
        write(self.out/'complete.json',dict(files={n:sha(self.out/n) for n in
            ('protocol.json','plan.json','metrics.json','summary.json','costs.json','choices_locked.json')}))

    def verify(self):
        self.lock(); self.seal(); count=0; trials_checked=0; max_error=0.; own_union={}
        for n,h in read(self.out/'complete.json')['files'].items():
            if sha(self.out/n)!=h: raise AssertionError('Completed new output changed')
        plan=read(self.out/'plan.json')
        for e in plan:
            xf,yf,xr,yr,*_=self.selection(e['source'],e['fold']); t=e['target']; f=self.folder(e); a=read(f/'selected.json')
            if any(a[k]!=e[k] for k in e): raise AssertionError('Selection no longer agrees with prepared plan')
            if a['components'] not in e['grid']: raise AssertionError('Component outside common grid')
            if e['arm']=='own':
                trials=read(f/'trials.json')
                for v in trials:
                    m=torch.load(f/f'trial_{v["components"]}.pt',map_location='cpu',weights_only=False)
                    self.check_model(m,xf,yf[:,t:t+1],e['subset'],v['components'])
                    if abs(score(yr[:,t:t+1],old.pls_predict(m,xr))['mean_r2']-v['mean_r2'])>1e-12:
                        raise AssertionError('New trial validation prediction mismatch')
                    trials_checked+=1
            else: trials=read(e['cached_trials'])
            best=choose_trial(trials,e['grid'])
            if a['components']!=best['components']: raise AssertionError('Validation-best common-grid model not selected')
            pair=(e['source'],e['fold'],e['seed'],e['target'])
            own_union.setdefault(pair,{})[e['arm']]=e
        for arms in own_union.values():
            a,b=arms['own'],arms['union']
            if a['grid']!=b['grid'] or a['global_input_union']!=b['global_input_union'] or not set(a['subset'])<=set(b['subset']):
                raise AssertionError('Paired inputs or common grid mismatch')
        tests={}; rows=read(self.out/'metrics.json')
        for row in rows:
            key=(row['source'],row['fold'])
            if key not in tests: tests[key]=self.test(*key)
            xt,yt,ids=tests[key]; predictions=[]
            xf,yf,xr,yr,*_=self.selection(*key); x,y=np.vstack([xf,xr]),np.vstack([yf,yr])
            for t,path in enumerate(row['refit_entries']):
                a=read(path); m=torch.load(a['model_path'],map_location='cpu',weights_only=False)
                self.check_model(m,x,y[:,t:t+1],a['subset'],a['components'])
                predictions.append(old.pls_predict(m,xt)); count+=1
            pred=np.hstack(predictions)
            with np.load(row['prediction_file'],allow_pickle=False) as z:
                if not np.array_equal(ids,z['ids']) or not np.array_equal(yt,z['target']): raise AssertionError('OOF identity mismatch')
                err=float(np.max(abs(pred-z['prediction']))); max_error=max(max_error,err)
                if err: raise AssertionError('Final prediction reload differs')
            for name,value in [('r2',r2_score(yt,pred,multioutput='raw_values')),
                ('rmse',np.sqrt(mean_squared_error(yt,pred,multioutput='raw_values'))),
                ('mae',mean_absolute_error(yt,pred,multioutput='raw_values'))]:
                if not np.allclose(value,row['score'][name],atol=1e-12,rtol=0): raise AssertionError('Independent metrics differ')
        summary=add_original_context(aggregate(rows,{s:e.targets for s,e in self.providers.items()}),self.providers)
        if summary!=read(self.out/'summary.json'): raise AssertionError('Pooled OOF summary changed')
        # Independently verify the source-level pooling as well as the fold scores.
        for source,v in summary.items():
            for arm in ARMS:
                for seed_record in v['seed_metrics'][arm]:
                    parts=[]
                    for row in rows:
                        if (row['source'],row['arm'],row['seed'])==(source,arm,seed_record['seed']):
                            with np.load(row['prediction_file'],allow_pickle=False) as z: parts.append((z['target'].copy(),z['prediction'].copy()))
                    yy,pp=[np.vstack([p[i] for p in parts]) for i in (0,1)]
                    if not np.allclose(r2_score(yy,pp,multioutput='raw_values'),seed_record['r2'],atol=1e-12,rtol=0):
                        raise AssertionError('Independent pooled R2 mismatch')
        cost=read(self.out/'costs.json')
        expected_refits=sum(e['arm']=='own' or e['reuse_refit'] is None for e in plan)
        expected_reuse=sum(e['arm']=='union' and e['reuse_refit'] is not None for e in plan)
        if (cost['new_selection_pls_fits'],cost['new_refit_pls_fits'],cost['reused_original_union_refit_models'])!=(trials_checked,expected_refits,expected_reuse):
            raise AssertionError('Actual paid fit/reuse accounting mismatch')
        self.lock(); self.seal()
        proof=dict(passed=True,pipeline_cases=len(rows),logical_target_models=count,new_selection_trials=trials_checked,
            max_prediction_reload_error=max_error,verification_new_fits=0,new_searches=0,new_nn_models=0,
            equal_global_input_union_verified=True,same_component_candidate_ranges_verified=True,
            post_specified_after_test_exposure=True,new_independent_confirmation_claimed=False)
        write(self.out/'verification.json',proof); return proof

    def report(self):
        s=read(self.out/'summary.json')
        lines=['# 相同公共输入集合下的PLS输入复用对照','',
            '这是已知CGL和Corn结果之后登记的机制检查。原选谱、划分及结果不变；不构成新的独立盲测。', '',
            '两组所需的原始输入并集相同，各目标可见输入不同；候选成分范围相同，不代表模型自由度完全相同。', '']
        for source,v in s.items():
            lines+=['## '+source.upper(),'', '|组别|平均R²|'+'|'.join(v['targets'])+'|', '|---|---:|'+'|'.join(['---:']*len(v['targets']))+'|']
            for arm,label in [('own','各目标仅自身子集'),('union','各目标使用完整并集')]:
                a=v['methods'][arm]; lines.append('|'+label+'|'+'|'.join(f'{x:.6f}' for x in [a['mean_r2'],*a['r2']])+'|')
            a=v['original_union_1to15_context']
            lines.append('|既有并集，1—15成分范围|'+'|'.join(f'{x:.6f}' for x in [a['mean_r2'],*a['r2']])+'|')
            d=v['union_minus_own']
            lines+=['',f'并集减自身子集平均R²差：{d["mean_r2_delta"]:+.6f}；算法种子总体胜出{d["seed_wins"]}/3。',
                '逐目标RMSE变化百分比：'+', '.join(f'{name} {value:+.2f}%' for name,value in zip(v['targets'],d['target_rmse_percent_change']))+'。','']
            d=v['operational_decomposition']
            lines+=['描述性分解：B−A（同候选范围下的输入可见范围变化）'
                f'{d["visibility_B_minus_A"]["mean_r2"]:+.6f}；'
                f'C−B（同并集下成分候选范围变化）{d["component_range_C_minus_B"]["mean_r2"]:+.6f}。',
                '两项之和等于C−A；这是固定配方之间的算术分解，不证明唯一化学机制或完全相同的模型自由度。','']
        lines+=['不把折或种子当独立样本；不据测试分数更改K、成分范围、种子或选择器。',
            '本对照只衡量增加对既有输入的可见范围带来的总收益，不证明跨目标化学互补是唯一机制；不宣称硬件成本下降。']
        (self.out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    def run(self):
        self.lock()
        if read(self.out/'status.json')['state']!='prepared_not_started': raise RuntimeError('Already started; no automatic retry')
        with (self.out/'operator_start.txt').open('x',encoding='utf-8') as f: f.write(datetime.now(timezone.utc).isoformat()+'\n')
        start=time.perf_counter()
        try:
            old.configure_runtime()
            with warnings.catch_warnings(),threadpool_limits(limits=1):
                warnings.simplefilter('error')
                write(self.out/'status.json',dict(state='selecting',new_control_test_access=False))
                fits=self.select_all()
                write(self.out/'status.json',dict(state='refitting_and_evaluating',new_control_test_access=True))
                self.evaluate_all(fits)
                with (patch.object(old,'pls_fit',side_effect=AssertionError('Verification fitted model')),
                      patch('sklearn.cross_decomposition.PLSRegression.fit',side_effect=AssertionError('Verification fitted PLS'))):
                    proof=self.verify()
                self.report()
            write(self.out/'status.json',dict(state='complete_verified',new_control_test_access=True,elapsed_seconds=time.perf_counter()-start))
            self.emit('COMPLETE input reuse control verified;no extra search,NN or parameter changes')
            write(self.out/'delivery_manifest.json',dict(files={str(p):sha(p) for p in self.out.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'}))
            print(json.dumps(proof,ensure_ascii=True),flush=True)
        except BaseException as exc:
            state=read(self.out/'status.json')
            write(self.out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
            write(self.out/'status.json',dict(state='stopped_on_problem',new_control_test_access=state.get('new_control_test_access',False)))
            self.emit('STOPPED on problem;retain files;no automatic retry'); raise


def build_plan(providers):
    plan=[]
    for source,e in providers.items():
        panels=read(e.out/'panels.json')
        for fold in e.folds:
            xf,yf,xr,yr,*_=e.selection(fold)
            if yf.shape[1]!=e.q or yr.shape[1]!=e.q: raise ValueError('Target count mismatch')
            k=e.k; grid=list(range(1,min(k,len(xf)-1)+1))
            if grid[-1]!=k: raise ValueError('Original own input budget infeasible')
            for seed in SEEDS:
                p=panels[fold][str(seed)]; union=p['union']
                if sorted(set().union(*map(set,p['individual'])))!=union or len(union)>e.cap:
                    raise AssertionError('Original union provenance mismatch')
                for arm in ARMS:
                    for t in range(e.q):
                        own=p['individual'][t]; validate_panel(own,union,k,xf.shape[1])
                        base=dict(source=source,fold=fold,seed=seed,target=t,arm=arm,
                                  subset=own if arm=='own' else union,global_input_union=union,grid=grid)
                        if arm=='union':
                            folder=e.pls_path(fold,seed,'union_pls',t); trials=read(folder/'trials.json')
                            best=choose_trial(trials,grid); chosen=read(folder/'selected.json')
                            refit=read(folder/'refit.json')
                            base.update(cached_folder=str(folder),cached_trials=str(folder/'trials.json'),
                                cached_best_components=best['components'],
                                reuse_refit=refit['model_path'] if best['components']==chosen['components'] else None)
                        else: base.update(reuse_refit=None)
                        plan.append(base)
    return plan


def prepare():
    if OUT.exists(): raise RuntimeError('New protocol output already exists; do not overwrite')
    ready=read(READY/'verification.json')
    if not ready['passed']: raise RuntimeError('Synthetic tests must pass before preparation')
    providers={s:old.Experiment(s) for s in SOURCES}
    original={}
    for source,e in providers.items():
        e.lock(); e.seal()
        if read(e.out/'status.json')['state']!='complete_verified': raise RuntimeError('Original source not complete')
        manifest=e.out/'delivery_manifest.json'; original.update(read(manifest)['files']); original[str(manifest)]=sha(manifest)
        p=read(e.out/'protocol.json'); original.update(p['source_sha256']); original.update(p['input_sha256'])
    old.check_files(original)
    plan=build_plan(providers)
    counts=dict(target_pairs=len(plan)//2,logical_target_models=len(plan),
        new_selection_pls_fits=sum(len(e['grid']) for e in plan if e['arm']=='own'),
        new_refit_pls_fits=sum(e['arm']=='own' or e['reuse_refit'] is None for e in plan),
        reused_original_union_refit_models=sum(e['arm']=='union' and e['reuse_refit'] is not None for e in plan),
        new_search_fits=0,new_nn_models=0)
    OUT.mkdir(); write(OUT/'plan.json',plan)
    files=[Path(__file__),ROOT/'tests/test_input_reuse_control.py',ROOT/'prepare_input_reuse_readiness.py',
           DOC,COMPARISON,READY/'verification.json',OUT/'plan.json']
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='At the same global input union, does letting each target read all union inputs help relative to its own subset?',
        sources=list(SOURCES),arms=list(ARMS),algorithm_seeds=list(SEEDS),
        common_component_limits={s:e.k for s,e in providers.items()},counts=counts,
        primary_metric='Source-specific equal-target mean R2,OOF pooled per seed;all target RMSE/MAE reported',
        inference='Post-specified mechanism control on already-exposed test sources;not fresh confirmation',
        scientific_scope='Total effect of input visibility with the same component candidate range;not equal model degrees of freedom or isolated chemical complementarity',
        no_new_selector=True,no_new_network=True,no_new_hyperparameter_ranges=True,
        component_choice_rule='Same existing component candidates restricted to own-subset size;validation best;exact ties choose smaller',
        original_union_context='Existing1through15results retained unchanged;CminusB describes candidate-range difference without new fitting',
        no_post_test_adjustments=True,on_problem='Stop and retain outputs;no automatic retry or silent changed protocol',
        runtime='Existing hsi environment;CPU Torch4/BLAS1/interop1;no GPU needed for these PLS fits',
        versions=old.frozen.duration.versions(),source_sha256={str(f):sha(f) for f in files},original_sha256=original)
    write(OUT/'protocol.json',p); write(OUT/'status.json',dict(state='prepared_not_started',new_control_test_access=False))
    Control().lock(); old.check_files(original)
    print(json.dumps(dict(prepared_only=True,counts=counts,scientific_training_started=False),ensure_ascii=True))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=['prepare','run','status']); a=p.parse_args()
    if a.action=='prepare': prepare()
    elif a.action=='run': Control().run()
    else: print(json.dumps(read(OUT/'status.json'),ensure_ascii=True))


if __name__=='__main__': main()
