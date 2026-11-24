"""Frozen public-source comparison. Preparation does not fit scientific data."""
from pathlib import Path
from datetime import datetime, timezone
from types import MethodType
import argparse
import json
import math
import signal
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
import run_ph_quota_check as frozen
from empirical_budget_core import GenericPLSEvaluator, covsel
from hsi_e2e.pool_boundary import CapacitySearch, PaidReplayEvaluator
from hsi_e2e.framework_screen import metrics, pls_fit, pls_predict, advance_patience
from hsi_e2e.formal_runtime import read, write, sha, torch_atomic

DATA = ROOT/'data/public_validation_v1'
OUT = ROOT/'results/public_validation_001'
READY = ROOT/'results/public_validation_readiness_001'
DOC = ROOT/'docs/公开数据冻结验证_协议与终端说明_2026-11-19.md'
RESOURCE = ROOT/'results/public_resource_gate_001/resource_decisions.json'
SEEDS = (17,29,43); WIDTHS = (24,23); COMPONENTS = 8; QUERY_CAP = 12288
DURATION = dict(min_epochs=2400,max_epochs=4800,patience=1200)
PLS_GRID = tuple(range(1,16))
NN = ('union_nn','shared_nn','covsel_nn','full_nn')
PLS = ('union_pls','covsel_pls','full_pls')
METHODS = NN+PLS
base = frozen.base


def source_files():
    return list(dict.fromkeys(frozen.sources()+[Path(__file__),ROOT/'tests/test_public_validation.py']))


def stop_check():
    if base.STOP: raise InterruptedError('Operator stop; no automatic continuation')


def check_files(files):
    for path,digest in files.items():
        if sha(path)!=digest: raise RuntimeError('Frozen artifact changed: '+path)


def normalized(value): return json.dumps(value,sort_keys=True,allow_nan=False)


def configure_runtime():
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)


def scored(y,p):
    result=metrics(y,p); result['mae']=abs(y-p).mean(0).tolist(); return result


class Experiment:
    def __init__(self,source):
        if source not in ('cgl','corn'): raise ValueError('Only publicly audited sources are enabled')
        self.source=source; self.out=OUT/source; self.meta=read(DATA/'source_audit.json')[source]
        self.targets=self.meta['targets']; self.q=len(self.targets); self.cap=self.meta['input_cap']
        self.folds=['official'] if source=='cgl' else list(self.meta['fold_records'])
        self.folder=DATA/source
        if self.cap%self.q or QUERY_CAP%self.q: raise ValueError('Equal quota/query allocation must be integral')
        self.k=self.cap//self.q; self.budget=QUERY_CAP//self.q
        if COMPONENTS>self.k: raise ValueError('Fixed PLS8 proxy infeasible')

    def cells(self): return [(fold,seed) for fold in self.folds for seed in SEEDS]
    def selection(self,fold):
        with np.load(self.folder/fold/'selection.npz',allow_pickle=False) as z:
            return tuple(z[n].copy() for n in ('X_fit','y_fit','X_reward','y_reward','fit_ids','reward_ids'))

    def lock(self):
        p=read(self.out/'protocol.json'); check_files(p['source_sha256']); check_files(p['input_sha256'])
        if p['versions']!=frozen.duration.versions(): raise RuntimeError('Environment changed')
        if (p['input_cap'],p['target_count'],p['quota_per_target'],p['search_queries_per_target'])!=(self.cap,self.q,self.k,self.budget):
            raise RuntimeError('Source configuration changed')
        frozen.lock(); frozen.seal()
        return p

    def emit(self,message):
        line=datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+message
        print(line,flush=True)
        with (self.out/'run.log').open('a',encoding='utf-8') as f: f.write(line+'\n')

    def search_path(self,fold,seed,task): return self.out/'search'/fold/str(seed)/task
    def nn_path(self,fold,seed,method,t): return self.out/'nn'/fold/str(seed)/method/str(t)
    def pls_path(self,fold,seed,method,t):
        # CovSel/full PLS are deterministic at a fixed fold. Reuse across seeds.
        cache_seed=seed if method=='union_pls' else SEEDS[0]
        return self.out/'pls'/fold/str(cache_seed)/method/str(t)

    def subset(self,panels,fold,seed,method): return panels[fold][str(seed)][method.split('_')[0]]

    def search(self,xf,yf,xr,yr,seed,k,budget,path):
        ev=GenericPLSEvaluator(xf,yf,xr,yr,budget,k=k,components=COMPONENTS)
        state=CapacitySearch(ev,seed,'greedy'); tick=time.perf_counter()
        try:
            stop_check(); state.initialize()
            while ev.fit_count<budget:
                stop_check(); state.one_batch()
                if ev.fit_count%1024==0: self.emit(f'SEARCH seed={seed} K={k} fits={ev.fit_count}/{budget}')
            if ev.fit_count!=budget or ev.cache_hits: raise AssertionError('Unique paid budget mismatch')
            return list(ev.best_subset)
        finally:
            state.elapsed_seconds=time.perf_counter()-tick
            write(path/'ledger.json',ev.ledger); write(path/'decisions.json',state.decisions)
            write(path/'boundary_events.json',getattr(state,'boundary_events',[]))
            write(path/'summary.json',state.summary() if ev.best_subset is not None else dict(incomplete=True,actual_fits=ev.fit_count))

    def search_all(self):
        panels={}
        for fold in self.folds:
            xf,yf,xr,yr,*_=self.selection(fold); tick=time.perf_counter(); cs=covsel(xf,yf,self.cap)
            cs['seconds']=time.perf_counter()-tick; write(self.out/'covsel'/fold/'selection.json',cs)
            for seed in SEEDS:
                own=[]
                for t in range(self.q):
                    self.emit(f'START SEARCH fold={fold} seed={seed} target={t} quota={self.k}')
                    own.append(self.search(xf,yf[:,t:t+1],xr,yr[:,t:t+1],seed,self.k,self.budget,self.search_path(fold,seed,str(t))))
                union=sorted(set().union(*map(set,own)))
                shared=self.search(xf,yf,xr,yr,seed,self.cap,QUERY_CAP,self.search_path(fold,seed,'shared'))
                if not self.k<=len(union)<=self.cap: raise AssertionError('Union cap')
                panels.setdefault(fold,{})[str(seed)]=dict(individual=own,union=union,shared=shared,covsel=cs['subset'],full=list(range(xf.shape[1])))
                write(self.out/'panels.json',panels)
        return panels

    def select_all(self,panels):
        trained=0; pls_count=0
        for fold,seed in self.cells():
            xf,yf,xr,yr,*_=self.selection(fold)
            for method in NN:
                subset=self.subset(panels,fold,seed,method)
                for t in range(self.q):
                    stop_check(); self.emit(f'SELECT fold={fold} seed={seed} method={method} target={t} columns={len(subset)}')
                    a=base.train(xf,yf[:,t:t+1],xr,yr[:,t:t+1],subset,seed,WIDTHS,**DURATION)
                    f=self.nn_path(fold,seed,method,t); f.mkdir(parents=True,exist_ok=True)
                    torch_atomic(f/'selected_model.pt',dict(state=a['model'].state_dict(),norm=a['norm'],subset=subset))
                    write(f/'history.json',a['history'])
                    write(f/'selected.json',dict(fold=fold,seed=seed,method=method,target=t,subset=subset,widths=list(WIDTHS),
                        parameters=sum(p.numel() for p in a['model'].parameters()),**{k:a[k] for k in ('epoch','epochs_run','updates','seconds','selection')}))
                    trained+=1; write(self.out/'status.json',dict(state='selecting',test_arrays_opened=False,new_nn_models=trained))
            for method in PLS:
                subset=self.subset(panels,fold,seed,method)
                for t in range(self.q):
                    f=self.pls_path(fold,seed,method,t)
                    if (f/'selected.json').exists(): continue
                    f.mkdir(parents=True,exist_ok=True); trials=[]; best=None; tick=time.perf_counter()
                    for components in PLS_GRID:
                        if components>min(len(subset),len(xf)-1): continue
                        stop_check(); model=pls_fit(xf,yf[:,t:t+1],subset,components)
                        score=metrics(yr[:,t:t+1],pls_predict(model,xr)); trials.append(dict(components=components,**score))
                        torch_atomic(f/f'trial_{components}.pt',model)
                        if best is None or score['mean_r2']>best[1]['mean_r2']: best=(model,score)
                    if best is None: raise ValueError('No feasible declared PLS component')
                    torch_atomic(f/'selected_model.pt',best[0]); write(f/'trials.json',trials)
                    write(f/'selected.json',dict(fold=fold,cache_seed=seed if method=='union_pls' else SEEDS[0],method=method,target=t,
                        subset=subset,components=best[0]['components'],selection=best[1],tuning_fits=len(trials),seconds=time.perf_counter()-tick))
                    pls_count+=1
        self.seal(create=True); self.lock()
        self.emit(f'ALL {trained}NN AND {pls_count}UNIQUE PLS CHOICES SEALED;current test arrays unopened')

    def choice_files(self):
        paths=[self.out/'panels.json']
        for fold in self.folds: paths.append(self.out/'covsel'/fold/'selection.json')
        for fold,seed in self.cells():
            for task in [*map(str,range(self.q)),'shared']:
                paths += [self.search_path(fold,seed,task)/n for n in ('ledger.json','summary.json','decisions.json','boundary_events.json')]
            for method in NN:
                for t in range(self.q): paths += [self.nn_path(fold,seed,method,t)/n for n in ('selected.json','selected_model.pt','history.json')]
            for method in PLS:
                for t in range(self.q):
                    f=self.pls_path(fold,seed,method,t)
                    paths += [f/n for n in ('selected.json','selected_model.pt','trials.json')]
                    paths += list(f.glob('trial_*.pt'))
        return list(dict.fromkeys(paths))

    def seal(self,create=False):
        values={str(p):sha(p) for p in self.choice_files()}; f=self.out/'choices_locked.json'
        if f.exists():
            if read(f)['files']!=values: raise RuntimeError('Sealed choices changed')
        elif create: write(f,dict(utc=datetime.now(timezone.utc).isoformat(),files=values))
        else: raise RuntimeError('Global selection seal missing')

    def test(self,fold):
        if not (self.out/'choices_locked.json').exists(): raise RuntimeError('Test access blocked before global seal')
        self.seal(); self.lock()
        if self.source=='cgl':
            with np.load(self.folder/fold/'test_features.npz',allow_pickle=False) as z: x,ids=z['X_test'].copy(),z['test_ids'].copy()
            # First numeric Ytest parse in this source workflow happens here.
            z=loadmat(self.meta['raw_mat'],struct_as_record=False,squeeze_me=True,variable_names=['Ytest'])
            y=np.asarray(z['Ytest'].data,dtype=float)
            names=[str(v).strip() for v in z['Ytest'].label[1,0]]
            if names!=self.targets: raise ValueError('Official test response order mismatch')
        else:
            with np.load(self.folder/fold/'test.npz',allow_pickle=False) as z:
                x,y,ids=(z[k].copy() for k in ('X_test','y_test','test_ids'))
        if y.shape!=(len(x),self.q) or not np.isfinite(y).all() or min(y.std(0))<=1e-12: raise ValueError('Invalid test responses')
        return x,y,ids

    def evaluate_all(self,panels):
        self.seal(); self.lock(); rows=[]; pls_cache={}; nn_refits=0
        write(self.out/'status.json',dict(state='refitting_and_evaluating',test_arrays_opened=True,new_nn_models=len(self.cells())*len(NN)*self.q))
        for fold in self.folds:
            xf,yf,xr,yr,fit_ids,reward_ids=self.selection(fold); x,y=np.vstack([xf,xr]),np.vstack([yf,yr]); xt,yt,ids=self.test(fold)
            if set(fit_ids)&set(reward_ids) or (set(fit_ids)|set(reward_ids))&set(ids): raise AssertionError('Partition overlap')
            for seed in SEEDS:
                for method in METHODS:
                    subset=self.subset(panels,fold,seed,method); predictions=[]; entries=[]
                    for t in range(self.q):
                        stop_check()
                        if method in NN:
                            f=self.nn_path(fold,seed,method,t); a=read(f/'selected.json')
                            if (a['fold'],a['seed'],a['method'])!=(fold,seed,method): raise AssertionError('NN seed/method identity')
                            self.emit(f'REFIT fold={fold} seed={seed} method={method} target={t} epochs={a["epoch"]}')
                            m=base.train(x,y[:,t:t+1],None,None,subset,seed,WIDTHS,fixed_epochs=a['epoch'])
                            path=f/'refit_model.pt'; torch_atomic(path,dict(state=m['model'].state_dict(),norm=m['norm'],subset=subset))
                            entry=dict(subset=subset,epoch=a['epoch'],updates=m['updates'],seconds=m['seconds'],target=t,model_path=str(path),selected_folder=str(f))
                            write(f/'refit.json',entry); entries.append(str(f/'refit.json')); nn_refits+=1
                            predictions.append(frozen.history.predict_saved(path,xt))
                        else:
                            f=self.pls_path(fold,seed,method,t); key=str(f)
                            if key not in pls_cache:
                                a=read(f/'selected.json'); tick=time.perf_counter(); m=pls_fit(x,y[:,t:t+1],subset,a['components'])
                                torch_atomic(f/'refit_model.pt',m)
                                entry=dict(subset=subset,components=a['components'],target=t,seconds=time.perf_counter()-tick,model_path=str(f/'refit_model.pt'),selected_folder=str(f),fits=1)
                                write(f/'refit.json',entry); pls_cache[key]=entry
                            entry=pls_cache[key]; entries.append(str(f/'refit.json'))
                            predictions.append(pls_predict(torch.load(entry['model_path'],map_location='cpu',weights_only=False),xt))
                    prediction=np.hstack(predictions); score=scored(yt,prediction)
                    f=self.out/'predictions'/fold/str(seed)/(method+'.npz'); f.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(f,prediction=prediction,target=yt,ids=ids)
                    rows.append(dict(fold=fold,seed=seed,method=method,subset=subset,score=score,prediction_file=str(f),refit_entries=entries))
                    self.emit(f'CASE fold={fold} seed={seed} method={method} meanR2={score["mean_r2"]:.7f}')
        write(self.out/'metrics.json',rows); write(self.out/'summary.json',self.summarize(rows))
        selected_pls=list({self.pls_path(fold,seed,m,t) for fold,seed in self.cells() for m in PLS for t in range(self.q)})
        write(self.out/'costs.json',dict(new_search_pls_fits=len(self.cells())*2*QUERY_CAP,
            new_nn_selections=len(self.cells())*len(NN)*self.q,new_nn_refits=nn_refits,
            unique_pls_selection_models=len(selected_pls),pls_tuning_fits=sum(read(f/'selected.json')['tuning_fits'] for f in selected_pls),
            pls_refit_fits=len(pls_cache),logical_pls_final_models=len(self.cells())*len(PLS)*self.q,
            verification_new_fits=0,strictly_equal_total_compute=False,
            interpretation='Uniform per-selector query cap applies union/shared;CovSel and full need no PLS search;shared scientific work reused across predictors;deterministic PLS references deduplicated across algorithm seeds'))
        write(self.out/'complete.json',dict(files={n:sha(self.out/n) for n in ('protocol.json','metrics.json','summary.json','costs.json','choices_locked.json')}))

    def summarize(self,rows):
        expected={(f,s,m) for f,s in self.cells() for m in METHODS}
        if len(rows)!=len(expected) or {(r['fold'],r['seed'],r['method']) for r in rows}!=expected: raise ValueError('Incomplete unique source grid')
        methods={}; seed_metrics={}
        for method in METHODS:
            groups=[]
            for seed in SEEDS:
                parts=[]
                for row in rows:
                    if (row['seed'],row['method'])==(seed,method):
                        with np.load(row['prediction_file'],allow_pickle=False) as z: parts.append((z['ids'].copy(),z['target'].copy(),z['prediction'].copy()))
                ids=np.concatenate([p[0] for p in parts]); y=np.vstack([p[1] for p in parts]); pred=np.vstack([p[2] for p in parts])
                if len(set(ids))!=len(ids): raise AssertionError('Each source sample must have exactly one OOF prediction per seed')
                order=np.argsort(ids); score=scored(y[order],pred[order]); score['seed']=seed; groups.append(score)
            seed_metrics[method]=groups
            methods[method]=dict(mean_r2=float(np.mean([g['mean_r2'] for g in groups])),
                r2=np.mean([g['r2'] for g in groups],axis=0).tolist(),rmse=np.mean([g['rmse'] for g in groups],axis=0).tolist(),
                sample_count_per_seed=len(ids),aggregation='All heldout/OOF rows combined before R2,then mean across algorithm seeds')
        comparisons={}
        for reference in METHODS[1:]:
            pairs=[dict(seed=seed,delta=(np.array(a['r2'])-b['r2']).tolist(),mean_delta=a['mean_r2']-b['mean_r2'])
                   for seed,a,b in zip(SEEDS,seed_metrics['union_nn'],seed_metrics[reference])]
            target=np.mean([p['delta'] for p in pairs],axis=0); mean=float(target.mean())
            folds={f:float(np.mean([next(r for r in rows if (r['fold'],r['seed'],r['method'])==(f,seed,'union_nn'))['score']['mean_r2']-
                next(r for r in rows if (r['fold'],r['seed'],r['method'])==(f,seed,reference))['score']['mean_r2'] for seed in SEEDS])) for f in self.folds}
            wins=sum(p['mean_delta']>0 for p in pairs)
            comparisons[reference]=dict(mean_delta=mean,target_delta=target.tolist(),seed_wins=wins,fold_mean_deltas=folds,pairs=pairs,
                descriptive_support=bool(mean>0 and min(target)>=0 and min(folds.values())>0 and wins>=math.ceil(2*len(SEEDS)/3)))
        return dict(source=self.source,targets=self.targets,methods=methods,seed_metrics=seed_metrics,union_nn_minus_reference=comparisons,
            all_nn_reference_support=all(comparisons[m]['descriptive_support'] for m in NN[1:]),
            all_reference_support=all(v['descriptive_support'] for v in comparisons.values()),
            independent_material_source=True,independent_grape_batch=False,independent_seeds_or_folds_claimed=False,
            statistical_superiority_claimed=False,configuration_promoted=False)

    def replay_search(self,xf,yf,xr,yr,seed,k,budget,path):
        ledger=read(path/'ledger.json')
        if len(ledger)!=budget: raise AssertionError('Paid search budget')
        seen=set(); best=-float('inf'); chosen=None
        for i,row in enumerate(ledger,1):
            subset=tuple(row['subset'])
            if row['actual_fits']!=i or len(subset)!=k or tuple(sorted(set(subset)))!=subset or subset in seen or min(subset)<0 or max(subset)>=xf.shape[1]:
                raise AssertionError('Canonical unique search record')
            if not np.isfinite(row['r2']).all() or abs(float(np.mean(row['r2']))-row['mean_r2'])>1e-13: raise AssertionError('Search objective record')
            if row['mean_r2']>best: best=row['mean_r2']; chosen=list(subset)
            if row['incumbent_mean_r2']!=best: raise AssertionError('Paid best prefix')
            seen.add(subset)
        ev=GenericPLSEvaluator(xf,yf,xr,yr,budget,k=k,components=COMPONENTS)
        ev.replay_log=ledger; ev.replay_target=budget
        ev.replay_records={tuple(r['subset']):dict(r2=np.array(r['r2']),nmse=r['nmse'],mean_r2=r['mean_r2']) for r in ledger}
        ev.evaluate=MethodType(PaidReplayEvaluator.evaluate,ev); state=CapacitySearch(ev,seed,'greedy'); state.initialize()
        while ev.fit_count<budget: stop_check(); state.one_batch()
        old=read(path/'summary.json')
        if normalized(state.decisions)!=normalized(read(path/'decisions.json')) or normalized(getattr(state,'boundary_events',[]))!=normalized(read(path/'boundary_events.json')):
            raise AssertionError('Exact paid-path replay mismatch')
        if list(ev.best_subset)!=chosen or old['best']['subset']!=chosen or ev.cache_hits or old['cache_hits'] or old['context_sha256']!=ev.context_hash:
            raise AssertionError('Search endpoint/context mismatch')
        return chosen

    def verify(self):
        from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
        self.lock(); self.seal()
        for n,h in read(self.out/'complete.json')['files'].items():
            if sha(self.out/n)!=h: raise AssertionError('Completed outputs changed')
        panels=read(self.out/'panels.json'); rows=read(self.out/'metrics.json'); count=0; max_error=0.; trials_count=0; checked_pls=set()
        for fold in self.folds:
            xf,yf,xr,yr,*_=self.selection(fold); x,y=np.vstack([xf,xr]),np.vstack([yf,yr]); xt,yt,ids=self.test(fold)
            cs=read(self.out/'covsel'/fold/'selection.json')
            # No refitting is needed to replay selectors or reload prediction models.
            for seed in SEEDS:
                own=[self.replay_search(xf,yf[:,t:t+1],xr,yr[:,t:t+1],seed,self.k,self.budget,self.search_path(fold,seed,str(t))) for t in range(self.q)]
                shared=self.replay_search(xf,yf,xr,yr,seed,self.cap,QUERY_CAP,self.search_path(fold,seed,'shared'))
                expected=dict(individual=own,union=sorted(set().union(*map(set,own))),shared=shared,covsel=cs['subset'],full=list(range(xf.shape[1])))
                if panels[fold][str(seed)]!=expected or len(expected['union'])>self.cap: raise AssertionError('Panel provenance')
                for method in METHODS:
                    row=next(r for r in rows if (r['fold'],r['seed'],r['method'])==(fold,seed,method)); subset=self.subset(panels,fold,seed,method); predictions=[]
                    if row['subset']!=subset: raise AssertionError('Case panel mismatch')
                    for t,entry_path in enumerate(row['refit_entries']):
                        f=self.nn_path(fold,seed,method,t) if method in NN else self.pls_path(fold,seed,method,t)
                        a=read(f/'selected.json'); e=read(entry_path)
                        if a['subset']!=subset or a['target']!=t or e['subset']!=subset or e['target']!=t or Path(e['selected_folder'])!=f:
                            raise AssertionError('Predictor/refit identity')
                        if method in NN:
                            h=read(f/'history.json')
                            if (a['fold'],a['seed'],a['method'])!=(fold,seed,method): raise AssertionError('NN seed/method identity')
                            if a['widths']!=list(WIDTHS) or not DURATION['min_epochs']<=a['epochs_run']<=DURATION['max_epochs'] or a['updates']!=math.ceil(len(xf)/128)*a['epochs_run']:
                                raise AssertionError('NN profile/updates')
                            if [v['epoch'] for v in h]!=list(range(10,a['epochs_run']+1,10)) or a['epoch']!=max(h,key=lambda v:v['mean_r2'])['epoch']: raise AssertionError('NN best checkpoint')
                            anchor,last,best=-float('inf'),0,-float('inf')
                            for v in h: best=max(best,v['mean_r2']); anchor,last=advance_patience(best,anchor,last,v['epoch'])
                            if a['epochs_run']!=DURATION['max_epochs'] and a['epochs_run']-last<DURATION['patience']: raise AssertionError('NN patience')
                            for path,xx,yy in [(f/'selected_model.pt',xf,yf[:,t:t+1]),(Path(e['model_path']),x,y[:,t:t+1])]:
                                saved=torch.load(path,map_location='cpu',weights_only=False)
                                norm=dict(xm=xx[:,subset].mean(0),xs=xx[:,subset].std(0),ym=yy.mean(0),ys=yy.std(0))
                                if saved['subset']!=subset or any(not np.array_equal(saved['norm'][k],v) for k,v in norm.items()): raise AssertionError('NN training-only normalization')
                            value=r2_score(yr[:,t:t+1],frozen.history.predict_saved(f/'selected_model.pt',xr))
                            if abs(value-a['selection']['mean_r2'])>1e-13 or e['epoch']!=a['epoch'] or e['updates']!=math.ceil(len(x)/128)*a['epoch']: raise AssertionError('NN score/refit epochs')
                            predictions.append(frozen.history.predict_saved(e['model_path'],xt))
                        else:
                            if str(f) not in checked_pls:
                                trials=read(f/'trials.json'); feasible=[c for c in PLS_GRID if c<=min(len(subset),len(xf)-1)]
                                if [v['components'] for v in trials]!=feasible or a['tuning_fits']!=len(feasible): raise AssertionError('Declared PLS grid')
                                for v in trials:
                                    m=torch.load(f/f'trial_{v["components"]}.pt',map_location='cpu',weights_only=False)
                                    if m['subset']!=subset or m['components']!=v['components'] or not np.array_equal(m['target_mean'],yf[:,t:t+1].mean(0)) or not np.array_equal(m['target_std'],yf[:,t:t+1].std(0)) or not np.array_equal(m['xmean'],xf[:,subset].mean(0)) or not np.allclose(m['xstd'],xf[:,subset].std(0,ddof=1),atol=1e-14,rtol=1e-12):
                                        raise AssertionError('PLS fit-only scales')
                                    got=metrics(yr[:,t:t+1],pls_predict(m,xr))
                                    if any(not np.allclose(got[k],v[k],atol=1e-13,rtol=0) for k in ('r2','rmse','mean_r2')): raise AssertionError('PLS trial reload')
                                    trials_count+=1
                                best=max(trials,key=lambda v:v['mean_r2'])
                                if a['components']!=best['components'] or a['selection']['mean_r2']!=best['mean_r2']: raise AssertionError('PLS selection/tie')
                                chosen=torch.load(f/'selected_model.pt',map_location='cpu',weights_only=False)
                                if chosen['components']!=best['components'] or abs(metrics(yr[:,t:t+1],pls_predict(chosen,xr))['mean_r2']-best['mean_r2'])>1e-13: raise AssertionError('PLS selected reload')
                                checked_pls.add(str(f))
                            m=torch.load(e['model_path'],map_location='cpu',weights_only=False)
                            if m['subset']!=subset or m['components']!=a['components'] or not np.array_equal(m['target_mean'],y[:,t:t+1].mean(0)) or not np.array_equal(m['target_std'],y[:,t:t+1].std(0)) or not np.array_equal(m['xmean'],x[:,subset].mean(0)) or not np.allclose(m['xstd'],x[:,subset].std(0,ddof=1),atol=1e-14,rtol=1e-12):
                                raise AssertionError('PLS refit identity/scales')
                            predictions.append(pls_predict(m,xt))
                        count+=1
                    pred=np.hstack(predictions)
                    with np.load(row['prediction_file'],allow_pickle=False) as z:
                        if not np.array_equal(ids,z['ids']) or not np.array_equal(yt,z['target']): raise AssertionError('Test IDs/labels')
                        error=float(np.max(abs(pred-z['prediction']))); max_error=max(max_error,error)
                        if error: raise AssertionError('Prediction reload mismatch')
                    for k,v in [('r2',r2_score(yt,pred,multioutput='raw_values')),('rmse',np.sqrt(mean_squared_error(yt,pred,multioutput='raw_values'))),('mae',mean_absolute_error(yt,pred,multioutput='raw_values'))]:
                        if not np.allclose(v,row['score'][k],atol=1e-12,rtol=0): raise AssertionError('Independent metric recomputation')
        if self.summarize(rows)!=read(self.out/'summary.json'): raise AssertionError('OOF/source summary mismatch')
        cost=read(self.out/'costs.json')
        if (cost['pls_tuning_fits'],cost['unique_pls_selection_models'],cost['pls_refit_fits'])!=(trials_count,len(checked_pls),len(checked_pls)): raise AssertionError('Deduplicated PLS costs')
        if cost['new_search_pls_fits']!=len(self.cells())*2*QUERY_CAP or cost['new_nn_selections']!=len(self.cells())*len(NN)*self.q or cost['new_nn_refits']!=cost['new_nn_selections']: raise AssertionError('Search/NN costs')
        self.lock(); self.seal()
        proof=dict(passed=True,source=self.source,pipeline_cases=len(rows),logical_target_models=count,
            unique_nn_models=len(self.cells())*len(NN)*self.q,unique_pls_models=len(checked_pls),
            paid_search_fits=len(self.cells())*2*QUERY_CAP,verification_new_fits=0,
            max_prediction_reload_error=max_error,test_access_after_global_seal=True,configuration_promoted=False)
        write(self.out/'verification.json',proof); return proof

    def report(self,proof):
        s=read(self.out/'summary.json'); lines=['# 公开来源冻结流程验证：'+self.source,'',
            '各指标在该来源训练部分重新拟合；不是葡萄权重零样本迁移。不同来源结果分别报告。','',
            '|方法|平均R²|'+'|'.join(self.targets)+'|','|---|---:|'+'|'.join(['---:']*self.q)+'|']
        for m,v in s['methods'].items(): lines.append('|'+m+'|'+'|'.join(f'{a:.6f}' for a in [v['mean_r2'],*v['r2']])+'|')
        lines+=['','所有源内测试／OOF样本先合并计算每种子的R²，再对算法种子取均值；不把折或种子当独立数据集。',
            'R²、RMSE、MAE、逐种子／逐折差值与成本完整保留。公共代码不可取得的MCSWS不纳入实测，也不宣称胜过它。',
            f'重载预测最大误差{proof["max_prediction_reload_error"]}；核验不新增拟合。',
            '不因测试结果修改输入配额、模型、学习率、训练预算或基线，不自动启动下一来源或晋级方法。']
        (self.out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    def run(self):
        self.lock()
        if read(self.out/'status.json')['state']!='prepared_not_started': raise RuntimeError('Already started; no automatic retry')
        with (self.out/'operator_start.txt').open('x',encoding='utf-8') as f: f.write(datetime.now(timezone.utc).isoformat()+'\n')
        old=signal.getsignal(signal.SIGINT)
        def interrupt(*args): base.STOP=True; raise InterruptedError('Operator stopped experiment')
        try:
            signal.signal(signal.SIGINT,interrupt); base.STOP=False
            configure_runtime()
            with warnings.catch_warnings(),threadpool_limits(limits=1):
                warnings.simplefilter('error'); runtime=frozen.matched.version.pin_and_record()
                if runtime['actual_interop_threads']!=1: raise RuntimeError('Interop1 required')
                runtime['python_executable']=sys.executable; write(self.out/'environment.json',runtime)
                write(self.out/'status.json',dict(state='searching',test_arrays_opened=False))
                panels=self.search_all(); frozen.matched.version.pin_and_record()
                self.select_all(panels); frozen.matched.version.pin_and_record()
                self.evaluate_all(panels); frozen.matched.version.pin_and_record(); proof=self.verify(); self.report(proof)
            write(self.out/'status.json',dict(state='complete_verified',test_arrays_opened=True))
            self.emit('COMPLETE frozen public-source validation verified;no other source or parameter changes started')
            write(self.out/'delivery_manifest.json',dict(files={str(p):sha(p) for p in self.out.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'}))
        except BaseException as error:
            state=read(self.out/'status.json'); write(self.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
            write(self.out/'status.json',dict(state='stopped_on_problem',test_arrays_opened=state.get('test_arrays_opened',False)))
            self.emit('STOPPED;retain files;no automatic retry or changed method'); raise
        finally: signal.signal(signal.SIGINT,old)


def prepare():
    if OUT.exists(): raise RuntimeError('Existing output; no overwrite')
    frozen.lock(); frozen.seal(); check_files(read(DATA/'manifest.json')['files'])
    ready=read(READY/'verification.json')
    if not ready['passed']: raise RuntimeError('Synthetic checks required')
    check_files(ready['source_sha256'])
    decisions=read(RESOURCE)
    if not decisions['covsel']['included'] or decisions['mcsws']['included']: raise RuntimeError('Public resource policy mismatch')
    for source in ('cgl','corn'):
        e=Experiment(source)
        for fold in e.folds:
            xf,yf,xr,yr,fi,ri=e.selection(fold)
            if yf.shape!=(len(xf),e.q) or yr.shape!=(len(xr),e.q) or (set(fi)&set(ri)) or e.cap>min(len(xf)-1,xf.shape[1]-1): raise ValueError('Source selection interface')
        e.out.mkdir(parents=True)
        input_files=[DATA/'source_audit.json',DATA/'manifest.json',DOC,RESOURCE,READY/'verification.json']+[Path(p) for p in read(DATA/'manifest.json')['files']]
        p=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=source,folds=e.folds,algorithm_seeds=list(SEEDS),
            targets=e.targets,target_count=e.q,input_cap=e.cap,quota_per_target=e.k,search_queries_per_target=e.budget,
            total_queries_per_selector_cell=QUERY_CAP,actual_new_search_fits=len(e.cells())*2*QUERY_CAP,
            methods=list(METHODS),widths=list(WIDTHS),nn_duration=DURATION,pls_final_grid=list(PLS_GRID),
            nn_models=len(e.cells())*len(NN)*e.q,unique_pls_models=len(e.folds)*(len(SEEDS)+2)*e.q,
            rule='Freeze grape-derived equal-source union and independentMLP recipe;all new-source targets use the already-tested broad2400/4800/1200 duration;no new-source hyperparameter search',
            paired_controls='Shared multiresponse greedy,CovSel,full use same independentMLP;union/CovSel/full also use targetwise tunedPLS1 with1through15feasible components',
            deterministic_reuse='CovSel chosen once per fold;deterministic CovSel/fullPLS trials and refits reused across algorithm seeds',
            test_firewall='All searches,NNcheckpoints,PLStrials and choices sealed globally across folds before any current test-loader call',
            aggregation='Per seed combine all disjoint heldout/OOF rows before R2;then mean across seeds;source results stay separate',
            gate='Uniform meanR2 improves,each target overall mean delta>=0,each fold mean delta>0,at least2/3algorithmseeds improve;descriptive only;all public and linear references reported',
            cost='Same input upper budget and union/shared search query caps;not equal actual columns,total compute or wallclock;no hardware savings claim',
            source_scope='Independent material source procedure validation,not grape new-batch validation,zero-shot transfer,or independentfold/seed significance',
            exclusions=['MCSWS executable public code not obtained;excluded by user rule','No previous grape re-training','No quota sweep or test-driven changes'],
            on_failure='Stop,read paper/official code to diagnose;do not guess,retry repeatedly,or switch method using test scores;any revised run gets a new declared protocol and exposure status',
            runtime='CPU actualTorch4/BLAS1/interop1',versions=frozen.duration.versions(),
            source_sha256={str(p):sha(p) for p in source_files()},input_sha256={str(p):sha(p) for p in dict.fromkeys(input_files)})
        write(e.out/'protocol.json',p); write(e.out/'status.json',dict(state='prepared_not_started',test_arrays_opened=False))
        e.lock(); print(dict(source=source,prepared_only=True,NN_models=p['nn_models'],unique_PLS_models=p['unique_pls_models'],search_fits=p['actual_new_search_fits']),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=['prepare','run','status']); parser.add_argument('source',nargs='?',choices=['cgl','corn']); a=parser.parse_args()
    if a.action=='prepare': prepare()
    elif a.source is None: parser.error('run/status requires source cgl or corn')
    elif a.action=='run': Experiment(a.source).run()
    else: print(read(Experiment(a.source).out/'status.json'))


if __name__=='__main__': main()
