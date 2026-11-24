"""Bounded four-panel PLS-vs-final-MLP ranking check;operator starts training.

Training/ranking is restricted to existing fit/reward data. Seal all choices
globally before current historical-audit access. No new-source evaluation.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import math
import signal
import sys
import traceback
import warnings

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run_training_duration_check as duration
from panel_reranking_core import CUTS,prefix_unions,choose_panel,strict_contrast
from hsi_e2e.framework_screen import metrics,nn_predict,pls_fit,pls_predict,advance_patience
from hsi_e2e.formal_runtime import read,write,sha,torch_atomic

previous=duration.previous;base=duration.base
OUT=ROOT/'results/panel_reranking_001'
READINESS=ROOT/'results/panel_reranking_readiness_001'
SPLITS=(7117,8111,9103);SEEDS=(17,29,43)
PIDS=('U0','U1','U2','U3');BASE_PIDS=('U3','COV','FULL')
METHODS=('original','PLS','MLP','covsel','full');WIDTHS=(24,23)


def profile(target):return duration.DURATION if target in (0,1) else previous.DURATION
def cells():return [(s,seed) for s in SPLITS for seed in SEEDS]
def baseline_method(pid):return {'U3':'union','COV':'covsel','FULL':'full'}[pid]
def new_model_cases():return [(s,seed,pid,t) for s,seed in cells() for pid in PIDS[:-1] for t in range(3)]


def ledger_paths(split,seed):
    return [(previous.prior.pilot.OUT/'search'/str(seed)/str(t)/'ledger.json' if split==7117 else
        previous.prior.OUT/str(split)/'search'/str(seed)/str(t)/'ledger.json') for t in range(3)]


def build_pools():
    old=previous.panels();pools={};files=[]
    for split,seed in cells():
        paths=ledger_paths(split,seed);files+=paths
        candidates=prefix_unions([read(p) for p in paths],old[str(split)][str(seed)]['union'])
        pools.setdefault(str(split),{})[str(seed)]=dict(candidates=candidates,
            COV=old[str(split)][str(seed)]['covsel'],FULL=old[str(split)][str(seed)]['full'])
    return pools,files


def subset_for(pools,split,seed,pid):
    row=pools[str(split)][str(seed)]
    return next(p['subset'] for p in row['candidates'] if p['id']==pid) if pid in PIDS else row[pid]


def choice_folder(split,seed,pid,target):
    if pid in BASE_PIDS:
        m=baseline_method(pid)
        return duration.new_folder((split,seed,m,target)) if target in (0,1) else previous.folder_for(split,seed,m,target)
    return OUT/'models'/str(split)/str(seed)/pid/str(target)


def choice(split,seed,pid,target):
    f=choice_folder(split,seed,pid,target);a=read(f/'selected.json')
    score=a['reward'] if pid in BASE_PIDS and target in (0,1) else a['selection']
    return dict(subset=a['subset'],widths=a['widths'],epoch=a['epoch'],epochs_run=a['epochs_run'],updates=a['updates'],
        selection=score,seconds=a['seconds'],folder=str(f))


def original_refit_folder(split,seed,pid,target):
    return previous.folder_for(split,seed,baseline_method(pid),target)


def sources():
    return list(dict.fromkeys(duration.sources()+[Path(__file__),ROOT/'panel_reranking_core.py',ROOT/'tests/test_panel_reranking_check.py']))


def lock():
    p=read(OUT/'protocol.json')
    for key in ('source_sha256','input_sha256'):
        for path,digest in p[key].items():
            if sha(path)!=digest:raise RuntimeError('Locked artifact changed:'+path)
    if p['versions']!=duration.versions():raise RuntimeError('Environment changed after preparation')
    duration.lock()
    return p


def prepare():
    if OUT.exists():raise RuntimeError('Output exists;no automatic restart or overwrite')
    duration.lock();previous.lock();previous.seal()
    if read(duration.OUT/'status.json')['state']!='complete_verified':raise RuntimeError('Completed duration check required')
    for folder in (duration.OUT,previous.OUT):
        for path,digest in read(folder/'delivery_manifest.json')['files'].items():
            if sha(path)!=digest:raise RuntimeError('Historical result changed:'+path)
    proof=read(READINESS/'verification.json')
    if not proof['passed']:raise RuntimeError('Reranking entry tests required')
    for path,digest in proof['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Code changed after tests:'+path)
    pools,ledgerfiles=build_pools()
    inputs=ledgerfiles+[READINESS/'verification.json',duration.OUT/'protocol.json',duration.OUT/'delivery_manifest.json',
        previous.OUT/'protocol.json',previous.OUT/'delivery_manifest.json',previous.OUT/'panels.json',
        ROOT/'docs/最终MLP面板重排_协议与终端说明_2026-11-19.md']
    lower=upper=refit_base=refit_upper=0.;baseline_new=0
    for split in SPLITS:
        arrays=duration.selection_arrays(split)
        inputs += [previous.prior.OLD/str(split)/name for name in ('selection_data.npz','audit_data.npz')]
        for seed in SEEDS:
            for pid in (*PIDS,'COV','FULL'):
                previous.validate_selection(*arrays,subset_for(pools,split,seed,pid))
            winner_rates=[]
            for pid in BASE_PIDS:
                for t in range(3):
                    f=choice_folder(split,seed,pid,t);a=choice(split,seed,pid,t)
                    if a['subset']!=subset_for(pools,split,seed,pid) or a['widths']!=list(WIDTHS):raise ValueError('Cached predictor mismatch')
                    inputs += [f/n for n in ('selected.json','selected_model.pt','history.json')]
                    old=original_refit_folder(split,seed,pid,t);oldc=read(old/'selected.json');oldr=read(old/'refit.json')
                    inputs += [old/n for n in ('selected.json','refit_model.pt','refit.json')]
                    rate=oldr['seconds']/oldr['epoch']
                    if a['epoch']!=oldc['epoch']:
                        baseline_new+=1;refit_base+=rate*a['epoch']
                    if pid=='U3':
                        cfg=profile(t);training_rate=a['seconds']/a['epochs_run']
                        lower+=3*training_rate*cfg['min_epochs'];upper+=3*training_rate*cfg['max_epochs']
                        winner_rates.append(rate*cfg['max_epochs'])
            refit_upper+=2*sum(winner_rates)
    OUT.mkdir(parents=True)
    write(OUT/'pools.json',pools)
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),purpose='Test final-predictor reranking on one fixed four-panel pool',
        partitions=list(SPLITS),seeds=list(SEEDS),methods=list(METHODS),cuts=list(CUTS),input_cap=48,
        historical_query_cap_per_cell=12288,additional_panel_pls_scores_per_cell=4,total_logical_pls_fits_per_reranked_cell=12292,
        new_selection_models=len(new_model_cases()),cached_selection_models=len(cells())*9,new_teacher_pls_fits=len(cells())*4,
        max_new_refits=baseline_new+len(cells())*6,baseline_new_refits=baseline_new,profiles={str(t):profile(t) for t in range(3)},
        pool='Four exact original paid-prefix unions;three targets each16;all source searches stay within historical3x4096queries;no new search paths',
        ranking='PLS=highest jointPLS8 reward meanR2;MLP=highest mean of3independent best rewardR2;exact tie picks earliest prefix',
        controls='original final union,CovSel48,full spectrum;all same independent24/23 and target-specific training rules',
        fairness='Common pool and identical predictor table isolate ranking;MLP reranking adds operational training cost. No strict equal-total-compute or speedup claim against PLS/CovSel',
        cost_accounting='36newPLS panel scores and81newalternativeNN selections for9cells;cached models count separately;unique winner/reference refits deduplicated',
        logical_cost_per_cell=dict(MLP=dict(nn_selection_models=12,teacher_panel_fits=4,nn_refit_models=3),
            PLS=dict(nn_selection_models=3,teacher_panel_fits=4,nn_refit_models=3),
            original=dict(nn_selection_models=3,nn_refit_models=3),covsel=dict(nn_selection_models=3,nn_refit_models=3),full=dict(nn_selection_models=3,nn_refit_models=3)),
        seal='All candidate scores,NNcheckpoints,chosen panels and epochs globally sealed before any current audit arrays',
        evaluation='Refit fixed epochs on fit+reward;historically exposed336audit per split. Not a new blind test or independent source',
        primary='MLP versus same-poolPLS,original endpoint,andCovSel;full remains precision/input reference',
        gate='Mean difference>0,all3partitionmeans>0,at least6/9wins,all3targetmeans>=0;descriptive only',
        no_tautology='Higher selection reward is guaranteed by maximization and is not evidence of held-out gain;report actual historical audit outcomes',
        stopping='Any warning,nonfinite,source change or exception stops;no new pools,seeds,epochs,LR,width,K,task weights or automatic promotion',
        runtime='CPU actualTorch4/BLAS1/interop1',versions=duration.versions(),
        estimate_minutes=dict(selection_low=lower/60,selection_high=upper/60,baseline_refits=refit_base/60,
            maximum_winner_refits=refit_upper/60,total_low=(lower+refit_base)/60,total_high=(upper+refit_base+refit_upper)/60,
            scope='Historical per-epoch extrapolation;not a guaranteed wall clock'),
        source_sha256={str(f):sha(f) for f in sources()},input_sha256={str(f):sha(f) for f in dict.fromkeys(inputs)})
    write(OUT/'protocol.json',p);write(OUT/'status.json',dict(state='prepared_not_started',audit_opened=False,new_models_trained=0))
    lock();print(dict(prepared_only=True,candidate_panels=len(cells())*4,new_nn_selections=len(new_model_cases()),
        new_pls_panel_scores=len(cells())*4,max_new_refits=p['max_new_refits'],estimate=p['estimate_minutes']),flush=True)


def emit(message):
    line=datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+message
    print(line,flush=True)
    with (OUT/'run.log').open('a',encoding='utf-8') as f:f.write(line+'\n')


class Progress:
    def __enter__(self):
        self.original=base.metrics;self.n=0;self.best=-float('inf');base.metrics=self.score;return self
    def __exit__(self,*args):base.metrics=self.original
    def score(self,y,pred):
        score=self.original(y,pred);self.n+=1;self.best=max(self.best,score['mean_r2'])
        if self.n%20==0:emit(f'EPOCH {self.n*10} best_reward={self.best:.7f}')
        return score


def select_all(pools):
    trained_count=0
    for split in SPLITS:
        x,y,xv,yv=duration.selection_arrays(split)
        for seed in SEEDS:
            for pid in PIDS:
                subset=subset_for(pools,split,seed,pid)
                model=pls_fit(x,y,subset,8);score=metrics(yv,pls_predict(model,xv))
                f=OUT/'teacher'/str(split)/str(seed)/pid
                f.mkdir(parents=True);torch_atomic(f/'model.pt',model)
                write(f/'score.json',dict(subset=subset,components=8,selection=score,fits=1))
                if pid=='U3':continue
                for t in range(3):
                    emit(f'SELECT split={split} seed={seed} panel={pid} target={t}')
                    with Progress():trained=base.train(x,y[:,t:t+1],xv,yv[:,t:t+1],subset,seed,WIDTHS,**profile(t))
                    f=choice_folder(split,seed,pid,t);f.mkdir(parents=True)
                    torch_atomic(f/'selected_model.pt',dict(state=trained['model'].state_dict(),norm=trained['norm'],subset=subset))
                    write(f/'history.json',trained['history'])
                    values={k:trained[k] for k in ('epoch','epochs_run','updates','seconds','selection')}
                    write(f/'selected.json',dict(split=split,seed=seed,panel=pid,target=t,subset=subset,widths=list(WIDTHS),**values))
                    trained_count+=1;write(OUT/'status.json',dict(state='selecting',audit_opened=False,new_models_trained=trained_count))
    choices={};tables={}
    for split,seed in cells():
        teacher={pid:read(OUT/'teacher'/str(split)/str(seed)/pid/'score.json')['selection']['mean_r2'] for pid in PIDS}
        neural={pid:[choice(split,seed,pid,t)['selection']['mean_r2'] for t in range(3)] for pid in PIDS}
        chosen=choose_panel(PIDS,teacher,neural);chosen.update(original='U3',covsel='COV',full='FULL')
        choices.setdefault(str(split),{})[str(seed)]=chosen
        tables.setdefault(str(split),{})[str(seed)]=dict(teacher=teacher,neural=neural)
    write(OUT/'selection_tables.json',tables);write(OUT/'chosen_panels.json',choices)
    seal(create=True);lock();emit('ALL CANDIDATE SCORES AND CHOICES SEALED;current audit arrays still unopened')


def choice_files():
    paths=[OUT/'pools.json',OUT/'selection_tables.json',OUT/'chosen_panels.json']
    for split,seed in cells():
        for pid in (*PIDS,'COV','FULL'):
            for target in range(3):
                f=choice_folder(split,seed,pid,target);paths += [f/name for name in ('selected.json','selected_model.pt','history.json')]
        for pid in PIDS:
            paths += [OUT/'teacher'/str(split)/str(seed)/pid/name for name in ('model.pt','score.json')]
    return paths


def seal(create=False):
    values={str(p):sha(p) for p in choice_files()};path=OUT/'choices_locked.json'
    if path.exists():
        if read(path)['files']!=values:raise RuntimeError('Sealed choices changed')
    elif create:write(path,dict(utc=datetime.now(timezone.utc).isoformat(),files=values))
    else:raise RuntimeError('Global choices not sealed')


def load_audit(split):
    if not (OUT/'choices_locked.json').exists():raise RuntimeError('Audit blocked before global selection seal')
    seal();lock()
    with np.load(previous.prior.OLD/str(split)/'audit_data.npz',allow_pickle=False) as z:
        return tuple(z[k].copy() for k in ('X_audit','y_audit','audit_ids'))


def refit(split,seed,pid,target,x,y,cache):
    key=(split,seed,pid,target)
    if key in cache:return cache[key]
    a=choice(*key);f=OUT/'refits'/str(split)/str(seed)/pid/str(target)
    f.mkdir(parents=True);reused=False
    if pid in BASE_PIDS:
        old=original_refit_folder(*key);c=read(old/'selected.json')
        reused=c['epoch']==a['epoch']
    if reused:
        model_path=old/'refit_model.pt';meta=read(old/'refit.json');seconds=meta['seconds'];updates=meta['updates']
    else:
        subset=a['subset'];trained=base.train(x,y[:,target:target+1],None,None,subset,seed,WIDTHS,fixed_epochs=a['epoch'])
        model_path=f/'model.pt';torch_atomic(model_path,dict(state=trained['model'].state_dict(),norm=trained['norm'],subset=subset))
        seconds=trained['seconds'];updates=trained['updates']
    entry=dict(split=split,seed=seed,panel=pid,target=target,epoch=a['epoch'],subset=a['subset'],reused=reused,
        model_path=str(model_path),updates=updates,seconds=seconds,selected_folder=a['folder'])
    write(f/'entry.json',entry);cache[key]=entry
    return entry


def predict_saved(path,x):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=base.ControlMLP(len(saved['subset']),1,WIDTHS);model.load_state_dict(saved['state']);model.eval()
    return nn_predict(model,saved['norm'],x,saved['subset'])


def summarize(rows):
    expected={(s,seed,m) for s,seed in cells() for m in METHODS}
    if len(rows)!=len(expected) or {(r['split'],r['seed'],r['method']) for r in rows}!=expected:raise ValueError('Incomplete unique method grid')
    methods={}
    for m in METHODS:
        group=[r for r in rows if r['method']==m]
        methods[m]=dict(mean_r2=float(np.mean([r['audit']['mean_r2'] for r in group])),
            r2=np.mean([r['audit']['r2'] for r in group],axis=0).tolist(),rmse=np.mean([r['audit']['rmse'] for r in group],axis=0).tolist(),
            partition_means={str(s):float(np.mean([r['audit']['mean_r2'] for r in group if r['split']==s])) for s in SPLITS})
    comparisons={m:strict_contrast(rows,'MLP',m,SPLITS,SEEDS) for m in METHODS if m!='MLP'}
    return dict(methods=methods,MLP_minus_reference=comparisons,
        primary_support=all(comparisons[m]['descriptive_support'] for m in ('PLS','original','covsel')),
        configuration_promoted=False,independent_validation=False,strict_equal_compute_claimed=False)


def evaluate_all(pools):
    seal();lock();choices=read(OUT/'chosen_panels.json');cache={};rows=[]
    write(OUT/'status.json',dict(state='refitting_and_evaluating',audit_opened=True,new_models_trained=len(new_model_cases())))
    for split in SPLITS:
        xf,yf,xr,yr=duration.selection_arrays(split);x,y=np.vstack([xf,xr]),np.vstack([yf,yr]);xa,ya,ids=load_audit(split)
        with np.load(previous.prior.OLD/str(split)/'selection_data.npz',allow_pickle=False) as z:
            previous.prior.pilot.prior.assert_partition(z['fit_ids'],z['reward_ids'],ids)
        for seed in SEEDS:
            for method in METHODS:
                pid=choices[str(split)][str(seed)][method];entries=[refit(split,seed,pid,t,x,y,cache) for t in range(3)]
                prediction=np.hstack([predict_saved(e['model_path'],xa) for e in entries])
                score=metrics(ya,prediction);score['mae']=abs(ya-prediction).mean(0).tolist()
                f=OUT/'predictions'/str(split)/str(seed)/(method+'.npz');f.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(f,prediction=prediction,target=ya,ids=ids)
                row=dict(split=split,seed=seed,method=method,panel=pid,subset=subset_for(pools,split,seed,pid),audit=score,
                    prediction_file=str(f),refit_entries=[str(OUT/'refits'/str(split)/str(seed)/pid/str(t)/'entry.json') for t in range(3)])
                write(OUT/'cases'/str(split)/str(seed)/(method+'.json'),row);rows.append(row)
                emit(f'CASE split={split} seed={seed} method={method} panel={pid} audit_meanR2={score["mean_r2"]:.7f}')
    write(OUT/'metrics.json',rows);write(OUT/'summary.json',summarize(rows))
    write(OUT/'costs.json',dict(new_teacher_pls_fits=len(cells())*4,new_nn_selections=len(new_model_cases()),
        cached_nn_selections=len(cells())*9,new_unique_refits=sum(not e['reused'] for e in cache.values()),
        reused_unique_refits=sum(e['reused'] for e in cache.values()),unique_refits=len(cache),
        strictly_equal_compute=False,scope='Shared scientific evaluation;operational logical costs differ and are predeclared'))
    seal();lock();write(OUT/'complete.json',dict(files={n:sha(OUT/n) for n in ('protocol.json','metrics.json','summary.json','costs.json','choices_locked.json')}))


def verify():
    from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
    lock();seal()
    for name,digest in read(OUT/'complete.json')['files'].items():
        if sha(OUT/name)!=digest:raise AssertionError('Completed output changed')
    pools=read(OUT/'pools.json');chosen=read(OUT/'chosen_panels.json');tables=read(OUT/'selection_tables.json')
    rebuilt,_=build_pools()
    if pools!=rebuilt:raise AssertionError('Candidate provenance changed')
    rows=read(OUT/'metrics.json');max_error=0.;selected_count=0;refit_entries={}
    for split in SPLITS:
        xf,yf,xr,yr=duration.selection_arrays(split);x,y=np.vstack([xf,xr]),np.vstack([yf,yr]);xa,ya,ids=load_audit(split)
        for seed in SEEDS:
            for pid in (*PIDS,'COV','FULL'):
                subset=subset_for(pools,split,seed,pid)
                for target in range(3):
                    a=choice(split,seed,pid,target);f=Path(a['folder']);h=read(f/'history.json');cfg=profile(target)
                    if a['subset']!=subset or a['widths']!=list(WIDTHS):raise AssertionError('Predictor/input mismatch')
                    if not cfg['min_epochs']<=a['epochs_run']<=cfg['max_epochs'] or a['updates']!=math.ceil(len(xf)/128)*a['epochs_run']:raise AssertionError('Training duration/updates')
                    if a['epoch']!=max(h,key=lambda z:z['mean_r2'])['epoch'] or [v['epoch'] for v in h]!=list(range(10,a['epochs_run']+1,10)):raise AssertionError('Checkpoint/history mismatch')
                    anchor,last,best=-float('inf'),0,-float('inf')
                    for v in h:
                        best=max(best,v['mean_r2']);anchor,last=advance_patience(best,anchor,last,v['epoch'])
                    if a['epochs_run']!=cfg['max_epochs'] and a['epochs_run']-last<cfg['patience']:raise AssertionError('Patience mismatch')
                    saved=torch.load(f/'selected_model.pt',map_location='cpu',weights_only=False)
                    expected=dict(xm=xf[:,subset].mean(0),xs=xf[:,subset].std(0),ym=yf[:,target:target+1].mean(0),ys=yf[:,target:target+1].std(0))
                    if saved['subset']!=subset or any(not np.array_equal(saved['norm'][k],v) for k,v in expected.items()):raise AssertionError('Training-only scales')
                    score=r2_score(yr[:,target:target+1],predict_saved(f/'selected_model.pt',xr))
                    if abs(score-a['selection']['mean_r2'])>1e-13:raise AssertionError('Reward model reload mismatch')
                    selected_count+=1
                if pid in PIDS:
                    folder=OUT/'teacher'/str(split)/str(seed)/pid;s=read(folder/'score.json')
                    model=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
                    if model['subset']!=subset or model['components']!=8 or s['fits']!=1:raise AssertionError('Teacher definition')
                    if not np.array_equal(model['target_mean'],yf.mean(0)) or not np.array_equal(model['target_std'],yf.std(0)):raise AssertionError('Teacher fit-only scales')
                    actual=metrics(yr,pls_predict(model,xr))
                    if actual!=s['selection']:raise AssertionError('Teacher reward reload')
            expected_choice=choose_panel(PIDS,tables[str(split)][str(seed)]['teacher'],tables[str(split)][str(seed)]['neural'])
            for pid in PIDS:
                if tables[str(split)][str(seed)]['teacher'][pid]!=read(OUT/'teacher'/str(split)/str(seed)/pid/'score.json')['selection']['mean_r2']:raise AssertionError('Teacher table mismatch')
                if tables[str(split)][str(seed)]['neural'][pid]!=[choice(split,seed,pid,t)['selection']['mean_r2'] for t in range(3)]:raise AssertionError('Neural table mismatch')
            expected_choice.update(original='U3',covsel='COV',full='FULL')
            if chosen[str(split)][str(seed)]!=expected_choice:raise AssertionError('Ranking/tie rule mismatch')
            for method in METHODS:
                row=next(r for r in rows if (r['split'],r['seed'],r['method'])==(split,seed,method));pid=expected_choice[method]
                subset=subset_for(pools,split,seed,pid)
                if row['panel']!=pid or row['subset']!=subset:raise AssertionError('Case panel mismatch')
                outputs=[]
                for target,entry_path in enumerate(row['refit_entries']):
                    e=read(entry_path);a=choice(split,seed,pid,target)
                    if (e['split'],e['seed'],e['panel'],e['target'])!=(split,seed,pid,target) or e['epoch']!=a['epoch'] or e['updates']!=math.ceil(len(x)/128)*a['epoch']:raise AssertionError('Refit epoch/identity/updates')
                    if e['reused']:
                        old=original_refit_folder(split,seed,pid,target)
                        if Path(e['model_path'])!=old/'refit_model.pt' or read(old/'selected.json')['epoch']!=a['epoch'] or read(old/'refit.json')['epoch']!=a['epoch']:raise AssertionError('Invalid cached refit')
                    saved=torch.load(e['model_path'],map_location='cpu',weights_only=False)
                    expected=dict(xm=x[:,subset].mean(0),xs=x[:,subset].std(0),ym=y[:,target:target+1].mean(0),ys=y[:,target:target+1].std(0))
                    if saved['subset']!=subset or any(not np.array_equal(saved['norm'][k],v) for k,v in expected.items()):raise AssertionError('Fixed-epoch refit scales')
                    outputs.append(predict_saved(e['model_path'],xa));refit_entries[entry_path]=e
                prediction=np.hstack(outputs)
                with np.load(row['prediction_file'],allow_pickle=False) as z:
                    if not np.array_equal(z['target'],ya) or not np.array_equal(z['ids'],ids):raise AssertionError('Audit IDs/targets')
                    error=float(np.max(abs(prediction-z['prediction'])));max_error=max(max_error,error)
                    if error!=0:raise AssertionError('Refit prediction reload')
                values=[r2_score(ya,prediction,multioutput='raw_values'),np.sqrt(mean_squared_error(ya,prediction,multioutput='raw_values')),mean_absolute_error(ya,prediction,multioutput='raw_values')]
                for key,value in zip(('r2','rmse','mae'),values):
                    if not np.allclose(value,row['audit'][key],atol=1e-12,rtol=0):raise AssertionError('Independent metric mismatch')
    if summarize(rows)!=read(OUT/'summary.json'):raise AssertionError('Summary mismatch')
    costs=read(OUT/'costs.json')
    if costs['new_unique_refits']!=sum(not e['reused'] for e in refit_entries.values()) or costs['reused_unique_refits']!=sum(e['reused'] for e in refit_entries.values()):raise AssertionError('Unique refit accounting')
    if costs['new_unique_refits']>read(OUT/'protocol.json')['max_new_refits']:raise AssertionError('Prespecified refit bound exceeded')
    lock();seal()
    proof=dict(passed=True,pipeline_cases=len(rows),selected_models=selected_count,new_selection_models=len(new_model_cases()),
        teacher_panel_fits=len(cells())*4,max_prediction_reload_error=max_error,unique_refits=len(refit_entries),
        independent_validation=False,configuration_promoted=False,strict_equal_compute_claimed=False)
    write(OUT/'verification.json',proof);return proof


def report_results(proof):
    s=read(OUT/'summary.json');cost=read(OUT/'costs.json')
    labels={'original':'原4096端点并集','PLS':'同池PLS8重排','MLP':'同池MLP重排','covsel':'CovSel48','full':'全谱641'}
    lines=['# 四面板最终预测器重排：结果','',
        '同一固定四候选池，比较PLS8与最终MLP排序；所有选择封存后统一重拟合与历史开发复核。'
        '这不是新来源独立验证。多候选奖励最大值提高不是成功证据，必须看下列复核差值。','',
        '|选择方案|平均R²|SSC|pH|硬度|','|---|---:|---:|---:|---:|']
    for method,a in s['methods'].items():lines.append('|'+labels[method]+'|'+'|'.join(f'{x:.6f}' for x in [a['mean_r2'],*a['r2']])+'|')
    lines+=['','|MLP重排减参照|平均差|SSC差|pH差|硬度差|配对胜出|预定描述性条件|','|---|---:|---:|---:|---:|---:|---|']
    for ref,a in s['MLP_minus_reference'].items():
        lines.append('|'+labels[ref]+'|'+'|'.join(f'{x:+.6f}' for x in [a['mean_delta'],*a['target_delta']])+f'|{a["wins"]}/{len(cells())}|'+('满足' if a['descriptive_support'] else '未满足')+'|')
    lines+=['',
        '描述性条件：平均差>0、每个划分平均差>0、至少2／3配对单元胜出、各目标总体均值差≥0。主比较为同池PLS8、原并集及CovSel。'
        '当前历史数据有适应与重叠，不能作独立显著性推断或根据复核结果重新加候选。','',
        f'新增PLS面板评分{cost["new_teacher_pls_fits"]}次；新增候选NN选择{cost["new_nn_selections"]}次，复用选择{cost["cached_nn_selections"]}个。'
        f'实际去重后新重拟合{cost["new_unique_refits"]}次，复用重拟合{cost["reused_unique_refits"]}个。'
        '历史选谱支付量、共享核查成本和单条方法的实际运行成本分开解释。MLP重排每单元需要12个选择模型，PLS重排运行时只需要最终面板的3个；不能宣称与CovSel严格同计算成本。','',
        f'核验{proof["selected_models"]}个选择模型及全部{proof["pipeline_cases"]}流程；预测重载误差{proof["max_prediction_reload_error"]:.1g}。'
        'SSC／pH最少2400最多4800耐心1200，硬度沿用最少1200最多2400耐心600；同目标各方法规则一致，触顶不证明充分收敛。',
        '无论结果如何均保留所有单元，本轮不继续扩大候选池或调参，不自动晋级主方法或启动独立来源实验。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run():
    lock()
    if read(OUT/'status.json')['state']!='prepared_not_started':raise RuntimeError('Already started;do not automatically retry')
    with (OUT/'operator_start.txt').open('x',encoding='utf-8') as f:f.write(datetime.now(timezone.utc).isoformat()+'\n')
    try:
        signal.signal(signal.SIGINT,lambda *_:setattr(base,'STOP',True))
        torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        with warnings.catch_warnings(),threadpool_limits(limits=1):
            warnings.simplefilter('error');runtime=previous.version.pin_and_record()
            if runtime['actual_interop_threads']!=1:raise RuntimeError('Interop1 required')
            runtime['python_executable']=sys.executable;write(OUT/'environment.json',runtime)
            write(OUT/'status.json',dict(state='selecting',audit_opened=False,new_models_trained=0))
            pools=read(OUT/'pools.json');select_all(pools);previous.version.pin_and_record()
            evaluate_all(pools);previous.version.pin_and_record();proof=verify();report_results(proof)
        write(OUT/'status.json',dict(state='complete_verified',audit_opened=True,new_models_trained=len(new_model_cases())))
        emit('COMPLETE fixed-pool reranking verified;no new-source experiment,extra candidates or parameter changes started')
        files=[f for f in OUT.rglob('*') if f.is_file() and f.name!='delivery_manifest.json']
        write(OUT/'delivery_manifest.json',dict(files={str(f):sha(f) for f in files}))
    except BaseException as error:
        previous_state=read(OUT/'status.json')
        write(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        write(OUT/'status.json',dict(state='stopped_on_problem',audit_opened=previous_state.get('audit_opened',False)))
        emit('STOPPED;retain files and diagnose;no automatic retry,model switch or changed parameters');raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run','status']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='run':run()
    else:print(read(OUT/'status.json'))


if __name__=='__main__':main()
