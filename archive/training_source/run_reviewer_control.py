"""Frozen D dual-range and Corn split controls. Real fits require CLI `run`."""
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager
import argparse
import signal
import sys
import time
import traceback
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import reviewer_control_core as c
from empirical_budget_core import GenericPLSEvaluator
from hsi_e2e.pool_boundary import CapacitySearch
from hsi_e2e.framework_screen import pls_fit,pls_predict
from hsi_e2e.formal_runtime import torch_atomic

STOP=False
def interrupt(signum,frame):
    global STOP
    STOP=True
def stop_check():
    if STOP: raise InterruptedError('Operator stop;no automatic rerun')
def configure_runtime():
    torch.set_num_threads(4)
    if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
def versions():
    import platform,sklearn
    return dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,sklearn=sklearn.__version__)


class Runner:
    def __init__(self,out=c.OUT):
        self.out=Path(out); self.protocol=c.read(self.out/'protocol.json'); self.spec=self.protocol['spec']
        self.cells=c.read(self.out/'cells.json'); self.sealed=False
        self.cost=dict(search_candidate_fits=0,calibration_candidate_fits=0,final_refits=0,
                       final_models_reused_within_panel=0,historical_ABC_target_records_reused=0)
        self.context='initialization'; self.data_cache={}

    def lock(self):
        c.check_hashes(c.read(self.out/'prepared_manifest.json')['files'])
        c.check_hashes(self.protocol['source_sha256']); c.check_hashes(self.protocol['protected_input_hashes'])
        if versions()!=self.protocol['versions']: raise RuntimeError('Pinned scientific versions changed')
        if not c.read(self.protocol['readiness_verification'])['passed']: raise RuntimeError('Readiness not passed')

    def emit(self,message):
        from datetime import datetime,timezone
        line=datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+message
        print(line,flush=True)
        with (self.out/'run.log').open('a',encoding='utf-8') as f:f.write(line+'\n')

    def selection(self,cell):
        key=cell['id']
        if key not in self.data_cache:
            with np.load(cell['selection_file'],allow_pickle=False) as z:
                a=tuple(z[k].copy() for k in ('X_fit','y_fit','X_reward','y_reward','fit_ids','reward_ids'))
            xf,yf,xv,yv,fi,vi=a
            if set(fi)&set(vi) or (set(fi)|set(vi))&set(cell['test_ids']): raise ValueError('Training/test ID overlap')
            if sorted(fi.tolist())!=sorted(cell['fit_ids']) or sorted(vi.tolist())!=sorted(cell['validation_ids']): raise ValueError('Selection sample identity changed')
            if any(not np.isfinite(v).all() for v in a[:4]) or xf.shape[1]!=cell['bands'] or yf.shape[1]!=len(cell['targets']): raise ValueError('Selection shape/nonfinite')
            self.data_cache[key]=a
        return self.data_cache[key]

    def model_check(self,m,x,y,subset,components):
        if m['subset']!=subset or m['components']!=components: raise AssertionError('Model/subset/component mismatch')
        for k,expected in [('xmean',x[:,subset].mean(0)),('xstd',x[:,subset].std(0,ddof=1)),('target_mean',y.mean(0)),('target_std',y.std(0))]:
            if not np.allclose(m[k],expected,atol=1e-13,rtol=1e-12): raise AssertionError('Training-only scale mismatch')

    def build_panels(self):
        panels={}; draws=[]
        for cell in self.cells:
            stop_check(); self.context='panels/'+cell['id']
            if cell['mode']=='original': own=cell['own']; union=cell['union']
            else:
                xf,yf,xv,yv,*_=self.selection(cell); own=[]
                for t in range(len(cell['targets'])):
                    stop_check(); folder=self.out/'search'/cell['id']/str(t); self.context='search/'+cell['id']+'/t'+str(t)
                    ev=GenericPLSEvaluator(xf,yf[:,t:t+1],xv,yv[:,t:t+1],self.spec['search_budget'],k=cell['quota'],components=self.spec['search_components'])
                    state=CapacitySearch(ev,cell['selector_seed'],'greedy'); start=time.perf_counter()
                    try:
                        state.initialize()
                        while ev.fit_count<self.spec['search_budget']: stop_check(); state.one_batch()
                        if ev.fit_count!=self.spec['search_budget'] or ev.cache_hits: raise AssertionError('Search paid-query budget changed')
                        own.append(list(ev.best_subset))
                    finally:
                        state.elapsed_seconds=time.perf_counter()-start
                        self.cost['search_candidate_fits']+=ev.fit_count
                        c.write(folder/'ledger.json',ev.ledger); c.write(folder/'decisions.json',state.decisions)
                        c.write(folder/'boundary_events.json',getattr(state,'boundary_events',[]))
                        c.write(folder/'summary.json',state.summary() if ev.best_subset is not None else dict(incomplete=True,actual_fits=ev.fit_count))
                    self.emit(f'SEARCH COMPLETE {cell["id"]} target={t};paid={ev.fit_count};no test scoring')
                union=sorted(set().union(*map(set,own)))
            for t,s in enumerate(own):
                c.validate_panel(s,union,cell['bands'],cell['quota'])
                for rep in range(1,self.spec['random_replicates']+1):
                    panel,entropy=c.random_panel(cell,s,union,t,rep,self.spec)
                    draws.append(dict(cell_id=cell['id'],target=t,replicate=rep,subset=panel,
                        own=s,union=union,rng_entropy=entropy,other_target_columns_in_draw=len((set(panel)-set(s))&set(union)),
                        per_model_input_count=len(panel),B_input_count=len(union)))
            panels[cell['id']]=dict(own=own,union=union)
        initial=self.out/'initial_random_panels.json'
        if initial.exists():
            original_ids={z['id'] for z in self.cells if z['mode']=='original'}
            if [d for d in draws if d['cell_id'] in original_ids]!=c.read(initial):
                raise AssertionError('Original random panels changed after preparation')
        c.write(self.out/'panels.json',panels); c.write(self.out/'random_panels.json',draws)
        return panels,draws

    def fit_entry(self,cell,kind,target,subset,modes,replicate=None):
        stop_check(); ident=c.entry_id(cell,kind,target,replicate); self.context='calibration/'+ident
        f=self.out/'models'/ident; f.mkdir(parents=True,exist_ok=False)
        xf,yf,xv,yv,*_=self.selection(cell)
        high=max(modes.values()); trials=[]
        for component in range(1,high+1):
            stop_check(); m=pls_fit(xf,yf[:,target:target+1],subset,component)
            self.cost['calibration_candidate_fits']+=1
            score=c.metrics(yv[:,target:target+1],pls_predict(m,xv))
            trials.append(dict(components=component,**score)); torch_atomic(f/f'trial_{component}.pt',m)
        c.write(f/'trials.json',trials)
        x,y=np.vstack([xf,xv]),np.vstack([yf,yv]); final_by_component={}; choices={}
        for arm,cap in modes.items():
            selected=c.choose(trials,cap); component=selected['components']
            if component not in final_by_component:
                stop_check(); m=pls_fit(x,y[:,target:target+1],subset,component); self.cost['final_refits']+=1
                path=f/f'refit_{component}.pt'; torch_atomic(path,m); final_by_component[component]=str(path)
            else:self.cost['final_models_reused_within_panel']+=1
            choices[arm]=dict(components=component,allowed_cap=cap,selection=selected,
                selected_model=str(f/f'trial_{component}.pt'),final_model=final_by_component[component])
        record=dict(id=ident,cell_id=cell['id'],target=target,kind=kind,replicate=replicate,subset=subset,
                    choices=choices,candidate_fits=high,trials_file=str(f/'trials.json'))
        path=f/'selected.json'; c.write(path,record)
        return str(path)

    def calibrate(self,panels,draws):
        records=[]; lookup=defaultdict(list)
        for d in draws: lookup[d['cell_id']].append(d)
        for index,cell in enumerate(self.cells,1):
            q=len(cell['targets']); panel=panels[cell['id']]; xf,*_=self.selection(cell)
            wide=min(self.spec['wide_cap'],len(panel['union']),len(xf)-1)
            if wide<cell['quota']: raise ValueError('Feasible broad grid below common grid')
            if cell['mode']=='new':
                for t in range(q):
                    records.append(self.fit_entry(cell,'own',t,panel['own'][t],{'A':cell['quota']}))
                    records.append(self.fit_entry(cell,'union',t,panel['union'],{'B':cell['quota'],'C':wide}))
            for d in sorted(lookup[cell['id']],key=lambda v:(v['replicate'],v['target'])):
                records.append(self.fit_entry(cell,'random',d['target'],d['subset'],
                    {'D_low':cell['quota'],'D_wide':wide},d['replicate']))
            c.write(self.out/'model_records.json',records); c.write(self.out/'costs.json',self.cost)
            self.emit(f'CALIBRATION {index}/{len(self.cells)} cells;trials={self.cost["calibration_candidate_fits"]};refits={self.cost["final_refits"]};no test scoring')
        return records

    def seal(self,create=False):
        path=self.out/'choices_locked.json'
        if create:
            records=c.read(self.out/'model_records.json'); expect=sum(len(z['targets'])*(self.spec['random_replicates']+(2 if z['mode']=='new' else 0)) for z in self.cells)
            if len(records)!=expect: raise RuntimeError('Incomplete global model choices')
            files=[self.out/n for n in ('panels.json','random_panels.json','model_records.json','protocol.json','cells.json','costs.json')]
            files+=sorted(p for p in (self.out/'models').rglob('*') if p.is_file())
            files+=sorted(p for p in (self.out/'search').rglob('*') if p.is_file()) if (self.out/'search').exists() else []
            if path.exists(): raise RuntimeError('Choice seal already exists')
            c.write(path,dict(files={str(p):c.sha(p) for p in files},scope='All new choices and final models sealed before new test scoring'))
        if not path.exists(): raise RuntimeError('Global choices not sealed;test access blocked')
        c.check_hashes(c.read(path)['files']); self.sealed=True

    def test(self,cell):
        if not self.sealed: self.seal()
        if cell.get('test_labels_from_prediction'):
            with np.load(cell['test_file'],allow_pickle=False) as z: x,ids=z['X_test'].copy(),z['test_ids'].copy()
            with np.load(cell['test_labels_from_prediction'],allow_pickle=False) as z:
                other,y=z['ids'].copy(),z['target'].copy()
            if not np.array_equal(ids,other): raise AssertionError('Archived CGL labels not aligned to features')
        else:
            with np.load(cell['test_file'],allow_pickle=False) as z: x,y,ids=[z[k].copy() for k in ('X_test','y_test','test_ids')]
        if ids.tolist()!=cell['test_ids'] or len(ids)!=len(set(ids.tolist())): raise ValueError('Heldout identity changed')
        if x.shape!=(len(ids),cell['bands']) or y.shape!=(len(ids),len(cell['targets'])) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Heldout shape/nonfinite value')
        *_,fi,vi=self.selection(cell)
        if (set(fi)|set(vi))&set(ids): raise ValueError('Test and selection samples overlap')
        return x,y,ids

    def evaluate(self):
        self.seal(); all_records=[c.read(p)|dict(record_file=p) for p in c.read(self.out/'model_records.json')]
        by_cell=defaultdict(list)
        for r in all_records: by_cell[r['cell_id']].append(r)
        rows=[]
        for index,cell in enumerate(self.cells,1):
            stop_check(); self.context='test/'+cell['id']; xt,yt,ids=self.test(cell)
            xf,yf,xv,yv,*_=self.selection(cell); x,y=np.vstack([xf,xv]),np.vstack([yf,yv]); sd=y.std(0,ddof=1)
            groups=defaultdict(dict)
            for r in by_cell[cell['id']]:
                for arm,ch in r['choices'].items(): groups[(arm,r['replicate'])][r['target']]=(r,ch)
            if cell['mode']=='original':
                for arm,a in cell['archived'].items():
                    with np.load(a['prediction_file'],allow_pickle=False) as z: ai,ay,ap=[z[k].copy() for k in ('ids','target','prediction')]
                    if not np.array_equal(ai,ids) or not np.array_equal(ay,yt): raise AssertionError('Archived ABC evaluation population differs')
                    refs=[c.read(p) for p in a['refit_entries']]
                    panels=[r['subset'] for r in refs]; self.cost['historical_ABC_target_records_reused']+=len(refs)
                    rows.append(self.row(cell,arm,None,a['prediction_file'],sd,panels,[r['components'] for r in refs],a['refit_entries'],yt,ap))
            for (arm,rep),items in sorted(groups.items()):
                if sorted(items)!=list(range(len(cell['targets']))): raise AssertionError('Missing target model')
                pred=[]; panels=[]; components=[]; refs=[]
                for t,(r,ch) in sorted(items.items()):
                    m=torch.load(ch['final_model'],map_location='cpu',weights_only=False)
                    self.model_check(m,x,y[:,t:t+1],r['subset'],ch['components'])
                    pred.append(pls_predict(m,xt)); panels.append(r['subset']); components.append(ch['components']); refs.append(r['record_file'])
                prediction=np.hstack(pred); path=self.out/'predictions'/cell['id']/(arm+('' if rep is None else f'_{rep:02d}')+'.npz')
                path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,ids=ids,target=yt,prediction=prediction)
                rows.append(self.row(cell,arm,rep,str(path),sd,panels,components,refs,yt,prediction))
            self.emit(f'EVALUATION {index}/{len(self.cells)} cells complete;results reviewed only after full run')
        c.write(self.out/'metrics.json',rows); c.write(self.out/'costs_final.json',self.cost)
        summary,raw=c.summarize(rows,self.cells,self.spec); c.write(self.out/'summary.json',summary); c.write(self.out/'pooled_metrics.json',raw)
        return summary

    def row(self,cell,arm,rep,path,sd,panels,components,refs,y,p):
        return dict(source=cell['source'],cohort=cell['cohort'],fold=cell['fold'],cell_id=cell['id'],selector_seed=cell['selector_seed'],
            arm=arm,replicate=rep,prediction_file=path,outer_training_sd=sd.tolist(),
            input_counts=list(map(len,panels)),global_input_union_count=len(set().union(*map(set,panels))),
            components=components,model_records=refs,score=c.metrics(y,p))

    def verify(self):
        self.lock(); self.seal(); rows=c.read(self.out/'metrics.json'); lookup={z['id']:z for z in self.cells}
        expected={(z['id'],a,rep) for z in self.cells for a in ('A','B','C','D_low','D_wide')
                  for rep in ([None] if a in ('A','B','C') else range(1,self.spec['random_replicates']+1))}
        identities=[(z['cell_id'],z['arm'],z['replicate']) for z in rows]
        if len(identities)!=len(set(identities)) or set(identities)!=expected: raise AssertionError('Prediction case coverage differs')
        panels=c.read(self.out/'panels.json'); draws=c.read(self.out/'random_panels.json')
        for d in draws:
            cell=lookup[d['cell_id']]; p=panels[cell['id']]
            actual,entropy=c.random_panel(cell,p['own'][d['target']],p['union'],d['target'],d['replicate'],self.spec)
            if actual!=d['subset'] or entropy!=d['rng_entropy']: raise AssertionError('Random-panel provenance differs')
        reload_error=0.; checked=0; trial_count=0; final_paths=set(); logical_choices=0
        for path in c.read(self.out/'model_records.json'):
            r=c.read(path); cell=lookup[r['cell_id']]; xf,yf,xv,yv,*_=self.selection(cell); t=r['target']
            trials=c.read(r['trials_file']); trial_count+=len(trials)
            for v in trials:
                m=torch.load(Path(path).parent/f'trial_{v["components"]}.pt',map_location='cpu',weights_only=False)
                self.model_check(m,xf,yf[:,t:t+1],r['subset'],v['components'])
                value=c.metrics(yv[:,t:t+1],pls_predict(m,xv))
                if abs(value['mean_r2']-v['mean_r2'])>1e-12: raise AssertionError('Trial score replay differs')
            for arm,ch in r['choices'].items():
                if c.choose(trials,ch['allowed_cap'])['components']!=ch['components']: raise AssertionError('Wrong bounded choice/tie rule')
                logical_choices+=1;final_paths.add(ch['final_model'])
            checked+=1
        tests={}
        for row in rows:
            cell=lookup[row['cell_id']]
            if cell['id'] not in tests:tests[cell['id']]=self.test(cell)
            xt,yt,ids=tests[cell['id']]
            with np.load(row['prediction_file'],allow_pickle=False) as z:
                saved=z['prediction'].copy()
                if not np.array_equal(z['ids'],ids) or not np.array_equal(z['target'],yt): raise AssertionError('Saved prediction ID/target mismatch')
            xf,yf,xv,yv,*_=self.selection(cell); x,y=np.vstack([xf,xv]),np.vstack([yf,yv]);sd=y.std(0,ddof=1)
            if row['outer_training_sd']!=sd.tolist(): raise AssertionError('Normalized-error training population differs')
            pred=[];subsets=[];components=[]
            for t,path in enumerate(row['model_records']):
                ref=c.read(path)
                if row['arm'] in ref.get('choices',{}): modelpath=ref['choices'][row['arm']]['final_model']
                else:modelpath=ref['model_path']
                m=torch.load(modelpath,map_location='cpu',weights_only=False)
                chosen=ref['choices'][row['arm']]['components'] if row['arm'] in ref.get('choices',{}) else ref['components']
                self.model_check(m,x,y[:,t:t+1],ref['subset'],chosen)
                pred.append(pls_predict(m,xt));subsets.append(ref['subset']);components.append(chosen)
            if list(map(len,subsets))!=row['input_counts'] or len(set().union(*map(set,subsets)))!=row['global_input_union_count'] or components!=row['components']:
                raise AssertionError('Reported input/component count differs')
            if row['arm'] in ('D_low','D_wide') and (any(len(s)!=len(panels[cell['id']]['union']) for s in subsets) or
                any(not set(own)<=set(s) for own,s in zip(panels[cell['id']]['own'],subsets))): raise AssertionError('Random panel dimension/retention differs')
            err=float(np.max(abs(np.hstack(pred)-saved))); reload_error=max(reload_error,err)
            if err>1e-12: raise AssertionError('Final prediction reload mismatch')
            from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
            actual=dict(r2=r2_score(yt,saved,multioutput='raw_values'),rmse=np.sqrt(mean_squared_error(yt,saved,multioutput='raw_values')),
                        mae=mean_absolute_error(yt,saved,multioutput='raw_values'))
            for key,value in actual.items():
                if not np.allclose(value,row['score'][key],rtol=0,atol=1e-12): raise AssertionError('Independent metric mismatch')
        summary,raw=c.summarize(rows,self.cells,self.spec)
        if summary!=c.read(self.out/'summary.json') or raw!=c.read(self.out/'pooled_metrics.json'): raise AssertionError('Pooled summaries changed')
        cost=c.read(self.out/'costs_final.json')
        if trial_count!=cost['calibration_candidate_fits']: raise AssertionError('Candidate fit cost mismatch')
        if len(final_paths)!=cost['final_refits'] or logical_choices-len(final_paths)!=cost['final_models_reused_within_panel']:
            raise AssertionError('Final fit/reuse cost mismatch')
        historical=sum(len(z['targets'])*3 for z in self.cells if z['mode']=='original')
        if historical!=cost['historical_ABC_target_records_reused']: raise AssertionError('Historical ABC reuse count differs')
        expected=sum(self.spec['search_budget']*len(z['targets']) for z in self.cells if z['mode']=='new')
        if cost['search_candidate_fits']!=expected: raise AssertionError('Search cost mismatch')
        for cell in self.cells:
            if cell['mode']=='new':
                for t in range(len(cell['targets'])):
                    folder=self.out/'search'/cell['id']/str(t); history=c.read(folder/'ledger.json'); state=c.read(folder/'summary.json')
                    if len(history)!=self.spec['search_budget'] or state['actual_fits']!=len(history): raise AssertionError('Search ledger incomplete')
                    if len({tuple(z['subset']) for z in history})!=len(history): raise AssertionError('Search repeated paid subset')
                    best=max(history,key=lambda z:z['mean_r2'])['subset']
                    if best!=panels[cell['id']]['own'][t]: raise AssertionError('Search incumbent mismatch')
        budget=self.protocol.get('budget')
        if budget and (len(rows)!=budget['expected_pipeline_cases'] or trial_count>budget['new_calibration_candidate_fits_upper'] or
            len(final_paths)>budget['final_refits_upper']): raise AssertionError('Frozen experiment budget exceeded')
        self.lock()
        proof=dict(passed=True,pipeline_cases=len(rows),logical_target_models=sum(len(z['targets']) for z in self.cells)*(3+2*self.spec['random_replicates']),
            stored_model_records=checked,candidate_records_replayed=trial_count,max_prediction_reload_error=reload_error,
            all_new_choices_sealed_before_test=True,D_per_model_counts_matched=True,D_global_budget_matched=False,
            reused_ABC_not_claimed_new=True,independent_validation_claimed=False,new_fits_during_verification=0,costs=cost)
        c.write(self.out/'verification.json',proof); return proof

    def report(self):
        summary=c.read(self.out/'summary.json'); lines=['# 审稿补强实验结果','',
            '这是预先封存规则后的同来源事后扩展，不是新的独立盲测。D仅匹配每模型列数，全球输入并集、相关结构及搜索成本不匹配。','',
            '|来源／划分|A|B|C|D窄均值|D宽均值|B−D窄|C−D宽|','|---|---:|---:|---:|---:|---:|---:|---:|']
        for key,v in summary['cohorts'].items():
            ms=v['methods']; lines.append('|'+key+'|'+'|'.join(f'{ms[a]["mean_r2"]:.6f}' for a in ('A','B','C','D_low','D_wide'))+
                f'|{v["contrasts"]["B_minus_D_low"]["mean_r2"]:+.6f}|{v["contrasts"]["C_minus_D_wide"]["mean_r2"]:+.6f}|')
        lines+=['','## 固定选谱种子17的Corn划分敏感性','',
            '|划分|A|B|C|D窄均值|D宽均值|B−A|C−B|','|---|---:|---:|---:|---:|---:|---:|---:|']
        for v in summary['split_sensitivity_fixed_seed']:
            lines.append('|'+v['cohort']+'|'+'|'.join(f'{v[a]:+.6f}' for a in ('A','B','C','D_low','D_wide','B_minus_A','C_minus_B'))+'|')
        lines+=['','D的10次随机化、算法种子与外层划分都不是独立材料来源；未把重复预测拼为240个独立样本。'
            '全部随机面板分布、逐目标R²／RMSE／MAE／训练SD归一化RMSE见summary.json与pooled_metrics.json。'
            '胜过随机面板次数仅作描述，不作p值。实验预算完成后停止，不按结果追加面板、划分或调参。']
        (self.out/'结果说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        self.figures(summary)

    def figures(self,summary):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,len(summary['cohorts']),figsize=(4*len(summary['cohorts']),4),squeeze=False)
        for ax,(key,v) in zip(axes[0],summary['cohorts'].items()):
            for x,(d,a,color) in enumerate([('D_low','B','#0072B2'),('D_wide','C','#D55E00')]):
                vals=[z['mean_r2'] for z in v['D_draws'][d]]
                ax.scatter(x+np.linspace(-.07,.07,len(vals)),vals,color=color,alpha=.65,s=25)
                ax.scatter(x,v['methods'][a]['mean_r2'],marker='D',s=70,color='black')
            ax.set_xticks([0,1],['D low / B','D wide / C']);ax.set_title(key,fontsize=9);ax.grid(axis='y',alpha=.2)
        axes[0,0].set_ylabel('Macro R2;seed metrics averaged');fig.tight_layout();fig.savefig(self.out/'D_reference_distribution.png',dpi=170);plt.close(fig)
        values=summary['split_sensitivity_fixed_seed'];fig,ax=plt.subplots(figsize=(7,4))
        for key,marker in [('B_minus_A','o'),('C_minus_B','s')]:ax.plot(range(len(values)),[v[key] for v in values],marker=marker,label=key)
        ax.axhline(0,color='gray',lw=.8);ax.set_xticks(range(len(values)),[v['cohort'] for v in values]);ax.set_ylabel('Macro R2 difference;selector seed17');ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(self.out/'Corn_split_sensitivity.png',dpi=170);plt.close(fig)

    def run(self):
        self.lock(); status=c.read(self.out/'status.json')
        if status['state']!='prepared_not_started': raise RuntimeError('Already started;review any problem before a versioned continuation')
        c.write(self.out/'status.json',dict(state='running_training',new_test_scoring=False)); start=time.perf_counter()
        try:
            configure_runtime()
            with threadpool_limits(limits=1):
                p,d=self.build_panels(); self.calibrate(p,d); self.seal(create=True)
                c.write(self.out/'status.json',dict(state='choices_sealed',new_test_scoring=False))
                self.emit('ALL NEW SUBSETS,CHOICES AND FINAL MODELS SEALED;starting test scoring')
                c.write(self.out/'status.json',dict(state='evaluating',new_test_scoring=True))
                self.evaluate(); proof=self.verify(); self.report()
            c.write(self.out/'status.json',dict(state='complete_verified',new_test_scoring=True,elapsed_seconds=time.perf_counter()-start))
            paths=sorted(p for p in self.out.rglob('*') if p.is_file() and p.name!='delivery_manifest.json')
            c.write(self.out/'delivery_manifest.json',dict(files={str(p):c.sha(p) for p in paths}))
            print('COMPLETE reviewer_control_v1;verified;no further experiment started',flush=True)
            print(__import__('json').dumps(proof,ensure_ascii=False),flush=True)
        except BaseException as exc:
            c.write(self.out/'problem.json',dict(context=self.context,error=repr(exc),traceback=traceback.format_exc(),costs=self.cost,
                new_test_scoring=c.read(self.out/'status.json').get('new_test_scoring',False)))
            exposed=c.read(self.out/'status.json').get('new_test_scoring',False)
            c.write(self.out/'status.json',dict(state='stopped_on_problem',new_test_scoring=exposed,
                reason='Stop and diagnose;do not blindly rerun or change rules using scores'))
            self.emit('STOPPED ON PROBLEM '+self.context+';'+repr(exc)); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=('run','verify')); args=parser.parse_args()
    signal.signal(signal.SIGINT,interrupt)
    if hasattr(signal,'SIGTERM'):signal.signal(signal.SIGTERM,interrupt)
    runner=Runner()
    if args.action=='run':runner.run()
    else:
        with threadpool_limits(limits=1):print(__import__('json').dumps(runner.verify(),ensure_ascii=False))
