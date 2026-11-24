"""New formal-study adapters; historical experiment implementations stay frozen."""
from pathlib import Path
import copy
import hashlib
import json
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from .variable_k import VariableTaskAttention,VariableEnsemble,bits


VARIANTS=('no_task_exchange','no_predictor_context','mean_mlp')


class AblatedAttention(VariableTaskAttention):
    def forward(self,query,tokens):
        if self.variant=='no_predictor_context':return super().forward(query,tokens)
        q=torch.tanh(self.query(query)+self.task_embedding)
        key=torch.tanh(self.key(tokens));value=torch.tanh(self.value(tokens))
        if self.variant=='mean_mlp':
            weights=torch.full(tokens.shape[:-1],1/(self.k-1),dtype=tokens.dtype,device=tokens.device)
            context=((key+value)*.5).mean(2)
        elif self.variant=='no_task_exchange':
            weights=((q[:,:,None,:]*key).sum(-1)/math.sqrt(self.dimension)).softmax(-1)
            context=(weights[:,:,:,None]*value).sum(2)
        else:raise ValueError('Unknown ablation')
        state=q+context
        # Replace the cross-task block by a per-task feed-forward transform.
        # All original projection parameters remain active, with identical count.
        hidden=torch.tanh(F.linear(state,self.task_relation.in_proj_weight,self.task_relation.in_proj_bias))
        exchange=self.task_relation.out_proj(hidden.reshape(*state.shape[:-1],3,self.dimension).mean(-2))
        relation=torch.eye(3,dtype=state.dtype,device=state.device).expand(len(query),3,3)
        return context,self.norm(state+exchange),weights,relation


def masked_context(q,t):
    q=q.copy();t=t.copy()
    q[...,[0,1,2,3,6,7]]=0
    t[...,[3,4,5]]=0
    return q,t


class AblationEnsemble(VariableEnsemble):
    def __init__(self,warm_groups,*,k,variant,seeds=(13,37,73)):
        if variant not in VARIANTS:raise ValueError(variant)
        self.variant=variant
        if variant=='no_predictor_context':
            warm_groups=[dict(q=masked_context(g['q'],g['t'])[0],t=masked_context(g['q'],g['t'])[1],gain=g['gain']) for g in warm_groups]
        super().__init__(warm_groups,k=k,adaptive=True,seeds=seeds)
        if variant in ['no_task_exchange','mean_mlp']:
            for model in self.models:
                model.attention.__class__=AblatedAttention
                model.attention.variant=variant

    def add(self,group):
        if self.variant=='no_predictor_context':
            q,t=masked_context(group['q'],group['t']);group=dict(q=q,t=t,gain=group['gain'])
        super().add(group)

    def predict(self,q,t):
        if self.variant=='no_predictor_context':q,t=masked_context(q,t)
        return super().predict(q,t)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def jsonable(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,Path):return str(v)
    raise TypeError(type(v).__name__)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=jsonable)+'\n',encoding='utf-8')
    temp.replace(path)


def torch_atomic(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp');torch.save(value,temp);temp.replace(path)


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def safe_unlink(path,root):
    path=Path(path).resolve();root=Path(root).resolve()
    if not path.is_relative_to(root) or path.name not in ['active_state.pt','active_manifest.json','inflight.json']:
        raise ValueError('Refuse to remove outside this formal run')
    path.unlink(missing_ok=True)


def save_active(folder,state,identity):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'active_state.pt'
    torch_atomic(path,dict(kind='formal_active_v1',identity=identity,search=state,torch_rng=torch.get_rng_state()))
    write(folder/'active_manifest.json',dict(sha256=sha(path),identity=identity,fits=state.ev.fit_count))
    write(folder/'progress.json',state.summary())


def restore_active(folder,identity):
    folder=Path(folder);m=read(folder/'active_manifest.json')
    if m['identity']!=identity or sha(folder/'active_state.pt')!=m['sha256']:raise RuntimeError('Active state identity/hash')
    d=torch.load(folder/'active_state.pt',weights_only=False,map_location='cpu');s=d['search']
    if d['kind']!='formal_active_v1' or d['identity']!=identity or s.pending is not None:raise RuntimeError('Incomplete active state')
    if s.ev.fit_count!=m['fits'] or len(s.ev.known)!=s.ev.fit_count or len(s.ev.ledger)!=s.ev.fit_count:raise RuntimeError('Active paid-fit ledger')
    if s.ev.paid_bits!={bits(subset) for subset in s.ev.known}:raise RuntimeError('Active cache identity')
    if 'k' in identity:
        v=identity['variant'];expected_arm=v if v in ['adaptive','frozen','diverse','greedy','warm'] else 'adaptive'
        if (s.ev.k,s.seed,s.arm,s.ev.budget)!=(identity['k'],identity['seed'],expected_arm,identity['budget']):raise RuntimeError('Active model/search identity')
        if v in VARIANTS and (not isinstance(s.learner,AblationEnsemble) or s.learner.variant!=v):raise RuntimeError('Ablation restored as wrong architecture')
        if v in ['adaptive','frozen'] and (isinstance(s.learner,AblationEnsemble) or s.learner.adaptive!=(v=='adaptive')):raise RuntimeError('Wrong full/frozen learner')
    journal=folder/'paid_batches.jsonl'
    if journal.exists():
        lines=journal.read_text(encoding='utf-8').splitlines()
        last=json.loads(lines[-1]) if lines else dict(fits=s.ev.fit_count)
        if last['fits']>s.ev.fit_count:raise RuntimeError('Hard interruption after last checkpoint; paid tail must be diagnosed, not silently rerun')
    marker=folder/'inflight.json'
    if marker.exists() and read(marker)['before_fits']>=s.ev.fit_count:
        raise RuntimeError('Unfinished batch after hard interruption; inspect before recovery')
    if s.learner is not None:s.learner.audit()
    torch.set_rng_state(d['torch_rng']);return s


class GracefulStop(Exception):pass


def drive(state,target,folder,identity,stop_requested,emit):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    tick=time.perf_counter();old=state.elapsed_seconds
    while state.ev.fit_count<target:
        if stop_requested():
            save_active(folder,state,identity);safe_unlink(folder/'inflight.json',folder)
            raise GracefulStop('Saved complete batch; resume with --resume')
        write(folder/'inflight.json',dict(identity=identity,before_fits=state.ev.fit_count))
        try:state.one_batch()
        except BaseException:
            torch_atomic(folder/'failure_state.pt',dict(identity=identity,search=state,torch_rng=torch.get_rng_state()))
            raise
        state.elapsed_seconds=old+time.perf_counter()-tick
        with (folder/'paid_batches.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(fits=state.ev.fit_count,decisions=len(state.decisions)))+'\n');f.flush()
        if state.ev.fit_count%800==0 or state.ev.fit_count==target:
            save_active(folder,state,identity)
            emit(f'{identity["stage"]} {identity["unit"]} seed={state.seed} K={state.ev.k} {identity["variant"]} fits={state.ev.fit_count}/{target} best_inner={state.ev.best()["mean_r2"]:.7f}')
        safe_unlink(folder/'inflight.json',folder)


def archive(folder,state,identity,root):
    """Retain exact learner/optimizer/RNG and paid logs, omit per-query fit caches."""
    folder=Path(folder);r=state.ev.known[state.ev.best_subset]
    payload=dict(kind='formal_completed_compact_v1',identity=identity,learner=state.learner,
        torch_rng=torch.get_rng_state(),search_rng=copy.deepcopy(state.rng.bit_generator.state),
        restart_rng=copy.deepcopy(state.restart_rng.bit_generator.state),roots=state.roots,
        current=state.current,episode=state.episode,step_in_episode=state.step_in_episode,
        context_sha256=state.ev.context_hash,summary=state.summary(),best_record=r)
    torch_atomic(folder/'completed_state.pt',payload)
    test=torch.load(folder/'completed_state.pt',weights_only=False,map_location='cpu')
    if test['identity']!=identity or test['summary']['actual_fits']!=state.ev.fit_count:raise RuntimeError('Compact identity')
    if not equal_tree(test['best_record'],r) or not equal_tree(test['torch_rng'],payload['torch_rng']):raise RuntimeError('Compact selected predictor/RNG')
    if state.learner is not None:
        for a,b in zip(state.learner.models,test['learner'].models):
            if not all(torch.equal(v,b.state_dict()[key]) for key,v in a.state_dict().items()):raise RuntimeError('Compact weights')
        test['learner'].audit()
        for a,b in zip(state.learner.optimizers,test['learner'].optimizers):
            if not equal_tree(a.state_dict(),b.state_dict()):raise RuntimeError('Compact optimizer state')
        if not equal_tree(state.learner.norm,test['learner'].norm) or not equal_tree(list(state.learner.groups),list(test['learner'].groups)):
            raise RuntimeError('Compact replay/normalizer')
        if state.learner.rng.bit_generator.state!=test['learner'].rng.bit_generator.state:raise RuntimeError('Compact sampler RNG')
    write(folder/'summary.json',state.summary());write(folder/'ledger.json',state.ev.ledger)
    write(folder/'decisions.json',state.decisions);write(folder/'restarts.json',state.restarts)
    pz=((state.ev.X_reward[:,state.ev.best_subset]-r['xmean'])/r['xstd'])@r['coef'].T+r['intercept']
    np.savez_compressed(folder/'inner_predictions.npz',prediction=pz*state.ev.context.std+state.ev.context.mean,target=state.ev.y_reward,subset=state.ev.best_subset)
    names=['completed_state.pt','summary.json','ledger.json','decisions.json','restarts.json','inner_predictions.npz']
    write(folder/'complete.json',dict(kind='formal_completed_compact_v1',identity=identity,files={n:sha(folder/n) for n in names},
        omitted='per-query coefficients/residual covariance;reconstructable from paid subset log and locked fit data;not a resumable active-search checkpoint'))
    # Only files in this new formal run are removed, after compact validation.
    for name in ['active_state.pt','active_manifest.json','inflight.json']:safe_unlink(folder/name,root)
    return state.summary()


def completed(folder,identity):
    folder=Path(folder);m=read(folder/'complete.json')
    if m['kind']!='formal_completed_compact_v1' or m['identity']!=identity:raise RuntimeError('Completed identity')
    for name,h in m['files'].items():
        if sha(folder/name)!=h:raise RuntimeError('Completed artifact hash '+str(folder/name))
    return read(folder/'summary.json')
