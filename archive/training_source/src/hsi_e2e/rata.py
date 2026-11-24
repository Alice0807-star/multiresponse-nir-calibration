"""On-policy reward-adaptive task gates/relations; no differentiated PLSR.

The frozen/adaptive arms differ ONLY in requires_grad of A/R logits.
This is a new conditional actor, not a continuation of the archived DQN.
"""
from __future__ import annotations
import copy,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .local_search import initial_pool,swap_subset


def features_from_fit(X,y):
    def z(a):return (a-a.mean(0))/np.maximum(a.std(0),1e-12)
    X,y=np.asarray(X,float),np.asarray(y,float)
    xz,yz=z(X),z(y)
    correlation=np.abs(xz.T@xz/len(X));np.fill_diagonal(correlation,0.)
    static=np.c_[np.linspace(-1.,1.,X.shape[1]),xz.T@yz/len(X)]
    return static.astype('float32'),correlation.astype('float32')


class RataPolicy(nn.Module):
    def __init__(self,A,R,static,correlation,*,adaptive,k=16):
        super().__init__();A,R=np.asarray(A),np.asarray(R)
        if A.shape[0]!=3 or R.shape!=(3,3) or np.any(A<=0) or np.any(R<=0):raise ValueError('Expected positive3-task initial A/R')
        self.B=A.shape[1];self.k=int(k);self.adaptive=bool(adaptive)
        self.gate_logits=nn.Parameter(torch.log(torch.as_tensor(A.copy(),dtype=torch.float32)),requires_grad=adaptive)
        self.relation_logits=nn.Parameter(torch.log(torch.as_tensor(R.copy(),dtype=torch.float32)),requires_grad=adaptive)
        self.register_buffer('static',torch.as_tensor(static.copy()))
        self.register_buffer('correlation',torch.as_tensor(correlation.copy()))
        self.scorer=nn.Sequential(nn.Linear(21,64),nn.Tanh(),nn.Linear(64,1))

    def attention(self):return self.gate_logits.softmax(-1),self.relation_logits.softmax(-1)

    def forward(self,subset,costs,step,pending=None):
        # Costs are already-paid per-target1-R2, not predicted future labels.
        mask=torch.zeros(self.B,dtype=torch.bool);mask[list(subset)]=True
        if mask.sum()!=self.k:raise ValueError('Expected complete fixed-K state')
        if pending is not None and not mask[pending]:raise ValueError('Illegal pending deletion')
        survivors=mask.clone()
        if pending is not None:survivors[pending]=False
        A,R=self.attention();reference=A.topk(self.k,dim=1).values.sum(1)
        coverage=((A*survivors).sum(1)/reference).clamp(0.,1.)
        demand=(1.-coverage).clamp_min(.05);mixed=.5*demand+.5*(R@demand)
        weighted=(A*mixed[:,None])*self.B
        # Candidate redundancy with survivors, excluding each candidate itself.
        corr=self.correlation[:,survivors]
        maximum=corr.max(1).values
        denominator=survivors.sum().float()-survivors.float()
        average=corr.sum(1)/denominator.clamp_min(1.)
        marker=torch.zeros(self.B)
        if pending is not None:marker[pending]=1.
        pending_corr=torch.zeros(self.B) if pending is None else self.correlation[:,pending]
        distance=torch.zeros(self.B) if pending is None else abs(torch.arange(self.B)-pending).float()/max(1,self.B-1)
        constants=torch.tensor([float(pending is not None),(16-step)/16,*np.asarray(costs,float).tolist()],dtype=torch.float32)
        inputs=torch.cat([self.static,A.T*self.B,weighted.T,mask.float()[:,None],marker[:,None],
            maximum[:,None],average[:,None],pending_corr[:,None],distance[:,None],constants[None].expand(self.B,-1)],1)
        if inputs.shape!=(self.B,21):raise AssertionError('Actor feature contract')
        logits=self.scorer(inputs).squeeze(1)
        legal=(~mask) if pending is not None else mask
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite actor logits')
        return logits.masked_fill(~legal,-torch.inf),legal


def score(ev,subset):
    ev.evaluate(subset);m=ev.reward_metrics(subset)
    if ev.warning_messages:raise RuntimeError('PLSR warning:'+ev.warning_messages[-1])
    return float(m['mean_r2']),np.array(1.-m['r2'],copy=True)


class RataTrainer:
    def __init__(self,ev,probe_ev,A,R,static,correlation,*,seed,adaptive):
        self.ev,self.pev=ev,probe_ev;self.seed=int(seed);self.adaptive=adaptive
        torch.manual_seed(seed);self.model=RataPolicy(A,R,static,correlation,adaptive=adaptive,k=ev.k)
        actor=list(self.model.scorer.parameters());groups=[dict(params=actor,lr=.001)]
        if adaptive:groups.append(dict(params=[self.model.gate_logits,self.model.relation_logits],lr=.0001))
        self.optimizer=torch.optim.Adam(groups)
        self.action_rng=torch.Generator().manual_seed(seed+1500000)
        self.restart_rng=np.random.default_rng(seed+910000)
        self.pool=initial_pool(ev.n_bands,ev.k,seed);self.probe_pool=initial_pool(ev.n_bands,ev.k,seed+700000)
        self.training_sets={};self.episode=self.step=self.swaps=self.updates=0
        self.baseline=np.zeros(32);self.pending=[];self.gradient_rows=[];self.training_trace=[];self.restarts=[];self.probe_paths=[]
        self.started=time.perf_counter();self.attempts_limit=4*(3200-16)
        for subset in self.pool:
            value,_=score(ev,subset);self.training_sets[tuple(subset)]=value
        self._restart()

    def _restart(self):
        coin=float(self.restart_rng.random());index=self.episode%16;rank=None
        if coin<.5:
            elite=sorted(self.training_sets.items(),key=lambda p:(-p[1],p[0]))[:16]
            rank=int(self.restart_rng.integers(len(elite)));root=elite[rank][0]
        else:root=tuple(self.pool[index])
        self.current=tuple(root);self.current_j,self.current_costs=score(self.ev,root)
        self.start_j=self.current_j
        self.restarts.append(dict(episode=self.episode+1,source='elite' if coin<.5 else 'original',coin=coin,
            elite_rank=None if rank is None else rank+1,training_fits=self.ev.fit_count,root=list(root),start_mean_r2=self.current_j))

    def select(self,subset,costs,step,pending=None,*,deterministic=False):
        logits,legal=self.model(subset,costs,step,pending)
        logp=logits.log_softmax(0)
        if deterministic:action=int(logits.argmax())
        else:action=int(torch.multinomial(logp.detach().exp(),1,generator=self.action_rng))
        if not legal[action]:raise AssertionError('Invalid actor action')
        return action,logp[action]

    def _episode_update(self):
        returns=[];future=0.
        for item in reversed(self.pending):future+=item['reward'];returns.append(future)
        returns=np.array(returns[::-1]);assert len(returns)==32
        if not np.isclose(returns[0],self.current_j-self.start_j,atol=1e-10):raise AssertionError('R2 reward does not telescope')
        # Baseline comes exclusively from PREVIOUS complete episodes.
        advantage=torch.tensor(returns-self.baseline,dtype=torch.float32)
        logp=torch.stack([item['logp'] for item in self.pending]);loss=-(advantage*logp).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite REINFORCE loss')
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        gradient={}
        for name,param in self.model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():raise FloatingPointError('Nonfinite actor gradient')
                gradient[name]=float(param.grad.detach().double().square().sum().sqrt())
        norm=torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],5.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
        self.optimizer.step();self.updates+=1
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):raise FloatingPointError('Nonfinite parameters')
        self.gradient_rows.append(dict(update=self.updates,episode=self.episode+1,training_fits=self.ev.fit_count,
            loss=float(loss.detach()),gradient_norm=float(norm),gate_gradient=gradient.get('gate_logits',0.),
            relation_gradient=gradient.get('relation_logits',0.),actual_return=float(returns[0])))
        self.baseline=.9*self.baseline+.1*returns
        self.pending=[]

    def advance(self,target):
        while self.ev.fit_count<target:
            if self.swaps>=self.attempts_limit:raise RuntimeError('Actor attempt cap; stop all seeds')
            parent=self.current;before=self.current_j;costs=self.current_costs.copy();step=self.step
            remove,lp_remove=self.select(parent,costs,step)
            add,lp_add=self.select(parent,costs,step,remove)
            child=swap_subset(parent,remove,add);oldfits=self.ev.fit_count
            after,nextcosts=score(self.ev,child);reward=after-before
            self.pending.extend([dict(subset=parent,costs=costs,step=step,pending=None,action=remove,reward=0.,logp=lp_remove),
                dict(subset=parent,costs=costs,step=step,pending=remove,action=add,reward=reward,logp=lp_add)])
            self.training_sets[child]=after;self.current=child;self.current_j=after;self.current_costs=nextcosts
            self.swaps+=1;self.step+=1
            self.training_trace.append(dict(episode=self.episode+1,step=self.step,current_subset_json=json.dumps(parent),
                subset_json=json.dumps(child),remove=remove,add=add,before_mean_r2=before,after_mean_r2=after,reward=reward,
                fit_count=self.ev.fit_count,cache_hit=self.ev.fit_count==oldfits,done=self.step==16))
            if self.step==16:
                self._episode_update();self.episode+=1;self.step=0;self._restart()

    def probe(self,checkpoint):
        # Deterministic, no optimizer/backward or training/action/restart RNG.
        torch_before=torch.get_rng_state().clone();action_before=self.action_rng.get_state().clone()
        restart_before=copy.deepcopy(self.restart_rng.bit_generator.state);values=[]
        with torch.no_grad():
            for index in [0,4,8,12]:
                current=tuple(self.probe_pool[index]);value,costs=score(self.pev,current);start=value
                for step in range(16):
                    remove,_=self.select(current,costs,step,deterministic=True)
                    add,_=self.select(current,costs,step,remove,deterministic=True)
                    after=swap_subset(current,remove,add);nextvalue,nextcosts=score(self.pev,after)
                    self.probe_paths.append(dict(checkpoint=checkpoint,index=index,step=step+1,current_json=json.dumps(current),
                        subset_json=json.dumps(after),remove=remove,add=add,before_r2=value,after_r2=nextvalue))
                    current,value,costs=after,nextvalue,nextcosts
                values.append(dict(root_index=index,start_r2=start,end_r2=value,return_=value-start))
        assert torch.equal(torch_before,torch.get_rng_state()) and torch.equal(action_before,self.action_rng.get_state())
        assert restart_before==self.restart_rng.bit_generator.state
        return dict(checkpoint=checkpoint,mean_end_r2=float(np.mean([r['end_r2'] for r in values])),
            mean_return=float(np.mean([r['return_'] for r in values])),training_fits=self.ev.fit_count,probe_fits=self.pev.fit_count,
            actual_total_fits=self.ev.fit_count+self.pev.fit_count,updates=self.updates,panels=values)

    def save(self,path):
        # Record unfinished raw transitions; graphs can be rebuilt without any
        # new action sampling because parameters never change mid-episode.
        from .training_snapshot import EV_FIELDS
        payload=dict(format_version=1,kind='rata_onpolicy_full_state',seed=self.seed,adaptive=self.adaptive,
            model={k:v.detach().clone() for k,v in self.model.state_dict().items()},optimizer=self.optimizer.state_dict(),
            action_rng=self.action_rng.get_state(),restart_rng=copy.deepcopy(self.restart_rng.bit_generator.state),
            attributes={k:v for k,v in self.__dict__.items() if k not in ['ev','pev','model','optimizer','action_rng','restart_rng','pending']},
            pending=[{k:v for k,v in r.items() if k!='logp'} for r in self.pending],
            evaluators={name:dict(fields={k:getattr(ev,k) for k in EV_FIELDS if hasattr(ev,k)},budget=ev.fit_budget,
                context=ev.cache_context_sha256,scope=getattr(ev,'scope',''),ledger=getattr(ev,'ledger',[])) for name,ev in [('ev',self.ev),('pev',self.pev)]})
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        torch.save(payload,path)


def load_rata_state(path,evaluator_factory):
    # Trusted locally produced full states contain numpy and require pickle.
    before=torch.get_rng_state().clone()
    try:
        p=torch.load(path,map_location='cpu',weights_only=False)
        if p.get('kind')!='rata_onpolicy_full_state' or p.get('format_version')!=1:raise ValueError('Not a RATA state')
        t=RataTrainer.__new__(RataTrainer);t.__dict__.update(p['attributes'])
        for name,packed in p['evaluators'].items():
            ev=evaluator_factory(packed['budget'],packed['scope'],packed['ledger'])
            if ev.cache_context_sha256!=packed['context'] or ev.fit_count!=0:raise ValueError('Immutable data mismatch')
            for key,value in packed['fields'].items():setattr(ev,key,value)
            setattr(t,name,ev)
        model=p['model'];A=model['gate_logits'].softmax(-1).numpy();R=model['relation_logits'].softmax(-1).numpy()
        t.model=RataPolicy(A,R,model['static'].numpy(),model['correlation'].numpy(),adaptive=p['adaptive'],k=t.ev.k)
        t.model.load_state_dict(model);groups=[dict(params=list(t.model.scorer.parameters()),lr=.001)]
        if t.adaptive:groups.append(dict(params=[t.model.gate_logits,t.model.relation_logits],lr=.0001))
        t.optimizer=torch.optim.Adam(groups);t.optimizer.load_state_dict(p['optimizer'])
        t.action_rng=torch.Generator();t.action_rng.set_state(p['action_rng'])
        t.restart_rng=np.random.default_rng();t.restart_rng.bit_generator.state=p['restart_rng']
        t.pending=[]
        for raw in p['pending']:
            logits,_=t.model(raw['subset'],raw['costs'],raw['step'],raw['pending']);item=dict(raw)
            item['logp']=logits.log_softmax(0)[raw['action']];t.pending.append(item)
        return t
    finally:torch.set_rng_state(before)


def run_r2_greedy(ev,pool,*,seed):
    started=time.perf_counter();rng=np.random.default_rng(seed+100000);restart=np.random.default_rng(seed+910000)
    known={};trace=[];decisions=[];restarts=[]
    def evaluate(s):
        j,_=score(ev,s);known[tuple(s)]=j
        trace.append(dict(fit_count=ev.fit_count,subset_json=json.dumps(s),mean_r2=j))
        return j
    for s in pool:evaluate(s)
    episode=0
    while ev.fit_count<ev.fit_budget:
        coin=float(restart.random());rank=None
        if coin<.5:
            elite=sorted(known.items(),key=lambda p:(-p[1],p[0]))[:16];rank=int(restart.integers(len(elite)));current,j=elite[rank]
        else:current=tuple(pool[episode%16]);j=known[current]
        restarts.append(dict(episode=episode+1,coin=coin,rank=rank,root=current,fit_count=ev.fit_count))
        for step in range(16):
            if ev.fit_count>=ev.fit_budget:break
            candidates=[];attempt=0
            while len(candidates)<8 and ev.fit_count<ev.fit_budget:
                attempt+=1
                if attempt>10000:raise RuntimeError('Greedy unseen-candidate cap')
                remove=int(rng.choice(current));adds=[a for a in range(ev.n_bands) if a not in current];add=int(rng.choice(adds))
                child=swap_subset(current,remove,add)
                if child in known:continue
                candidates.append((child,evaluate(child)))
            selected,value=max(candidates,key=lambda p:p[1]);take=value>j
            decisions.append(dict(episode=episode+1,step=step+1,parent=current,before=j,selected=selected,value=value,accepted=take,fit_count=ev.fit_count))
            if take:current,j=selected,value
        episode+=1
    return dict(trace=trace,decisions=decisions,restarts=restarts,seconds=time.perf_counter()-started,
        best_subset=max(known,key=known.get),best_mean_r2=max(known.values()))
