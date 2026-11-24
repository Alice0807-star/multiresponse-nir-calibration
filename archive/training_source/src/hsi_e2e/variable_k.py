"""Variable-K form of the locked conditional MSE search, without new weights.

At K16 all mathematical operations, random draws and trainable tensors match
the prior version. Integer bitset cache checks and reuse of static survivor
geometry avoid constructing every hypothetical child tuple. K does not alter
the number of proposals evaluated or the sixteen-decision restart horizon.
"""
from collections import deque
import copy
import hashlib
import math
import time
import warnings

import numpy as np
import torch
from torch import nn
from sklearn.cross_decomposition import PLSRegression

from .conditional_feedback import ConditionalTaskAttention,ConditionalGainModel,QUERY_NAMES,TOKEN_NAMES
from .online_conditional import FitContext,PaidEvaluator,FeedbackEnsemble,OnlineSearch,select_diverse
from .evaluation import canonical_subset
from .local_search import initial_pool,swap_subset


def bits(subset):return sum(1<<int(i) for i in subset)


class VariableTaskAttention(ConditionalTaskAttention):
    def __init__(self,k,dimension=32):
        super().__init__(dimension);self.k=int(k)

    def forward(self,query,tokens):
        if query.shape[1:]!=(3,len(QUERY_NAMES)) or tokens.shape[1:]!=(3,self.k-1,len(TOKEN_NAMES)):
            raise ValueError('Variable-K feature contract')
        q=torch.tanh(self.query(query)+self.task_embedding)
        key=torch.tanh(self.key(tokens));value=torch.tanh(self.value(tokens))
        weights=(q[:,:,None,:]*key).sum(-1)/math.sqrt(self.dimension);weights=weights.softmax(-1)
        context=(weights[:,:,:,None]*value).sum(2)
        state=q+context;exchange,relation=self.task_relation(state,state,state,need_weights=True)
        return context,self.norm(state+exchange),weights,relation


class VariableGainModel(ConditionalGainModel):
    kind='conditional_three_target_gain_attention_variable_k_v1'
    def __init__(self,*,k,adaptive=True,dimension=32):
        nn.Module.__init__(self)
        self.k=int(k)
        if self.k!=k or self.k<4:raise ValueError('PLSR4 requires integer K>=4')
        self.attention=VariableTaskAttention(self.k,dimension)
        self.head=nn.Sequential(nn.Linear(len(QUERY_NAMES)+2*dimension,64),nn.GELU(),nn.Linear(64,1))
        for p in self.attention.parameters():p.requires_grad_(adaptive)
        self.adaptive=adaptive;self.dimension=dimension


class VariableFitContext(FitContext):
    def features(self,parent,paid,actions):
        parent=tuple(parent);k=len(parent)
        if k<4:raise ValueError('PLSR4 subset too small')
        coef,cov=paid['coef'],paid['residual_cov'];corr,tc=self.corr,self.target_corr
        if coef.shape!=(3,k):raise ValueError('Parent coefficient size')
        queries,tokens=[],[];geometry={}
        for d,a in actions:
            if d not in parent or a in parent or not 0<=a<self.n_bands:raise ValueError('Illegal complete swap')
            if d not in geometry:
                survivors=[j for j in parent if j!=d];di=parent.index(d);ci=[parent.index(j) for j in survivors]
                other=abs(corr[np.ix_(survivors,survivors)]).copy();np.fill_diagonal(other,0.)
                other=other.sum(1)/(k-2)
                geometry[d]=(survivors,di,ci,other)
            survivors,di,ci,other=geometry[d]
            ac,dc=corr[a,survivors],corr[d,survivors]
            q,t=[],[]
            for task in range(3):
                q.append([1-paid['r2'][task],paid['fit_error'][task],cov[a,task],cov[d,task],
                    tc[a,task],tc[d,task],coef[task,di],abs(coef[task]).mean(),corr[a,d],
                    abs(ac).max(),abs(ac).mean(),2*a/(self.n_bands-1)-1,
                    2*d/(self.n_bands-1)-1,tc[survivors,task].mean()])
                t.append(np.column_stack([ac,dc,tc[survivors,task],cov[survivors,task],coef[task,ci],
                    abs(coef[task,ci]),2*np.array(survivors)/(self.n_bands-1)-1,other]))
            queries.append(q);tokens.append(t)
        q,t=np.asarray(queries,dtype=np.float32),np.asarray(tokens,dtype=np.float32)
        if not np.isfinite(q).all() or not np.isfinite(t).all():raise FloatingPointError('Nonfinite K features')
        return q,t


class VariableEvaluator(PaidEvaluator):
    def __init__(self,X,y,X_reward,y_reward,budget,*,k):
        # Preserve the exact validated target normalization and evaluation data.
        super().__init__(X,y,X_reward,y_reward,budget)
        self.context.__class__=VariableFitContext  # Data-only subclass, no tensor/RNG mutation.
        self.k=int(k)
        if self.k!=k or not 4<=self.k<=self.context.n_bands:raise ValueError('Invalid K for PLSR4')
        self.paid_bits=set()
        if self.k!=16:
            digest=hashlib.sha256()
            for a in [X,y,X_reward,y_reward]:
                digest.update(str(a.shape).encode());digest.update(np.ascontiguousarray(a).tobytes())
            digest.update(f'K{self.k}_PLS4_scaleTrue_max5000_tol1e-6_R2_unclipped'.encode())
            self.context_hash=digest.hexdigest()

    def evaluate(self,subset,*,parent=None,phase='search',extra=None):
        s=canonical_subset(subset,self.context.n_bands)
        if len(s)!=self.k:raise ValueError('Wrong complete K')
        if s in self.known:
            self.cache_hits+=1;return self.known[s]
        if self.fit_count>=self.budget:raise RuntimeError('Paid fit budget exhausted')
        tick=time.perf_counter();reg=PLSRegression(n_components=4,scale=True,max_iter=5000,tol=1e-6)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');reg.fit(self.context.X[:,s],self.context.y)
            pred=reg.predict(self.X_reward[:,s]);fitpred=reg.predict(self.context.X[:,s])
        self.fit_count+=1;self.fit_seconds+=time.perf_counter()-tick
        if caught:raise RuntimeError(f'PLSR warning;K={self.k};paid={self.fit_count};subset={s};'+str(caught[0].message))
        coef=np.asarray(reg.coef_)
        if coef.shape!=(3,self.k):raise ValueError('PLSR coefficient convention changed')
        residual=self.context.y-fitpred
        cov=self.context.target_corr-self.context.corr[:,s]@(coef.T*np.sqrt((len(residual)-1)/len(residual)))
        if self.fit_count<=8:
            direct=self.context.xz.T@residual/len(residual)
            if not np.allclose(cov,direct,atol=1e-10,rtol=1e-8):raise AssertionError('PLSR covariance units')
            check=((self.context.X[:,s]-reg._x_mean)/reg._x_std)@coef.T+reg.intercept_
            if not np.allclose(check,fitpred,atol=1e-10,rtol=1e-8):raise AssertionError('PLSR prediction units')
        r2=1.-((self.y_reward_z-pred)**2).sum(0)/self.denominator
        err=(self.y_reward_z-pred)*self.context.std
        record=dict(r2=r2,mean_r2=float(r2.mean()),nmse=float(((self.y_reward_z-pred)**2).mean()),
            rmse=np.sqrt((err**2).mean(0)),mae=abs(err).mean(0),coef=coef.copy(),residual_cov=cov,
            fit_error=(residual**2).mean(0),xmean=reg._x_mean.copy(),xstd=reg._x_std.copy(),intercept=reg.intercept_.copy())
        if any(not np.isfinite(v).all() for v in record.values()):raise FloatingPointError('Nonfinite paid K context')
        self.known[s]=record;self.paid_bits.add(bits(s))
        if self.best_subset is None or record['mean_r2']>self.known[self.best_subset]['mean_r2']:self.best_subset=s
        self.ledger.append(dict(actual_fits=self.fit_count,subset=list(s),parent=None if parent is None else list(parent),
            phase=phase,r2=r2.tolist(),mean_r2=record['mean_r2'],nmse=record['nmse'],
            incumbent_mean_r2=self.known[self.best_subset]['mean_r2'],**(extra or {})))
        return record


def variable_pool(ev,parent,rng,size=128):
    parent=tuple(int(j) for j in parent)
    k=len(parent);base=bits(parent);selected=set(parent)
    available=[i for i in range(ev.context.n_bands) if i not in selected]
    quotient,remainder=divmod(size,k)
    extra=set(rng.choice(k,remainder,replace=False).tolist()) if remainder else set()
    actions=[]
    for idx,d in enumerate(parent):
        count=quotient+int(idx in extra)
        if count==0:continue
        removed=base^(1<<d)
        unseen=[a for a in available if (removed|(1<<a)) not in ev.paid_bits]
        if len(unseen)<count:raise RuntimeError('Unseen parent neighborhood exhausted;do not alter pool silently')
        actions.extend((int(d),int(a)) for a in rng.choice(unseen,count,replace=False))
    if len(actions)!=size:raise AssertionError('Candidate pool size')
    return actions


def variable_diverse(actions,predictions,corr,rng,count=8):
    unique_deletes=len({d for d,a in actions})
    if unique_deletes>=count:return select_diverse(actions,predictions,corr,rng,count=count)
    # K4 boundary: cover each deletion before a second deletion cycle. All
    # indices remain unique and add-correlation control spans the full batch.
    selected,relax=[],[];used=set();random_order=rng.permutation(len(actions)).tolist()
    slots=[0,None,1,None,2,None,'mean',None]
    for slot in slots[:count]:
        if len(used)==unique_deletes:used.clear()
        if predictions is None or slot is None:order=random_order
        else:
            score=predictions.mean(1) if slot=='mean' else predictions[:,slot]
            order=np.argsort(-score,kind='stable').tolist()
        chosen=None
        for threshold in [.95,.99,1.0000001]:
            for i in order:
                d,a=actions[i]
                if i in selected or d in used:continue
                if any(abs(corr[a,actions[j][1]])>=threshold for j in selected):continue
                chosen=i;break
            if chosen is not None:relax.append(threshold);break
        if chosen is None:raise RuntimeError('K4 diverse batch exhausted')
        selected.append(chosen);used.add(actions[chosen][0])
    return selected,relax


class VariableEnsemble(FeedbackEnsemble):
    def __init__(self,warm_groups,*,k,adaptive,seeds=(13,37,73)):
        self.k=int(k)
        q=np.concatenate([g['q'] for g in warm_groups]);t=np.concatenate([g['t'] for g in warm_groups])
        self.norm=dict(qmu=q.mean((0,1),keepdims=True),qsd=np.maximum(q.std((0,1),keepdims=True),1e-6),
            tmu=t.mean((0,1,2),keepdims=True),tsd=np.maximum(t.std((0,1,2),keepdims=True),1e-6))
        self.models,self.optimizers,self.initial_attention=[],[],[]
        self.rng=np.random.default_rng(624000);self.groups=deque(maxlen=512)
        self.updates,self.adaptive,self.last_loss=0,adaptive,None
        self.min_attention_gradient=float('inf') if adaptive else 0.;self.max_gradient=0.
        for g in warm_groups:self.add(g)
        for seed in seeds:
            torch.manual_seed(seed);m=VariableGainModel(k=k,adaptive=adaptive)
            self.models.append(m);self.initial_attention.append(copy.deepcopy(m.attention.state_dict()))
            self.optimizers.append(torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.001))


class VariableSearch(OnlineSearch):
    def __init__(self,evaluator,seed,arm,*,learner=None):
        if arm not in ['warm','adaptive','frozen','diverse','greedy']:raise ValueError(arm)
        if evaluator.k>=evaluator.context.n_bands:raise ValueError('Full spectrum is a reference fit,not swap search')
        self.ev,self.seed,self.arm,self.learner=evaluator,seed,arm,learner
        self.rng=np.random.default_rng(seed+100000);self.restart_rng=np.random.default_rng(seed+910000)
        self.roots=initial_pool(evaluator.context.n_bands,evaluator.k,seed)
        self.episode,self.step_in_episode=0,16;self.current=None
        self.decisions,self.restarts,self.warm_groups=[],[],[];self.elapsed_seconds=0.;self.pending=None

    def one_batch(self):
        if self.pending is None:
            if self.step_in_episode==16:self.restart()
            parent=self.current
            if self.arm=='greedy':
                actions=[];ps=set(parent);base=bits(parent)
                available=[a for a in range(self.ev.context.n_bands) if a not in ps]
                attempts=0
                while len(actions)<8:
                    attempts+=1
                    if attempts>10000:raise RuntimeError('Unseen greedy proposal cap')
                    d=int(self.rng.choice(parent));a=int(self.rng.choice(available))
                    if (d,a) not in actions and ((base^(1<<d))|(1<<a)) not in self.ev.paid_bits:actions.append((d,a))
                predictions,thresholds,selected=None,[],list(range(8))
            else:
                actions=variable_pool(self.ev,parent,self.rng)
                if self.learner is None:predictions=None
                else:
                    q,t=self.ev.context.features(parent,self.ev.known[parent],actions)
                    predictions=self.learner.predict(q,t)
                selected,thresholds=variable_diverse(actions,predictions,self.ev.context.corr,self.rng)
            remaining=min(8,self.ev.budget-self.ev.fit_count)
            self.pending=dict(parent=parent,actions=[actions[i] for i in selected[:remaining]],
                prediction=None if predictions is None else predictions[selected[:remaining]].tolist(),
                thresholds=thresholds[:remaining],cursor=0,children=[])
        p=self.pending;parent=p['parent']
        for d,a in p['actions'][p['cursor']:]:
            child=swap_subset(parent,d,a)
            self.ev.evaluate(child,parent=parent,phase='warm' if self.arm=='warm' else 'search',
                extra=dict(remove=d,add=a,decision=len(self.decisions)))
            p['children'].append(child);p['cursor']+=1
        children=p['children']
        gain=np.array([self.ev.known[s]['r2']-self.ev.known[parent]['r2'] for s in children],dtype=np.float32)
        if self.arm=='warm' or self.learner is not None:
            q,t=self.ev.context.features(parent,self.ev.known[parent],p['actions']);group=dict(q=q,t=t,gain=gain)
            if self.arm=='warm':self.warm_groups.append(group)
            else:self.learner.add(group);self.learner.update()
        child=max(children,key=lambda s:self.ev.known[s]['mean_r2'])
        take=self.ev.known[child]['mean_r2']>self.ev.known[parent]['mean_r2']
        self.decisions.append(dict(parent=list(parent),children=[list(s) for s in children],actions=p['actions'],
            prediction=p['prediction'],gain=gain.tolist(),thresholds=p['thresholds'],accepted=take,
            selected=list(child),actual_fits=self.ev.fit_count,parent_mean_r2=self.ev.known[parent]['mean_r2']))
        if take:self.current=child
        self.step_in_episode+=1;self.pending=None

    def summary(self):return dict(**super().summary(),k=self.ev.k,engine='variable_K_MSE_v1')
