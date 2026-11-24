"""Generic PLS evaluator with unchanged historical greedy search and CovSel.

CovSel follows the published covariance/projection criterion. X is centered,
Y is fit-standardized; this is an explicit setting, not a tuned scaling grid.
Reference: Roger et al.2011 doi:10.1016/j.chemolab.2010.10.003.
Author implementation inspected: ChemHouse-group/rchemo R/covsel.R (GPL-3).
This module implements the mathematical procedure, not the R package API.
"""
from types import SimpleNamespace
import hashlib,time,warnings
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from hsi_e2e.evaluation import canonical_subset,target_scale
from hsi_e2e.variable_k import bits
from hsi_e2e.pool_boundary import CapacitySearch


class GenericPLSEvaluator:
    def __init__(self,x,y,xv,yv,budget,*,k,components=8):
        x,y,xv,yv=[np.asarray(a,dtype=np.float64) for a in [x,y,xv,yv]]
        if any(a.ndim!=2 or not np.isfinite(a).all() for a in [x,y,xv,yv]):raise ValueError('Invalid arrays')
        if x.shape[1]!=xv.shape[1] or y.shape[1]!=yv.shape[1] or len(x)!=len(y) or len(xv)!=len(yv):
            raise ValueError('Array alignment')
        if not 1<=components<=min(k,len(x)-1) or not 1<=k<x.shape[1] or budget<16:
            raise ValueError('Invalid PLS/search dimensions')
        if np.any(x.std(0)<=1e-12):raise ValueError('Constant spectral column')
        mean,std=target_scale(y)
        self.context=SimpleNamespace(X=x,y=(y-mean)/std,mean=mean,std=std,n_bands=x.shape[1])
        self.X_reward=xv;self.y_reward=yv;self.y_reward_z=(yv-mean)/std
        self.denominator=((self.y_reward_z-self.y_reward_z.mean(0))**2).sum(0)
        if min(self.denominator)<=1e-12:raise ValueError('Constant reward target')
        self.k,self.components,self.budget=int(k),int(components),int(budget)
        self.fit_count,self.cache_hits,self.fit_seconds=0,0,0.
        self.known,self.ledger,self.paid_bits={},[],set();self.best_subset=None
        h=hashlib.sha256()
        for a in [x,y,xv,yv]:h.update(str(a.shape).encode());h.update(a.tobytes())
        h.update(f'generic_K{k}_PLS{components}_R2'.encode());self.context_hash=h.hexdigest()

    def evaluate(self,subset,*,parent=None,phase='search',extra=None):
        s=canonical_subset(subset,self.context.n_bands)
        if len(s)!=self.k:raise ValueError('Wrong fixed K')
        if s in self.known:self.cache_hits+=1;return self.known[s]
        if self.fit_count>=self.budget:raise RuntimeError('Paid fit budget exhausted')
        start=time.perf_counter()
        reg=PLSRegression(n_components=self.components,scale=True,max_iter=5000,tol=1e-6)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            reg.fit(self.context.X[:,s],self.context.y);pred=reg.predict(self.X_reward[:,s])
        self.fit_count+=1;self.fit_seconds+=time.perf_counter()-start
        if caught:raise RuntimeError('PLS warning:'+str(caught[0].message))
        if pred.ndim==1:pred=pred[:,None]
        if pred.shape!=self.y_reward_z.shape or not np.isfinite(pred).all():raise ValueError('PLS output contract')
        diff=self.y_reward_z-pred;r2=1-(diff**2).sum(0)/self.denominator;err=diff*self.context.std
        row=dict(r2=r2,mean_r2=float(r2.mean()),nmse=float((diff**2).mean()),
                 rmse=np.sqrt((err**2).mean(0)),mae=abs(err).mean(0))
        self.known[s]=row;self.paid_bits.add(bits(s))
        if self.best_subset is None or row['mean_r2']>self.known[self.best_subset]['mean_r2']:self.best_subset=s
        self.ledger.append(dict(actual_fits=self.fit_count,subset=list(s),
            parent=None if parent is None else list(parent),phase=phase,r2=r2.tolist(),
            mean_r2=row['mean_r2'],nmse=row['nmse'],incumbent_mean_r2=self.known[self.best_subset]['mean_r2'],**(extra or {})))
        return row

    def best(self):return dict(subset=list(self.best_subset),**self.known[self.best_subset])


def greedy(x,y,xv,yv,*,seed,k,budget,components=8,emit=None):
    ev=GenericPLSEvaluator(x,y,xv,yv,budget,k=k,components=components)
    state=CapacitySearch(ev,seed,'greedy');start=time.perf_counter();state.initialize()
    while ev.fit_count<budget:
        state.one_batch()
        if emit and ev.fit_count%1024==0:emit(f'GREEDY seed={seed} K={k} fits={ev.fit_count}/{budget}')
    state.elapsed_seconds=time.perf_counter()-start
    if ev.fit_count!=budget or ev.cache_hits:raise AssertionError('Unique paid query accounting')
    return state


def covsel(x,y,k):
    """Equal-row weighted CovSel: Xscaling=none,Yscaling=sd,ddof0."""
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid CovSel arrays')
    if not 1<=k<=min(x.shape[1],len(x)-1):raise ValueError('CovSel rank/cardinality')
    mean,std=target_scale(y);a=x-x.mean(0);b=(y-mean)/std
    selected=[]
    for _ in range(k):
        score=((a.T@b/len(a))**2).sum(1);score[selected]=-np.inf
        j=int(np.argmax(score));u=a[:,j].copy();den=float(u@u)
        if den<=1e-24 or not np.isfinite(score[j]):raise ValueError('CovSel residual rank exhausted')
        selected.append(j)
        a-=np.outer(u,u@a/den);b-=np.outer(u,u@b/den)
    return dict(order=selected,subset=sorted(selected),x_scaling='center only',y_scaling='fit sd ddof0',
                sample_weights='uniform',selection_pls_fits=0)
