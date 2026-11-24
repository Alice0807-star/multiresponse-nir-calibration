"""Hard K task-attention with direct PLSR prediction feedback.

The forward pass selects actual columns; straight-through derivatives are a
biased relaxation, NOT an unbiased discrete policy gradient. Gram PLS2 uses the
same regression deflation, scaling and NIPALS stopping tolerance as sklearn.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

class GramPLS:
    def __init__(self,Xf,yf,Xv,yv,components=4):
        arrays=[np.asarray(a,np.float64) for a in (Xf,yf,Xv,yv)]
        if any(not np.isfinite(a).all() for a in arrays):raise ValueError('Nonfinite data')
        xf,yf,xv,yv=arrays;self.n=len(xf);self.nv=len(xv);self.B=xf.shape[1];self.components=components
        xm,xs=xf.mean(0),xf.std(0,ddof=1);ym,ys=yf.mean(0),yf.std(0,ddof=1)
        if np.any(xs<=1e-12) or np.any(ys<=1e-12):raise ValueError('Constant input')
        xf=(xf-xm)/xs;xv=(xv-xm)/xs;yf=(yf-ym)/ys;yv=(yv-ym)/ys
        for key,a in dict(fxx=xf.T@xf,fxy=xf.T@yf,fyy=yf.T@yf,
            vxx=xv.T@xv,vxy=xv.T@yv,vyy=(yv*yv).sum(0),sst=((yv-yv.mean(0))**2).sum(0)).items():
            setattr(self,key,torch.as_tensor(a.copy(),dtype=torch.float64))

    def fit(self,S):
        if S.ndim!=2 or S.shape[1]!=self.B or S.shape[0]<self.components:raise ValueError('Selection matrix')
        gram=S@self.fxx@S.T;std=(torch.diagonal(gram)/(self.n-1)).sqrt()
        if bool((std.detach()<=1e-12).any()):raise FloatingPointError('Constant selected feature')
        G=gram/std[:,None]/std[None,:];C=(S@self.fxy)/std[:,None];YY=self.fyy.clone()
        eps=torch.finfo(S.dtype).eps;W=[];P=[];Q=[];iters=[]
        for component in range(self.components):
            v=torch.zeros(C.shape[1],dtype=S.dtype);v[0]=1.;old=None
            for iteration in range(500):
                w=(C@v)/(v@YY@v);w=w/(torch.linalg.vector_norm(w)+eps)
                denom=w@G@w;c=C.T@w/denom;v=c/(c@c+eps)
                if old is not None and float(((w-old).detach()**2).sum())<1e-6:break
                old=w
            else:raise RuntimeError('NIPALS did not converge')
            sign=torch.sign(w.detach()[int(w.detach().abs().argmax())]);w=w*sign
            denom=w@G@w;p=G@w/denom;q=C.T@w/denom
            if not bool(torch.isfinite(torch.cat([w,p,q])).all()) or float(denom.detach())<=1e-12:raise FloatingPointError('Degenerate PLS component')
            W.append(w);P.append(p);Q.append(q);iters.append(iteration+1)
            C=C-torch.outer(G@w,q);G=G-denom*torch.outer(p,p);YY=YY-denom*torch.outer(q,q)
        W=torch.stack(W,1);P=torch.stack(P,1);Q=torch.stack(Q,1)
        coef=W@torch.linalg.solve(P.T@W,Q.T);coef=coef/std[:,None]
        vg=S@self.vxx@S.T;vc=S@self.vxy
        sse=self.vyy-2*(coef*vc).sum(0)+(coef*(vg@coef)).sum(0)
        r2=1.-sse/self.sst
        if not bool(torch.isfinite(r2).all()) or bool((sse.detach()<-1e-8).any()):raise FloatingPointError('Nonfinite/negative PLS loss')
        return r2,dict(iterations=iters,coefficients=coef,sse=sse,
            nmse=sse.mean()/self.nv*self.n/(self.n-1))

    def columns(self,subset):
        s=tuple(sorted(int(j) for j in subset));S=torch.zeros(len(s),self.B,dtype=torch.float64)
        if len(set(s))!=len(s):raise ValueError('Duplicate column')
        S[torch.arange(len(s)),list(s)]=1.
        with torch.no_grad():return self.fit(S)

class HardTaskAttention(nn.Module):
    def __init__(self,A,R,centers,*,adaptive):
        super().__init__();A,R=np.asarray(A,float),np.asarray(R,float)
        if A.shape[0]!=3 or R.shape!=(3,3) or np.any(A<=0) or np.any(R<=0):raise ValueError('Positive 3-task A/R required')
        self.B=A.shape[1];self.k=len(centers);self.adaptive=adaptive
        self.gate_logits=nn.Parameter(torch.tensor(np.log(A),dtype=torch.float64),requires_grad=adaptive)
        self.relation_logits=nn.Parameter(torch.tensor(np.log(R),dtype=torch.float64),requires_grad=adaptive)
        x=torch.arange(self.B,dtype=torch.float64)/max(1,self.B-1)
        center=torch.tensor(centers,dtype=torch.float64)/max(1,self.B-1)
        self.register_buffer('position_bias',-.5*((x[None,:]-center[:,None])/.15)**2)
        self.register_buffer('task_ids',torch.arange(self.k)%3)

    def attention(self):return self.gate_logits.softmax(-1),self.relation_logits.softmax(-1)

    def forward(self,gumbels=None,temperature=1.):
        A,R=self.attention();mixed=R@A;logits=mixed[self.task_ids].log()+self.position_bias
        if gumbels is not None:logits=logits+gumbels
        chosen=[];softs=[];hards=[]
        for i in range(self.k):
            available=torch.ones(self.B,dtype=torch.bool);available[chosen]=False
            score=logits[i].masked_fill(~available,-torch.inf);p=(score/temperature).softmax(0);index=int(score.detach().argmax())
            hard=torch.zeros_like(p);hard[index]=1.;chosen.append(index);softs.append(p);hards.append(hard)
        soft=torch.stack(softs);hard=torch.stack(hards)
        # Exact hard forward; relaxed derivative only in backward.
        S=hard+(soft-soft.detach())
        if len(set(chosen))!=self.k or not torch.equal(S.detach(),hard):raise AssertionError('Hard K contract')
        return S,tuple(chosen),soft

def gumbel_noise(rng,k,B):
    u=np.clip(rng.random((k,B)),np.finfo(float).tiny,1.-np.finfo(float).eps)
    return torch.tensor(-np.log(-np.log(u)),dtype=torch.float64)
