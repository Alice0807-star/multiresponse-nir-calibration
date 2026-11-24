"""Exact discrete log-probability feedback for the same hard task attention.

The estimator uses batches of independent complete K-band samples and a
leave-one-out baseline. It does NOT differentiate spectral mixtures or PLSR.
"""
import torch
from .direct_feedback import HardTaskAttention

class PopulationAttention(HardTaskAttention):
    def base_logits(self):
        A,R=self.attention();return (R@A)[self.task_ids].log()+self.position_bias

    def draw(self,gumbels):
        logits=self.base_logits();chosen=[];logps=[]
        for i in range(self.k):
            available=torch.ones(self.B,dtype=torch.bool);available[chosen]=False
            raw=logits[i].masked_fill(~available,-torch.inf)
            index=int((raw.detach()+gumbels[i]).argmax());chosen.append(index)
            logps.append(raw[index]-torch.logsumexp(raw,0))
        return tuple(chosen),torch.stack(logps).sum()

    def log_probability(self,chosen):
        logits=self.base_logits();previous=[];parts=[]
        for i,action in enumerate(chosen):
            available=torch.ones(self.B,dtype=torch.bool);available[previous]=False
            if not available[action]:raise ValueError('Invalid trajectory')
            raw=logits[i].masked_fill(~available,-torch.inf);parts.append(raw[action]-torch.logsumexp(raw,0));previous.append(action)
        return torch.stack(parts).sum()

def loo_loss(logps,scores):
    scores=torch.as_tensor(scores,dtype=logps.dtype)
    if len(scores)<2:raise ValueError('Complete population required')
    baseline=(scores.sum()-scores)/(len(scores)-1)
    return -((scores-baseline).detach()*logps).mean()
