"""Isolated offline TD diagnostic; true-R2 guards run before optimizer.step."""
import numpy as np
import torch
from torch.nn import functional as F


class ValueHealthProblem(RuntimeError):
    pass


class BatchGuard:
    def __init__(self,warmup=128,patience=8,persistent_limit=.25,immediate_limit=1.):
        self.warmup,self.patience=int(warmup),int(patience)
        self.persistent_limit,self.immediate_limit=float(persistent_limit),float(immediate_limit)
        self.streak=0

    def observe(self,update,metrics):
        if not all(np.isfinite(v) for v in metrics.values()):
            raise ValueHealthProblem('nonfinite batch statistic')
        violation=max(metrics['q_bound_violation'],metrics['target_bound_violation'])
        if violation>self.immediate_limit:
            raise ValueHealthProblem('immediate severe true-R2 bound violation')
        self.streak=self.streak+1 if update>self.warmup and violation>self.persistent_limit else 0
        if self.streak>=self.patience:
            raise ValueHealthProblem('persistent true-R2 bound violation')


def td_terms(online,target,batch,terminal_weight=1.):
    states=torch.from_numpy(np.stack([t.state for t in batch]))
    actions=torch.tensor([t.action for t in batch],dtype=torch.long)
    reward=torch.tensor([t.reward for t in batch],dtype=torch.float32)
    done=torch.tensor([t.done for t in batch],dtype=torch.bool)
    nxt=torch.from_numpy(np.stack([t.next_state for t in batch]))
    masks=torch.from_numpy(np.stack([t.next_mask for t in batch])).bool()
    selected=online(states).gather(1,actions[:,None]).squeeze(1)
    with torch.no_grad():
        bootstrap=torch.zeros_like(reward)
        active=~done
        if active.any():
            if masks[active].all(1).any():raise ValueError('No legal next action')
            choice=online(nxt[active]).masked_fill(masks[active],-torch.inf).argmax(1)
            bootstrap[active]=target(nxt[active]).gather(1,choice[:,None]).squeeze(1)
        labels=reward+bootstrap
    losses=F.smooth_l1_loss(selected,labels,reduction='none')
    weights=torch.where(done,float(terminal_weight),1.)
    loss=(weights*losses).sum()/weights.sum()
    return loss,selected,labels,done


def batch_metrics(loss,q,labels,bounds):
    b=torch.as_tensor(bounds,dtype=q.dtype)
    return dict(loss=float(loss.detach()),q_abs_max=float(q.detach().abs().max()),
        target_abs_max=float(labels.abs().max()),
        q_bound_violation=float((q.detach()-b).clamp_min(0).max()),
        target_bound_violation=float((labels-b).clamp_min(0).max()))


def guarded_update(online,target,optimizer,batch,bounds,guard,update,terminal_weight=1.):
    loss,q,labels,done=td_terms(online,target,batch,terminal_weight)
    metrics=batch_metrics(loss,q,labels,bounds)
    guard.observe(update,metrics)  # Abort BEFORE modifying parameters.
    optimizer.zero_grad(set_to_none=True);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(online.parameters(),5.)
    if not torch.isfinite(norm):raise ValueHealthProblem('nonfinite gradient')
    optimizer.step()
    metrics.update(gradient_norm=float(norm),terminal_fraction=float(done.float().mean()))
    return metrics
