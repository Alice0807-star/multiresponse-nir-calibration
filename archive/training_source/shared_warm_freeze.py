"""Fork an already trained warm learner; never reinitialize its optimizer."""
import copy
import numpy as np
import torch
from hsi_e2e.variable_k import VariableEnsemble
from hsi_e2e.conditional_feedback import actual_gain_feedback


class HeadOnlyAfterWarm(VariableEnsemble):
    """Same forward/backward kernels;discard attention gradients before clipping.

    Keeping requires_grad flags unchanged prevents a flag-dependent numerical
    dispatch difference from becoming a second factor. Adam never receives
    an attention gradient, so warm weights and moments remain exactly fixed.
    """
    def update(self):
        ids=self.rng.integers(len(self.groups),size=16)
        samples=[]
        for i in ids:
            group=self.groups[i]
            j=self.rng.choice(len(group['gain']),8,replace=len(group['gain'])<8)
            samples.append({k:v[j] for k,v in group.items()})
        q,t=self.tensors(np.concatenate([g['q'] for g in samples]),np.concatenate([g['t'] for g in samples]))
        y=torch.from_numpy(np.concatenate([g['gain'] for g in samples]).astype('float32'))
        losses=[]
        for model,optimizer in zip(self.models,self.optimizers):
            optimizer.zero_grad(set_to_none=True)
            loss=actual_gain_feedback(model(q,t),y);loss.backward()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError('Nonfinite head-only feedback gradient')
            for p in model.attention.parameters():p.grad=None
            gn=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.))
            self.max_gradient=max(self.max_gradient,gn)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite head-only parameter')
            losses.append(float(loss.detach()))
        self.last_loss=float(np.mean(losses));self.updates+=1


def fork_after_warm(prepared,condition):
    if condition not in ['online_full','head_only']:
        raise ValueError('Unregistered shared-warm condition')
    state=copy.deepcopy(prepared)
    learner=state.learner
    if learner is None or not learner.adaptive or learner.updates!=512:
        raise ValueError('Require exactly512 full warm updates before branching')
    if condition=='head_only':
        learner.__class__=HeadOnlyAfterWarm
        state.arm='frozen';learner.adaptive=False
        learner.initial_attention=[copy.deepcopy(m.attention.state_dict()) for m in learner.models]
        learner.min_attention_gradient=0.
        learner.max_gradient=0.
        for model in learner.models:
            model.adaptive=False
            # Adam retains all warm moments and the same parameter groups.
            # update discards attention gradients before clipping and Adam.
    else:state.arm='adaptive'
    learner.audit()
    return state
