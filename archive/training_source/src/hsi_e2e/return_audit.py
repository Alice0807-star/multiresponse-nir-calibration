"""Fixed-trajectory return prediction controls; no data/test loading here."""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F


def state_for(subset, error, step, *, n_bands, horizon=16, pending=None):
    mask=np.zeros(n_bands,bool)
    mask[list(subset)]=True
    if pending is not None and not mask[pending]:
        raise ValueError('Removal must belong to current complete subset')
    state=np.zeros(4*n_bands+3,np.float32)
    state[:n_bands]=mask
    if pending is not None:
        state[n_bands+pending]=1.
    state[-3:]=[error,float(pending is not None),(horizon-step)/horizon]
    return state,mask.copy() if pending is not None else ~mask


def invalid_actions(states,n_bands):
    mask=states[:,:n_bands]>.5
    return torch.where((states[:,-2]>.5)[:,None],mask,~mask)


def episode_arrays(records,final_state,final_error,episode_id):
    """Both stages share the same known terminal cost, not an estimated label."""
    states=np.stack([r['state'] for r in records])
    actions=np.array([r['action'] for r in records],np.int64)
    n=len(records)
    rewards=np.array([r['reward'] for r in records],np.float64)
    end=np.full(n,final_error,np.float64)
    # Complete episodes only; do not turn collection cutoffs into terminal states.
    done=np.zeros(n,bool)
    done[-1]=True
    next_states=np.concatenate([states[1:],final_state[None]],axis=0)
    result=dict(states=states,actions=actions,rewards=rewards,
        next_states=next_states,next_actions=np.r_[actions[1:],0],done=done,
        final_cost=end,episode_id=np.full(n,episode_id,np.int64))
    if not np.allclose(np.cumsum(rewards[::-1])[::-1],states[:,-3]-final_error,atol=1e-7):
        raise AssertionError('Complete return did not telescope')
    return result


def concatenate_episodes(episodes):
    return {key:np.concatenate([e[key] for e in episodes],axis=0) for key in episodes[0]}


def batch_targets(mode,online,target,batch,n_bands):
    if mode=='mc':
        return batch['final_cost']
    done=batch['done']
    values=batch['next_states'][:,-3].clone()
    active=~done
    if active.any():
        ns=batch['next_states'][active]
        if mode=='sarsa':
            actions=batch['next_actions'][active]
        elif mode=='max_td':
            # min terminal cost == max improvement Q at this same state.
            online_cost=F.softplus(online(ns))
            q=ns[:,-3,None]-online_cost
            actions=q.masked_fill(invalid_actions(ns,n_bands),-torch.inf).argmax(1)
        else:
            raise ValueError(mode)
        values[active]=F.softplus(target(ns)).gather(1,actions[:,None]).squeeze(1)
    return values


def tensor_batch(data,indices):
    keys=['states','actions','next_states','next_actions','done','final_cost']
    batch={k:torch.from_numpy(data[k][indices]) for k in keys}
    batch['final_cost']=batch['final_cost'].float()
    return batch


def update(online,target,optimizer,batch,mode,n_bands):
    cost=F.softplus(online(batch['states'])).gather(1,batch['actions'][:,None]).squeeze(1)
    with torch.no_grad():
        truth=batch_targets(mode,online,target,batch,n_bands)
    loss=F.smooth_l1_loss(cost,truth)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite return prediction loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(online.parameters(),5.)
    optimizer.step()
    return dict(loss=float(loss.detach()),target_mean=float(truth.mean()),
        prediction_mean=float(cost.detach().mean()),gradient_norm=float(grad))


def predict_cost(net,data):
    chunks=[]
    with torch.no_grad():
        for offset in range(0,len(data['actions']),512):
            states=torch.from_numpy(data['states'][offset:offset+512])
            actions=torch.from_numpy(data['actions'][offset:offset+512])
            chunks.append(F.softplus(net(states)).gather(1,actions[:,None]).squeeze(1).numpy())
    return np.concatenate(chunks)


def cost_metrics(prediction,data):
    error=np.asarray(prediction,dtype=np.float64)-data['final_cost']
    terminal=data['done']
    return dict(mae=float(np.abs(error).mean()),rmse=float(np.sqrt(np.mean(error**2))),
        bias=float(error.mean()),predicted_cost_mean=float(np.mean(prediction)),
        actual_cost_mean=float(data['final_cost'].mean()),
        terminal_mae=float(np.abs(error[terminal]).mean()),
        nonterminal_mae=float(np.abs(error[~terminal]).mean()),n_transitions=len(error))
