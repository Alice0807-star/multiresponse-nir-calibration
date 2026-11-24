"""Versioned finite-neighbourhood repair; the historical implementation is unchanged."""
import copy
import numpy as np

from .variable_k import (VariableSearch, variable_pool as legacy_pool, variable_diverse, bits)
from .component_evaluator import ComponentEvaluator
from .local_search import swap_subset

POLICY='capacity_capped_pool_v1'


class NeighborhoodExhausted(RuntimeError):
    def __init__(self,available,required):
        self.available=int(available);self.required=int(required)
        super().__init__(f'Only {available} unpaid neighbours; need {required} for the next paid batch')


def unseen_rows(ev,parent):
    parent=tuple(int(x) for x in parent);base=bits(parent);selected=set(parent)
    available=[a for a in range(ev.context.n_bands) if a not in selected]
    return [[a for a in available if ((base^(1<<d))|(1<<a)) not in ev.paid_bits] for d in parent]


def capacity_pool(ev,parent,rng,size=128,minimum=8,events=None):
    """Delegate normal calls exactly; redistribute only when a direction is short.

    At most128 distinct unpaid proposals; if the whole neighbourhood has8..127,
    use all remaining proposals. Fewer than the next paid batch requests a restart.
    """
    parent=tuple(int(x) for x in parent);before=copy.deepcopy(rng.bit_generator.state)
    try:
        return legacy_pool(ev,parent,rng,size=size)
    except RuntimeError as exc:
        if str(exc)!='Unseen parent neighborhood exhausted;do not alter pool silently':raise
        rng.bit_generator.state=before
    rows=unseen_rows(ev,parent);capacities=np.array([len(r) for r in rows],dtype=int)
    total=int(capacities.sum())
    if total<minimum:raise NeighborhoodExhausted(total,minimum)
    target=min(int(size),total);k=len(parent);q,remainder=divmod(size,k)
    extra=set(rng.choice(k,remainder,replace=False).tolist()) if remainder else set()
    allocation=np.minimum(capacities,np.array([q+int(i in extra) for i in range(k)]))
    # Deterministic capped balancing: no new labels or stochastic hyperparameters.
    while int(allocation.sum())<target:
        eligible=np.flatnonzero(allocation<capacities)
        i=min(eligible,key=lambda j:(int(allocation[j]),int(j)))
        allocation[i]+=1
    actions=[]
    for d,row,n in zip(parent,rows,allocation):
        if n:actions.extend((int(d),int(a)) for a in rng.choice(row,int(n),replace=False))
    if len(actions)!=target or len(set(actions))!=target:raise AssertionError('Invalid repaired candidate pool')
    if events is not None:
        events.append(dict(kind='capacity_reallocation',fits=ev.fit_count,parent=list(parent),
            requested=size,returned=target,capacities=capacities.tolist(),allocation=allocation.tolist()))
    return actions


class CapacitySearch(VariableSearch):
    """Same paid scoring/training/acceptance; only finite-pool boundaries differ."""
    def one_batch(self):
        if not hasattr(self,'boundary_events'):self.boundary_events=[]
        if self.pending is None:
            # Exhausted parents consume neither a paid fit nor a training update.
            for attempt in range(64):
                if self.step_in_episode==16:self.restart()
                parent=self.current;remaining=min(8,self.ev.budget-self.ev.fit_count)
                if remaining<=0:raise RuntimeError('Paid budget already complete')
                try:
                    if self.arm=='greedy':
                        actions=[];ps=set(parent);base=bits(parent)
                        available=[a for a in range(self.ev.context.n_bands) if a not in ps]
                        attempts=0
                        while len(actions)<remaining:
                            attempts+=1
                            if attempts>10000:
                                all_actions=[(d,a) for d,row in zip(parent,unseen_rows(self.ev,parent)) for a in row]
                                left=[x for x in all_actions if x not in actions]
                                need=remaining-len(actions)
                                if len(left)<need:raise NeighborhoodExhausted(len(all_actions),remaining)
                                ids=self.rng.choice(len(left),need,replace=False)
                                actions.extend(left[int(i)] for i in ids)
                                self.boundary_events.append(dict(kind='greedy_rejection_fallback',fits=self.ev.fit_count,parent=list(parent),available=len(all_actions)))
                                break
                            d=int(self.rng.choice(parent));a=int(self.rng.choice(available))
                            if (d,a) not in actions and ((base^(1<<d))|(1<<a)) not in self.ev.paid_bits:actions.append((d,a))
                        predictions,thresholds,selected=None,[],list(range(len(actions)))
                    else:
                        actions=capacity_pool(self.ev,parent,self.rng,minimum=remaining,events=self.boundary_events)
                        if self.learner is None:predictions=None
                        else:
                            q,t=self.ev.context.features(parent,self.ev.known[parent],actions)
                            predictions=self.learner.predict(q,t)
                        selected,thresholds=variable_diverse(actions,predictions,self.ev.context.corr,self.rng,count=remaining)
                    break
                except NeighborhoodExhausted as exc:
                    self.boundary_events.append(dict(kind='exhausted_parent_restart',fits=self.ev.fit_count,parent=list(parent),available=exc.available,required=exc.required))
                    self.step_in_episode=16
            else:
                raise RuntimeError('64 existing-rule restarts found no parent with enough unpaid neighbours;stop without duplicate charges or convergence claim')
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

    def summary(self):
        result=super().summary();result['pool_policy']=POLICY
        result['boundary_events']=len(getattr(self,'boundary_events',[]))
        return result


class PaidReplayEvaluator(ComponentEvaluator):
    """Restore completed paid batches, rejecting any request not in that exact log."""
    def evaluate(self,subset,*,parent=None,phase='search',extra=None):
        subset=tuple(int(x) for x in subset)
        if subset in self.known:
            self.cache_hits+=1;return self.known[subset]
        if self.fit_count>=self.replay_target:raise RuntimeError('Replay requested an unpaid future score')
        expected=self.replay_log[self.fit_count]
        if list(subset)!=expected['subset'] or (None if parent is None else list(parent))!=expected['parent'] or phase!=expected['phase']:
            raise RuntimeError('Replay diverged from the paid sequence')
        for key,value in (extra or {}).items():
            if expected[key]!=value:raise RuntimeError('Replay action metadata differs')
        record=copy.deepcopy(self.replay_records[subset])
        self.fit_count+=1;self.known[subset]=record;self.paid_bits.add(bits(subset));self.ledger.append(copy.deepcopy(expected))
        if self.best_subset is None or record['mean_r2']>self.known[self.best_subset]['mean_r2']:self.best_subset=subset
        return record
