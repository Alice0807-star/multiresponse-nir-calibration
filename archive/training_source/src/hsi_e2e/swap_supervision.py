"""Direct one-swap labels, training-only action features and guided proposals."""
from __future__ import annotations
import json,time
import numpy as np
import torch
from torch.nn import functional as F
from .local_search import swap_subset,evaluate_pool

FEATURE_NAMES=['current_nmse','fit_residual_nmse','drop_importance','mean_parent_importance',
    'max_parent_importance','add_std','remove_std','add_remove_correlation','add_remove_abs_correlation',
    'add_remove_squared_correlation','max_add_remaining_abs_corr','mean_add_remaining_abs_corr',
    'mean_add_remaining_squared_corr','drop_squared_importance','add_residual_energy',
    'remove_residual_energy','add_target_energy','remove_target_energy','analytic_proxy']
FEATURE_NAMES += [f'{kind}_{target}' for kind in ['add_residual_cov','remove_residual_cov','add_target_abs_corr','remove_target_abs_corr'] for target in ['SSC','pH','firmness']]


class ActionFeatures:
    """No child score or reward-validation target enters these features.

    Only current NMSE (already observed by the old RL state) uses reward data.
    Residuals and contributions reuse the paid full K-column parent PLSR fit.
    """
    def __init__(self,X_fit,y_fit):
        self.x_std=np.maximum(np.std(X_fit,axis=0,ddof=0),1e-12)
        self.xz=(X_fit-X_fit.mean(0))/self.x_std
        self.yz=(y_fit-y_fit.mean(0))/y_fit.std(0,ddof=0)
        self.correlation=self.xz.T@self.xz/len(X_fit)
        self.target_correlation=self.xz.T@self.yz/len(X_fit)

    def context(self,evaluator,parent):
        key=tuple(sorted(parent))
        if key not in evaluator._fit_hint_cache:
            raise ValueError('Already paid complete-parent fit with retained hints required')
        hints=evaluator._fit_hint_cache[key]
        residual=hints['fit_residual']
        return dict(parent=key,error=float(evaluator._cache[key]),
            fit_error=float(np.mean(residual**2)),residual_cov=self.xz.T@residual/len(residual),
            importance=np.asarray(hints['coefficient_importance']))

    def matrix(self,context,remove,adds):
        parent=context['parent']
        if remove not in parent or any(int(a) in parent for a in adds):
            raise ValueError('Invalid complete swap')
        adds=np.asarray(adds,dtype=int)
        remaining=[a for a in parent if a!=remove]
        index=parent.index(remove)
        drop=context['importance'][index]
        cov=context['residual_cov']
        ac=cov[adds]
        rc=cov[remove]
        tc=self.target_correlation
        corr=self.correlation[adds,remove]
        other=self.correlation[np.ix_(adds,remaining)]
        add_energy=np.mean(ac**2,axis=1)
        remove_energy=float(np.mean(rc**2))
        proxy=add_energy-drop**2*(1-corr**2)
        scalar=lambda x:np.full(len(adds),x)
        columns=[scalar(context['error']),scalar(context['fit_error']),scalar(drop),
            scalar(context['importance'].mean()),scalar(context['importance'].max()),
            self.x_std[adds],scalar(self.x_std[remove]),corr,np.abs(corr),corr**2,
            np.abs(other).max(1),np.abs(other).mean(1),np.mean(other**2,1),scalar(drop**2),
            add_energy,scalar(remove_energy),np.mean(tc[adds]**2,1),
            scalar(np.mean(tc[remove]**2)),proxy]
        columns.extend(ac[:,i] for i in range(3))
        columns.extend(scalar(rc[i]) for i in range(3))
        columns.extend(np.abs(tc[adds,i]) for i in range(3))
        columns.extend(scalar(abs(tc[remove,i])) for i in range(3))
        result=np.column_stack(columns).astype(np.float64)
        if result.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(result).all():
            raise AssertionError('Invalid action features')
        return result


def candidate_states(parent,error,remove,n_bands=641):
    state=np.zeros(4*n_bands+3,np.float32)
    state[list(parent)]=1.
    state[n_bands+remove]=1.
    state[-3:]=[error,1.,1/16]
    return state


def predict_scores(model,mode,features,state,adds):
    if mode=='analytic_proxy':
        return features[:,FEATURE_NAMES.index('analytic_proxy')]
    if mode in ['ridge','extra_trees']:
        return model.predict(features)
    with torch.no_grad():
        raw=model(torch.from_numpy(state)[None]).squeeze(0)
        if mode=='mlp_cost':
            result=state[-3]-F.softplus(raw)
        else:
            result=raw
    return result.numpy()[np.asarray(adds)]


def bootstrap_lower(values,seed,replicates=2000):
    values=np.asarray(values,np.float64)
    rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),size=(replicates,len(values)))].mean(1)
    return float(np.quantile(means,.025))


def run_guided(evaluator,pool,*,seed,mode,model,feature_builder,horizon=16,batch_size=8):
    """Same accept-best-of-eight rule as greedy_swap8; ranked proposals only.

    Full 16x625 neighborhood is accessible. Cached or previously paid proposals
    are skipped; a lack of positive predictions never forces zero progress.
    """
    started=time.perf_counter()
    scored=evaluate_pool(evaluator,pool)
    best,best_error=min(scored,key=lambda item:item[1])
    trace=[dict(phase='initial_pool',subset_json=json.dumps(s),validation_nmse=e,
        fit_count=i+1,incumbent_nmse=min(x[1] for x in scored[:i+1])) for i,(s,e) in enumerate(scored)]
    rankings={}
    cursors={}
    episode=decisions=accepted=0
    while evaluator.fit_count<evaluator.fit_budget:
        current,error=scored[episode%len(scored)]
        for step in range(horizon):
            if evaluator.fit_count>=evaluator.fit_budget:
                break
            if current not in rankings:
                ctx=feature_builder.context(evaluator,current)
                adds=np.array([i for i in range(evaluator.n_bands) if i not in current])
                candidates=[]
                for remove in current:
                    f=feature_builder.matrix(ctx,remove,adds)
                    s=candidate_states(current,error,remove,evaluator.n_bands)
                    scores=predict_scores(model,mode,f,s,adds)
                    candidates.extend((float(v),int(remove),int(add)) for v,add in zip(scores,adds))
                # Stable order resolves equal predictions by remove/add index.
                candidates.sort(key=lambda item:(-item[0],item[1],item[2]))
                rankings[current]=candidates
                cursors[current]=0
            candidates=rankings[current]
            proposals=[]
            while len(proposals)<batch_size and evaluator.fit_count<evaluator.fit_budget:
                if cursors[current]>=len(candidates):
                    break
                prediction,remove,add=candidates[cursors[current]]
                cursors[current]+=1
                child=swap_subset(current,remove,add)
                if child in evaluator._cache:
                    continue
                score=float(evaluator.evaluate(child))
                if score<best_error:
                    best,best_error=child,score
                proposals.append((child,score))
                trace.append(dict(phase='search',episode=episode+1,step=step+1,decision=decisions+1,
                    current_subset_json=json.dumps(current),subset_json=json.dumps(child),
                    removed=remove,added=add,before_nmse=error,validation_nmse=score,
                    predicted_gain=prediction,true_gain=error-score,fit_count=evaluator.fit_count,
                    incumbent_nmse=best_error))
            if proposals:
                chosen=min(proposals,key=lambda item:item[1])
                if chosen[1]<error:
                    current,error=chosen
                    accepted+=1
            decisions+=1
        episode+=1
        if episode>10000:
            raise RuntimeError('Guided search exhausted useful neighborhoods')
    return dict(best_subset=list(best),best_validation_nmse=best_error,
        initial_best_nmse=min(e for _,e in scored),trace=trace,search_seconds=time.perf_counter()-started,
        decisions=decisions,accepted_moves=accepted,episodes=episode,
        evaluator_counters=evaluator.snapshot())
