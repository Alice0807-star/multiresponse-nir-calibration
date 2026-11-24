"""PLSR numerical repair: same tolerance, allow up to5000 NIPALS iterations.

Only increases the iteration cap. Any warning still stops the experiment.
Old entries remain immutable. Restored metrics reuse already-paid scores.
"""
import time,warnings
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from .evaluation import canonical_subset
from stability_experiment import LoggedEvaluator

class ConvergentLoggedEvaluator(LoggedEvaluator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._restored_metrics={}
    def evaluate(self,subset):
        key=canonical_subset(subset,self.n_bands)
        if key in self._cache:return super().evaluate(key)
        if len(key)!=self.k or self.fit_count>=self.fit_budget:raise ValueError('Invalid/budget-exhausted evaluation')
        self.total_calls+=1;start=time.perf_counter()
        model=PLSRegression(n_components=self.n_components,scale=True,max_iter=5000,tol=1e-6)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(self.X_fit[:,key],self.y_fit_z);prediction=model.predict(self.X_reward[:,key])
        self.fit_count+=1;self.fit_seconds+=time.perf_counter()-start
        if caught:
            details=';'.join(type(w.message).__name__+':'+str(w.message) for w in caught)
            raise RuntimeError(f'PLSR5000 warning at{key}: {details}')
        value=float(((self.y_reward_z-prediction)**2).mean())
        if not np.isfinite(value):raise FloatingPointError('Nonfinite PLSR')
        self._cache[key]=value;self._prediction_cache[key]=prediction.copy()
        total=(self.ledger[-1]['total_actual_fits'] if self.ledger else 0)+1
        import json
        self.ledger.append(dict(query=len(self.ledger)+1,scope=self.scope,actual_fit_increment=1,scope_fit_count=self.fit_count,
            total_actual_fits=total,subset_json=json.dumps(key),validation_nmse=value))
        return value
    def reward_metrics(self,subset):
        key=canonical_subset(subset,self.n_bands)
        if key in self._restored_metrics:return self._restored_metrics[key]
        return super().reward_metrics(key)

def restore_evaluators(ev,pev,rows):
    import json
    seen={'selection':set(),'probe':set()};paid=0
    for row in rows:
        target=ev if row['scope']=='selection' else pev;s=tuple(row['subset']);cache=seen[target.scope];inc=int(s not in cache);cache.add(s);paid+=inc
        target._cache[s]=row['nmse'];target._restored_metrics[s]=dict(validation_nmse=row['nmse'],mean_r2=row['mean_r2'],r2=np.array(row['r2']))
        target.total_calls+=1;target.cache_hits+=1-inc;target.fit_count+=inc
        ev.ledger.append(dict(query=len(ev.ledger)+1,scope=target.scope,actual_fit_increment=inc,scope_fit_count=target.fit_count,
            total_actual_fits=paid,subset_json=json.dumps(s),validation_nmse=row['nmse']))
        if paid!=row['actual_fits']:raise AssertionError('Restored fee prefix mismatch')
