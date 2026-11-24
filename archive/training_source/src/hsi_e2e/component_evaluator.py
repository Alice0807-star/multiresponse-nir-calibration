"""PLS-component matched evaluator; A4 preserves the historical evaluator exactly."""
import time
import warnings
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from .evaluation import canonical_subset
from .variable_k import VariableEvaluator,bits


class ComponentEvaluator(VariableEvaluator):
    """A4 delegates exactly; other A change only the paid PLS fit/context."""
    def __init__(self, *args, k, components):
        super().__init__(*args, k=k)
        if components not in (2, 4, 6, 8) or components > k:
            raise ValueError('Unregistered PLS components')
        self.components = int(components)
        if components != 4:
            import hashlib
            self.context_hash = hashlib.sha256(
                (self.context_hash + f'|selection_PLS{components}').encode()).hexdigest()

    def evaluate(self, subset, *, parent=None, phase='search', extra=None):
        if self.components == 4:
            return super().evaluate(subset, parent=parent, phase=phase, extra=extra)
        s = canonical_subset(subset, self.context.n_bands)
        if len(s) != self.k:
            raise ValueError('Wrong complete K')
        if s in self.known:
            self.cache_hits += 1
            return self.known[s]
        if self.fit_count >= self.budget:
            raise RuntimeError('Paid fit budget exhausted')
        tick = time.perf_counter()
        reg = PLSRegression(n_components=self.components, scale=True, max_iter=5000, tol=1e-6)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            reg.fit(self.context.X[:, s], self.context.y)
            pred = reg.predict(self.X_reward[:, s])
            fitpred = reg.predict(self.context.X[:, s])
        self.fit_count += 1
        self.fit_seconds += time.perf_counter() - tick
        if caught:
            raise RuntimeError(f'PLSR warning;A={self.components};fit={self.fit_count};{caught[0].message}')
        coef = np.asarray(reg.coef_)
        if coef.shape != (3, self.k):
            raise ValueError('PLSR coefficient convention')
        residual = self.context.y - fitpred
        cov = self.context.target_corr - self.context.corr[:, s] @ (
            coef.T * np.sqrt((len(residual) - 1) / len(residual)))
        if self.fit_count <= 8:
            if not np.allclose(cov, self.context.xz.T @ residual / len(residual), atol=1e-10, rtol=1e-8):
                raise AssertionError('Parent covariance units')
            check = ((self.context.X[:, s] - reg._x_mean) / reg._x_std) @ coef.T + reg.intercept_
            if not np.allclose(check, fitpred, atol=1e-10, rtol=1e-8):
                raise AssertionError('Parent coefficient units')
        r2 = 1. - ((self.y_reward_z - pred) ** 2).sum(0) / self.denominator
        err = (self.y_reward_z - pred) * self.context.std
        row = dict(r2=r2, mean_r2=float(r2.mean()), nmse=float(((self.y_reward_z - pred) ** 2).mean()),
            rmse=np.sqrt((err ** 2).mean(0)), mae=abs(err).mean(0), coef=coef.copy(), residual_cov=cov,
            fit_error=(residual ** 2).mean(0), xmean=reg._x_mean.copy(), xstd=reg._x_std.copy(),
            intercept=reg.intercept_.copy())
        if any(not np.isfinite(v).all() for v in row.values()):
            raise FloatingPointError('Nonfinite paid PLS context')
        self.known[s] = row
        self.paid_bits.add(bits(s))
        if self.best_subset is None or row['mean_r2'] > self.known[self.best_subset]['mean_r2']:
            self.best_subset = s
        self.ledger.append(dict(actual_fits=self.fit_count, subset=list(s),
            parent=None if parent is None else list(parent), phase=phase, r2=r2.tolist(),
            mean_r2=row['mean_r2'], nmse=row['nmse'],
            incumbent_mean_r2=self.known[self.best_subset]['mean_r2'], **(extra or {})))
        return row
