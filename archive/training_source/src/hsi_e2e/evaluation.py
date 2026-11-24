"""Fit/reward evaluation. This module never receives test data during search."""
from __future__ import annotations

import hashlib
import time
import warnings

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def canonical_subset(subset, n_bands):
    raw = tuple(subset)
    if any(isinstance(v, (bool, np.bool_)) or int(v) != v for v in raw):
        raise ValueError("Band indices must be integers")
    key = tuple(sorted(int(v) for v in raw))
    if len(key) != len(set(key)):
        raise ValueError("Repeated band in subset")
    if any(v < 0 or v >= n_bands for v in key):
        raise ValueError("Band index out of range")
    return key


def target_scale(y):
    mean = np.mean(y, axis=0)
    std = np.std(y, axis=0, ddof=0)
    if np.any(~np.isfinite(std)) or np.any(std <= 1e-12):
        raise ValueError("Constant or near-constant target; explicit treatment required")
    return mean, std


class RegressionEvaluator:
    """An instance-local cache scoped to exact fit/reward arrays and PLS settings."""

    def __init__(self, X_fit, y_fit, X_reward, y_reward, *, n_components=4, k=16,
                 retain_predictions=False, retain_fit_hints=False):
        arrays = [np.asarray(a, dtype=np.float64) for a in
                  (X_fit, y_fit, X_reward, y_reward)]
        if any(a.ndim != 2 or not np.isfinite(a).all() for a in arrays):
            raise ValueError("Expected finite two-dimensional input arrays")
        self.X_fit, self.y_fit, self.X_reward, self.y_reward = arrays
        if (len(self.X_fit) != len(self.y_fit) or
                len(self.X_reward) != len(self.y_reward) or
                self.X_fit.shape[1] != self.X_reward.shape[1] or
                self.y_fit.shape[1] != self.y_reward.shape[1]):
            raise ValueError("Mismatching sample, feature or target dimensions")
        self.n_bands = self.X_fit.shape[1]
        self.n_components, self.k = int(n_components), int(k)
        self.y_mean, self.y_std = target_scale(self.y_fit)
        self.y_fit_z = (self.y_fit - self.y_mean) / self.y_std
        self.y_reward_z = (self.y_reward - self.y_mean) / self.y_std
        digest = hashlib.sha256()
        for a in arrays:
            digest.update(str(a.shape).encode())
            digest.update(np.ascontiguousarray(a).tobytes())
        digest.update(repr((self.n_components, self.k, "NMSE", "ddof0", "PLS_scale_true")).encode())
        self.cache_context_sha256 = digest.hexdigest()
        self._cache = {}
        self._prediction_cache = {} if retain_predictions else None
        self._fit_hint_cache = {} if retain_fit_hints else None
        if retain_fit_hints:
            self._X_fit_std = np.maximum(self.X_fit.std(axis=0, ddof=0), 1e-12)
            self._X_fit_z = (self.X_fit - self.X_fit.mean(axis=0)) / self._X_fit_std
        self.total_calls = self.cache_hits = self.fit_count = self.empty_evaluations = 0
        self.fit_seconds = 0.0
        self.warning_messages = []

    def evaluate(self, subset):
        self.total_calls += 1
        key = canonical_subset(subset, self.n_bands)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        if not key:
            prediction = np.zeros_like(self.y_reward_z)
            self.empty_evaluations += 1
        else:
            start = time.perf_counter()
            model = PLSRegression(n_components=min(self.n_components, len(key)),
                                  scale=True, max_iter=500)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(self.X_fit[:, key], self.y_fit_z)
                prediction = model.predict(self.X_reward[:, key])
                if self._fit_hint_cache is not None:
                    coefficients = np.asarray(model.coef_)
                    if coefficients.shape != (self.y_fit.shape[1], len(key)):
                        coefficients = coefficients.T
                    if coefficients.shape != (self.y_fit.shape[1], len(key)):
                        raise ValueError("Unexpected PLSR coefficient orientation")
                    self._fit_hint_cache[key] = {
                        "fit_residual": self.y_fit_z - model.predict(self.X_fit[:, key]),
                        "coefficient_importance": np.sqrt(np.mean(
                            (coefficients * self._X_fit_std[list(key)][None, :]) ** 2, axis=0)),
                    }
            self.fit_count += 1
            self.fit_seconds += time.perf_counter() - start
            self.warning_messages.extend(str(w.message) for w in caught)
        score = float(np.mean((self.y_reward_z - prediction) ** 2))
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite reward score")
        self._cache[key] = score
        if self._prediction_cache is not None:
            self._prediction_cache[key] = np.array(prediction, copy=True)
        return score

    def reward_metrics(self, subset):
        """Describe an already queried selection score, without another model fit.

        These metrics concern the repeatedly used reward-validation data and
        must never be labelled as independent evaluation or test performance.
        """
        if self._prediction_cache is None:
            raise RuntimeError("Construct with retain_predictions=True to obtain reward metrics")
        key = canonical_subset(subset, self.n_bands)
        if key not in self._cache:
            raise KeyError("Subset has not been evaluated")
        prediction = self._prediction_cache[key] * self.y_std + self.y_mean
        mse = mean_squared_error(self.y_reward, prediction, multioutput="raw_values")
        r2 = r2_score(self.y_reward, prediction, multioutput="raw_values")
        mae = mean_absolute_error(self.y_reward, prediction, multioutput="raw_values")
        return dict(validation_nmse=self._cache[key], r2=r2, rmse=np.sqrt(mse), mae=mae,
                    mean_r2=float(np.mean(r2)), prediction=prediction)

    def conditional_hints(self, subset):
        """Training-only residual/contribution proxies; never refit a model here."""
        key = canonical_subset(subset, self.n_bands)
        if self._fit_hint_cache is None or key not in self._fit_hint_cache:
            raise KeyError("A fitted nonempty subset with retain_fit_hints=True is required")
        cached = self._fit_hint_cache[key]
        if "add_hint_raw" not in cached:
            residual = cached["fit_residual"]
            centered = residual - residual.mean(axis=0)
            std = centered.std(axis=0, ddof=0)
            residual_z = centered / np.where(std > 1e-12, std, 1.)
            residual_z[:, std <= 1e-12] = 0.
            cached["add_hint_raw"] = np.abs(self._X_fit_z.T @ residual_z / len(residual_z)).mean(axis=1)
        drop_importance = np.zeros(self.n_bands, dtype=np.float64)
        drop_importance[list(key)] = cached["coefficient_importance"]
        return cached["add_hint_raw"].copy(), drop_importance

    def snapshot(self):
        complete = sum(len(key) == self.k for key in self._cache)
        return dict(total_calls=self.total_calls, cache_hits=self.cache_hits,
                    cache_misses=len(self._cache), fit_count=self.fit_count,
                    real_fits=self.fit_count, empty_evaluations=self.empty_evaluations,
                    complete_unique_subsets=complete, fit_seconds=self.fit_seconds,
                    warning_count=len(self.warning_messages),
                    cache_context_sha256=self.cache_context_sha256)


def fit_and_score(X_train, y_train, X_test, y_test, subset, *, n_components=4):
    """Final scoring entry point, invoked only after writing the selection lock."""
    subset = canonical_subset(subset, X_train.shape[1])
    if not subset:
        raise ValueError("Final scoring requires a nonempty subset")
    mean, std = target_scale(y_train)
    model = PLSRegression(n_components=min(n_components, len(subset)), scale=True,
                          max_iter=500)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train[:, subset], (y_train - mean) / std)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    prediction = model.predict(X_test[:, subset]) * std + mean
    predict_seconds = time.perf_counter() - start
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite final predictions")
    r2 = r2_score(y_test, prediction, multioutput="raw_values")
    mse = mean_squared_error(y_test, prediction, multioutput="raw_values")
    mae = mean_absolute_error(y_test, prediction, multioutput="raw_values")
    return dict(prediction=prediction, r2=r2, rmse=np.sqrt(mse), mae=mae,
                mean_r2=float(np.mean(r2)), nmse=float(np.mean(mse / std ** 2)),
                final_fit_seconds=fit_seconds, predict_seconds=predict_seconds,
                warning_messages=[str(w.message) for w in caught],
                y_mean=mean, y_std=std)
