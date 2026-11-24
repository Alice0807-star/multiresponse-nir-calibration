"""Budgeted complete-subset search; every regression query has exactly K bands."""
from __future__ import annotations

import json
import time
import numpy as np

from .evaluation import RegressionEvaluator, canonical_subset


class BudgetExhausted(RuntimeError):
    pass


class CompleteEvaluator(RegressionEvaluator):
    def __init__(self, *args, fit_budget, **kwargs):
        super().__init__(*args, **kwargs)
        self.fit_budget = int(fit_budget)

    def evaluate(self, subset):
        key = canonical_subset(subset, self.n_bands)
        if len(key) != self.k:
            raise ValueError("Local-search regressions must use exactly K distinct columns")
        if key not in self._cache and self.fit_count >= self.fit_budget:
            raise BudgetExhausted("Regression fit budget exhausted")
        return super().evaluate(key)


def initial_pool(n_bands, k, seed, size=16):
    rng = np.random.default_rng(seed)
    pool = []
    while len(pool) < size:
        subset = tuple(sorted(rng.choice(n_bands, k, replace=False).tolist()))
        if subset not in pool:
            pool.append(subset)
    return pool


def evaluate_pool(evaluator, pool):
    return [(tuple(s), float(evaluator.evaluate(s))) for s in pool]


def swap_subset(subset, remove, add):
    current = set(subset)
    if remove not in current or add in current:
        raise ValueError("Invalid one-for-one replacement")
    return tuple(sorted((current - {int(remove)}) | {int(add)}))


def random_swap(subset, n_bands, rng):
    remove = int(rng.choice(subset))
    available = np.ones(n_bands, dtype=bool)
    available[list(subset)] = False
    add = int(rng.choice(np.flatnonzero(available)))
    return remove, add, swap_subset(subset, remove, add)


def run_classical(evaluator, pool, *, method, seed, horizon=16, batch_size=8):
    if method not in {"global_random", "random_swap", "greedy_swap8"}:
        raise ValueError(method)
    started = time.perf_counter()
    rng = np.random.default_rng(seed + 100000)
    scored_pool = evaluate_pool(evaluator, pool)
    best_subset, best_error = min(scored_pool, key=lambda pair: pair[1])
    initial_best = best_error
    trace = []
    for i, (subset, score) in enumerate(scored_pool):
        trace.append(dict(phase="initial_pool", query=i + 1, fit_count=i + 1,
            subset_json=json.dumps(subset), validation_nmse=score,
            incumbent_nmse=min(v for _, v in scored_pool[:i + 1]),
            elapsed_seconds=time.perf_counter() - started))
    episode = decision = proposals = accepted = 0
    while evaluator.fit_count < evaluator.fit_budget:
        current, current_error = scored_pool[episode % len(scored_pool)]
        for step in range(horizon):
            if evaluator.fit_count >= evaluator.fit_budget:
                break
            candidates = []
            needed = batch_size if method == "greedy_swap8" else 1
            attempts = 0
            while len(candidates) < needed and evaluator.fit_count < evaluator.fit_budget:
                attempts += 1
                if attempts > 10000:
                    raise RuntimeError("Could not propose an unseen complete subset")
                if method == "global_random":
                    candidate = tuple(sorted(rng.choice(evaluator.n_bands, evaluator.k, replace=False).tolist()))
                    remove = add = None
                else:
                    remove, add, candidate = random_swap(current, evaluator.n_bands, rng)
                if candidate in evaluator._cache:
                    continue
                before = current_error
                score = float(evaluator.evaluate(candidate))
                proposals += 1
                candidates.append((candidate, score, remove, add))
                if score < best_error:
                    best_subset, best_error = candidate, score
                trace.append(dict(phase="search", episode=episode + 1, step=step + 1,
                    decision=decision + 1, proposal_in_decision=len(candidates),
                    current_subset_json=json.dumps(current), subset_json=json.dumps(candidate),
                    removed_1based=None if remove is None else remove + 1,
                    added_1based=None if add is None else add + 1,
                    validation_nmse=score, before_nmse=before, reward=before - score,
                    incumbent_nmse=best_error, fit_count=evaluator.fit_count,
                    elapsed_seconds=time.perf_counter() - started))
            if not candidates:
                break
            chosen = min(candidates, key=lambda item: item[1]) if method == "greedy_swap8" else candidates[0]
            if method != "greedy_swap8" or chosen[1] < current_error:
                current, current_error = chosen[:2]
                accepted += 1
            decision += 1
        episode += 1
    return dict(best_subset=list(best_subset), best_validation_nmse=best_error,
        initial_best_nmse=initial_best, search_seconds=time.perf_counter() - started,
        trace=trace, evaluator_counters=evaluator.snapshot(), decisions=decision,
        proposed_complete_subsets=proposals, accepted_moves=accepted,
        episodes=episode, horizon=horizon)
