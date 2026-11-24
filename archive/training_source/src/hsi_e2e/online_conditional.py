"""Prospective, fully costed conditional gain feedback and local search.

The predictor sees only fit data and an already paid parent score. Candidate
scores are acquired after the whole proposal batch has been locked. Historical
experiments and their evaluators are deliberately untouched.
"""
from collections import deque
import copy
import hashlib
import time
import warnings

import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression

from .conditional_feedback import ConditionalGainModel, actual_gain_feedback
from .evaluation import canonical_subset, target_scale
from .local_search import initial_pool, swap_subset


class FitContext:
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float64)
        self.mean, self.std = target_scale(y)
        self.y = (y - self.mean) / self.std
        self.xmean = self.X.mean(0)
        self.xstd = self.X.std(0)
        if np.any(self.xstd <= 1e-12):
            raise ValueError('Constant spectral column')
        self.xz = (self.X - self.xmean) / self.xstd
        self.corr = self.xz.T @ self.xz / len(X)
        self.target_corr = self.xz.T @ self.y / len(X)
        self.n_bands = X.shape[1]

    def features(self, parent, paid, actions):
        """Identical tensors for fitting and ranking; no child label argument."""
        parent = tuple(parent)
        coef, cov = paid['coef'], paid['residual_cov']
        corr, tc = self.corr, self.target_corr
        queries, tokens = [], []
        for d, a in actions:
            child = swap_subset(parent, d, a)
            if len(child) != 16:
                raise ValueError('K contract')
            survivors = [j for j in parent if j != d]
            di = parent.index(d)
            ci = [parent.index(j) for j in survivors]
            ac, dc = corr[a, survivors], corr[d, survivors]
            other = abs(corr[np.ix_(survivors, survivors)]).copy()
            np.fill_diagonal(other, 0.)
            other = other.sum(1) / 14
            q, t = [], []
            for task in range(3):
                q.append([1-paid['r2'][task], paid['fit_error'][task], cov[a, task], cov[d, task],
                    tc[a, task], tc[d, task], coef[task, di], abs(coef[task]).mean(), corr[a, d],
                    abs(ac).max(), abs(ac).mean(), 2*a/(self.n_bands-1)-1,
                    2*d/(self.n_bands-1)-1, tc[survivors, task].mean()])
                t.append(np.column_stack([ac, dc, tc[survivors, task], cov[survivors, task],
                    coef[task, ci], abs(coef[task, ci]), 2*np.array(survivors)/(self.n_bands-1)-1, other]))
            queries.append(q)
            tokens.append(t)
        q, t = np.asarray(queries, dtype=np.float32), np.asarray(tokens, dtype=np.float32)
        if not np.isfinite(q).all() or not np.isfinite(t).all():
            raise FloatingPointError('Nonfinite fit-only features')
        return q, t


class PaidEvaluator:
    """Every new complete subset incurs exactly one PLSR fit, context retained."""
    def __init__(self, X, y, X_reward, y_reward, budget):
        for a in [X, y, X_reward, y_reward]:
            if a.ndim != 2 or not np.isfinite(a).all():
                raise ValueError('Invalid evaluation arrays')
        if y.shape[1] != 3 or X.shape[1] != X_reward.shape[1]:
            raise ValueError('Expected aligned spectra and three targets')
        self.context = FitContext(X, y)
        self.X_reward = np.asarray(X_reward, dtype=np.float64)
        self.y_reward = np.asarray(y_reward, dtype=np.float64)
        self.y_reward_z = (y_reward-self.context.mean)/self.context.std
        self.denominator = ((self.y_reward_z-self.y_reward_z.mean(0))**2).sum(0)
        if np.any(self.denominator <= 1e-12):
            raise ValueError('Constant reward target')
        self.budget, self.fit_count, self.cache_hits = int(budget), 0, 0
        self.fit_seconds = 0.
        self.known, self.ledger = {}, []
        self.best_subset = None
        digest = hashlib.sha256()
        for a in [X, y, X_reward, y_reward]:
            digest.update(str(a.shape).encode())
            digest.update(np.ascontiguousarray(a).tobytes())
        digest.update(b'K16_PLS4_scaleTrue_max5000_tol1e-6_R2_unclipped')
        self.context_hash = digest.hexdigest()

    def evaluate(self, subset, *, parent=None, phase='search', extra=None):
        s = canonical_subset(subset, self.context.n_bands)
        if len(s) != 16:
            raise ValueError('Only complete K=16 fits are allowed')
        if s in self.known:
            self.cache_hits += 1
            return self.known[s]
        if self.fit_count >= self.budget:
            raise RuntimeError('Paid fit budget exhausted')
        tick = time.perf_counter()
        reg = PLSRegression(n_components=4, scale=True, max_iter=5000, tol=1e-6)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            reg.fit(self.context.X[:, s], self.context.y)
            pred = reg.predict(self.X_reward[:, s])
            fitpred = reg.predict(self.context.X[:, s])
        self.fit_count += 1
        self.fit_seconds += time.perf_counter()-tick
        if caught:
            raise RuntimeError(f'PLSR warning; paid fit={self.fit_count}; subset={s}; '+str(caught[0].message))
        coef = np.asarray(reg.coef_)
        if coef.shape != (3, 16):
            raise ValueError('PLSR coefficient convention changed')
        residual = self.context.y-fitpred
        # sklearn 1.4 coefficients multiply sample-standardized X (ddof=1).
        # This exact covariance identity avoids a second fit or large residual cache.
        cov = self.context.target_corr-self.context.corr[:, s] @ (coef.T*np.sqrt((len(residual)-1)/len(residual)))
        if self.fit_count <= 8:
            direct = self.context.xz.T @ residual/len(residual)
            if not np.allclose(cov, direct, atol=1e-10, rtol=1e-8):
                raise AssertionError('Standardized coefficient/covariance reconstruction')
            check = ((self.context.X[:, s]-reg._x_mean)/reg._x_std) @ coef.T+reg.intercept_
            if not np.allclose(check, fitpred, atol=1e-10, rtol=1e-8):
                raise AssertionError('PLSR prediction convention')
        r2 = 1.-((self.y_reward_z-pred)**2).sum(0)/self.denominator
        err = (self.y_reward_z-pred)*self.context.std
        record = dict(r2=r2, mean_r2=float(r2.mean()), nmse=float(((self.y_reward_z-pred)**2).mean()),
            rmse=np.sqrt((err**2).mean(0)), mae=abs(err).mean(0),
            coef=coef.copy(), residual_cov=cov, fit_error=(residual**2).mean(0),
            xmean=reg._x_mean.copy(), xstd=reg._x_std.copy(), intercept=reg.intercept_.copy())
        if any(not np.isfinite(v).all() for v in record.values()):
            raise FloatingPointError('Nonfinite paid model/context')
        self.known[s] = record
        if self.best_subset is None or record['mean_r2'] > self.known[self.best_subset]['mean_r2']:
            self.best_subset = s
        self.ledger.append(dict(actual_fits=self.fit_count, subset=list(s), parent=None if parent is None else list(parent),
            phase=phase, r2=r2.tolist(), mean_r2=record['mean_r2'], nmse=record['nmse'],
            incumbent_mean_r2=self.known[self.best_subset]['mean_r2'], **(extra or {})))
        return record

    def best(self):
        r = self.known[self.best_subset]
        return dict(subset=list(self.best_subset), **{k: r[k] for k in ['mean_r2', 'r2', 'rmse', 'mae', 'nmse']})


def candidate_pool(ev, parent, rng, per_delete=8):
    actions = []
    available = np.array([i for i in range(ev.context.n_bands) if i not in parent])
    for d in parent:
        unseen = [int(a) for a in available if swap_subset(parent, d, int(a)) not in ev.known]
        if len(unseen) < per_delete:
            raise RuntimeError('Parent neighborhood exhausted; do not silently alter proposal distribution')
        actions.extend((int(d), int(a)) for a in rng.choice(unseen, per_delete, replace=False))
    return actions


def select_diverse(actions, predictions, corr, rng, count=8):
    """Shared redundancy control; the predictor only changes four priorities.

    One predicted candidate per task and one mean-gain candidate, interleaved
    with four uniform probes. This changes proposal coverage, never acceptance
    or the reward. Both learned controls use identical rules.
    """
    selected, relax = [], []
    random_order = rng.permutation(len(actions)).tolist()
    slots = [0, None, 1, None, 2, None, 'mean', None]
    for slot in slots[:count]:
        if predictions is None or slot is None:
            order = random_order
        else:
            scores = predictions.mean(1) if slot == 'mean' else predictions[:, slot]
            order = np.argsort(-scores, kind='stable').tolist()
        chosen = None
        for threshold in [.95, .99, 1.0000001]:
            for i in order:
                d, a = actions[i]
                if any(d == actions[j][0] for j in selected):
                    continue
                if any(abs(corr[a, actions[j][1]]) >= threshold for j in selected):
                    continue
                chosen = i
                break
            if chosen is not None:
                relax.append(threshold)
                break
        if chosen is None:
            raise RuntimeError('Diverse action batch cannot be completed')
        selected.append(chosen)
    return selected, relax


class FeedbackEnsemble:
    def __init__(self, warm_groups, *, adaptive, seeds=(13, 37, 73)):
        q = np.concatenate([g['q'] for g in warm_groups])
        t = np.concatenate([g['t'] for g in warm_groups])
        self.norm = dict(qmu=q.mean((0, 1), keepdims=True), qsd=np.maximum(q.std((0, 1), keepdims=True), 1e-6),
            tmu=t.mean((0, 1, 2), keepdims=True), tsd=np.maximum(t.std((0, 1, 2), keepdims=True), 1e-6))
        self.models, self.optimizers, self.initial_attention = [], [], []
        self.rng = np.random.default_rng(624000)
        self.groups = deque(maxlen=512)
        self.updates, self.adaptive, self.last_loss = 0, adaptive, None
        self.min_attention_gradient = float('inf') if adaptive else 0.
        self.max_gradient = 0.
        for g in warm_groups:
            self.add(g)
        for seed in seeds:
            torch.manual_seed(seed)
            model = ConditionalGainModel(adaptive=adaptive)
            self.models.append(model)
            self.initial_attention.append(copy.deepcopy(model.attention.state_dict()))
            self.optimizers.append(torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001))

    def tensors(self, q, t):
        return (torch.from_numpy((q-self.norm['qmu'])/self.norm['qsd']),
                torch.from_numpy((t-self.norm['tmu'])/self.norm['tsd']))

    def add(self, group):
        # Keep all three targets until the prediction's final mean aggregation.
        self.groups.append({k: np.array(group[k], copy=True) for k in ['q', 't', 'gain']})

    def predict(self, q, t):
        q, t = self.tensors(q, t)
        with torch.no_grad():
            predictions = torch.stack([m(q, t) for m in self.models]).mean(0).numpy()
        if not np.isfinite(predictions).all():
            raise FloatingPointError('Nonfinite ensemble ranking')
        return predictions

    def update(self):
        ids = self.rng.integers(len(self.groups), size=16)
        samples = []
        for i in ids:
            group = self.groups[i]
            j = self.rng.choice(len(group['gain']), 8, replace=len(group['gain']) < 8)
            samples.append({k: v[j] for k, v in group.items()})
        q, t = self.tensors(np.concatenate([g['q'] for g in samples]), np.concatenate([g['t'] for g in samples]))
        y = torch.from_numpy(np.concatenate([g['gain'] for g in samples]).astype('float32'))
        losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            optimizer.zero_grad(set_to_none=True)
            loss = actual_gain_feedback(model(q, t), y)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError('Nonfinite feedback gradient')
            ag = sum(float(p.grad.detach().double().square().sum()) for p in model.attention.parameters() if p.grad is not None)**.5
            if self.adaptive and ag <= 0:
                raise RuntimeError('Attention feedback disconnected')
            self.min_attention_gradient = min(self.min_attention_gradient, ag)
            gn = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.))
            self.max_gradient = max(self.max_gradient, gn)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite feedback parameters')
            losses.append(float(loss.detach()))
        self.last_loss = float(np.mean(losses))
        self.updates += 1

    def audit(self):
        changed = [any(not torch.equal(v, model.attention.state_dict()[k]) for k, v in initial.items())
                   for model, initial in zip(self.models, self.initial_attention)]
        if self.adaptive != all(changed) or (not self.adaptive and any(changed)):
            raise AssertionError('Frozen/adaptive attention control corrupted')
        return dict(updates=self.updates, models=len(self.models), attention_changed=changed,
            min_attention_gradient=self.min_attention_gradient, max_gradient=self.max_gradient, last_loss=self.last_loss)


class OnlineSearch:
    def __init__(self, evaluator, seed, arm, *, learner=None, warm_groups=None):
        if arm not in ['warm', 'adaptive', 'frozen', 'diverse', 'greedy']:
            raise ValueError(arm)
        self.ev, self.seed, self.arm, self.learner = evaluator, seed, arm, learner
        self.rng = np.random.default_rng(seed+100000)
        self.restart_rng = np.random.default_rng(seed+910000)
        self.roots = initial_pool(evaluator.context.n_bands, 16, seed)
        self.episode, self.step_in_episode = 0, 16
        self.current = None
        self.decisions, self.restarts = [], []
        self.warm_groups = [] if warm_groups is None else warm_groups
        self.elapsed_seconds = 0.
        self.pending = None

    def initialize(self):
        for s in self.roots:
            self.ev.evaluate(s, phase='initial')

    def restart(self):
        coin = self.restart_rng.random()
        rank = None
        if coin < .5:
            elite = sorted(self.ev.known, key=lambda s: (-self.ev.known[s]['mean_r2'], s))[:16]
            rank = int(self.restart_rng.integers(len(elite)))
            self.current = elite[rank]
        else:
            self.current = self.roots[self.episode % 16]
        self.restarts.append(dict(episode=self.episode, coin=float(coin), rank=rank, parent=list(self.current), actual_fits=self.ev.fit_count))
        self.episode += 1
        self.step_in_episode = 0

    def one_batch(self):
        if self.pending is None:
            if self.step_in_episode == 16:
                self.restart()
            parent = self.current
            if self.arm == 'greedy':
                actions = []
                while len(actions) < 8:
                    d = int(self.rng.choice(parent))
                    available = [a for a in range(self.ev.context.n_bands) if a not in parent]
                    a = int(self.rng.choice(available))
                    if (d, a) not in actions and swap_subset(parent, d, a) not in self.ev.known:
                        actions.append((d, a))
                predictions, thresholds, selected = None, [], list(range(8))
            else:
                actions = candidate_pool(self.ev, parent, self.rng)
                q, t = self.ev.context.features(parent, self.ev.known[parent], actions)
                predictions = None if self.learner is None else self.learner.predict(q, t)
                selected, thresholds = select_diverse(actions, predictions, self.ev.context.corr, self.rng)
            remaining = min(8, self.ev.budget-self.ev.fit_count)
            # Lock all decisions before scoring any child. Unpaid predictions are
            # diagnostic artifacts, never used as teacher labels.
            self.pending = dict(parent=parent, actions=[actions[i] for i in selected[:remaining]],
                prediction=None if predictions is None else predictions[selected[:remaining]].tolist(),
                thresholds=thresholds[:remaining], cursor=0, children=[])
        p = self.pending
        parent = p['parent']
        for d, a in p['actions'][p['cursor']:]:
            child = swap_subset(parent, d, a)
            self.ev.evaluate(child, parent=parent, phase='warm' if self.arm == 'warm' else 'search',
                extra=dict(remove=d, add=a, decision=len(self.decisions)))
            p['children'].append(child)
            p['cursor'] += 1
        children = p['children']
        q, t = self.ev.context.features(parent, self.ev.known[parent], p['actions'])
        gain = np.array([self.ev.known[s]['r2']-self.ev.known[parent]['r2'] for s in children], dtype=np.float32)
        group = dict(q=q, t=t, gain=gain)
        if self.arm == 'warm':
            self.warm_groups.append(group)
        elif self.learner is not None:
            self.learner.add(group)
            self.learner.update()
        child = max(children, key=lambda s: self.ev.known[s]['mean_r2'])
        take = self.ev.known[child]['mean_r2'] > self.ev.known[parent]['mean_r2']
        self.decisions.append(dict(parent=list(parent), children=[list(s) for s in children], actions=p['actions'],
            prediction=p['prediction'], gain=gain.tolist(), thresholds=p['thresholds'], accepted=take,
            selected=list(child), actual_fits=self.ev.fit_count, parent_mean_r2=self.ev.known[parent]['mean_r2']))
        if take:
            self.current = child
        self.step_in_episode += 1
        self.pending = None

    def summary(self):
        return dict(seed=self.seed, arm=self.arm, actual_fits=self.ev.fit_count, cache_hits=self.ev.cache_hits,
            best=self.ev.best(), seconds=self.elapsed_seconds, fit_seconds=self.ev.fit_seconds,
            decisions=len(self.decisions), learner=None if self.learner is None else self.learner.audit(),
            context_sha256=self.ev.context_hash)
