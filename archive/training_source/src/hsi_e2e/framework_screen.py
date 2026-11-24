"""Bounded, development-only C -> A -> conditional B framework screening.

Historical search modules are reused unchanged. This module never loads the
reserved development audit labels; only the separate runner may do so after
all model/subset choices have been sealed.
"""
from pathlib import Path
import copy
import json
import math
import time
import warnings

import numpy as np
import torch
from torch import nn
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import MultiTaskElasticNet

from . import pool_boundary as boundary
from .component_evaluator import ComponentEvaluator
from .pool_boundary import CapacitySearch
from .variable_k import VariableEnsemble, variable_diverse


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    den = ((y-y.mean(0))**2).sum(0)
    if y.shape != prediction.shape or min(den) <= 1e-12 or not np.isfinite(prediction).all():
        raise ValueError('Invalid multi-target prediction')
    r2 = 1-((y-prediction)**2).sum(0)/den
    return dict(r2=r2.tolist(), mean_r2=float(r2.mean()),
                rmse=np.sqrt(((y-prediction)**2).mean(0)).tolist())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)), encoding='utf-8')
    tmp.replace(path)


def pls_fit(x, y, subset, components=8):
    subset = np.asarray(subset, dtype=int)
    mu, sd = y.mean(0), y.std(0)
    if min(sd) <= 1e-12 or len(set(subset)) != len(subset):
        raise ValueError('Invalid target or subset')
    model = PLSRegression(n_components=components, scale=True, tol=1e-6, max_iter=5000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(x[:, subset], (y-mu)/sd)
    if caught:
        raise RuntimeError('PLS warning: '+str(caught[0].message))
    return dict(subset=subset.tolist(), components=components, coef=model.coef_.copy(),
        xmean=model._x_mean.copy(), xstd=model._x_std.copy(), intercept=model.intercept_.copy(),
        target_mean=mu, target_std=sd)


def pls_predict(model, x):
    selected = x[:, model['subset']]
    return (((selected-model['xmean'])/model['xstd'])@model['coef'].T+
            model['intercept'])*model['target_std']+model['target_mean']


def cars_shared(x, y, xv, yv, seed, *, k=48, paths=32, steps=24, components=8):
    """Declared fixed-K multi-output CARS adaptation, not an exact CARS replica.

    Monte Carlo 80% fit samples, exponential elimination schedule, adaptive
    sampling by mean absolute PLS2 coefficients across scaled target outputs.
    Each path terminates at K. Both importance fits and validation fits count.
    """
    if not components <= k < x.shape[1] or steps < 2:
        raise ValueError('CARS dimension contract')
    rng = np.random.default_rng(seed+810000)
    z = (y-y.mean(0))/y.std(0)
    cache, records, best = {}, [], None
    for path in range(paths):
        selected = np.arange(x.shape[1])
        for step in range(1, steps+1):
            ids = rng.choice(len(x), math.floor(.8*len(x)), replace=False)
            model = PLSRegression(n_components=components, scale=True, tol=1e-6, max_iter=5000)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(x[np.ix_(ids, selected)], z[ids])
            if caught:
                raise RuntimeError('CARS importance PLS warning: '+str(caught[0].message))
            weight = np.abs(model.coef_).mean(0)
            if not np.isfinite(weight).all() or weight.sum() <= 0:
                raise FloatingPointError('Invalid CARS importance')
            target = max(k, round(x.shape[1]*(k/x.shape[1])**(step/steps)))
            probability = np.maximum(weight, 1e-14)
            probability /= probability.sum()
            keep = rng.choice(len(selected), min(target, len(selected)), replace=False, p=probability)
            records.append(dict(kind='importance_fit', path=path, step=step,
                n_samples=len(ids), input_bands=len(selected), output_bands=len(keep)))
            selected = np.sort(selected[keep])
        subset = tuple(selected.tolist())
        if len(subset) != k:
            raise AssertionError('CARS did not produce exact K')
        if subset not in cache:
            fitted = pls_fit(x, y, subset, components)
            score = metrics(yv, pls_predict(fitted, xv))
            cache[subset] = score
            records.append(dict(kind='selection_fit', path=path, subset=list(subset), **score))
        score = cache[subset]
        if best is None or score['mean_r2'] > best['selection']['mean_r2']:
            best = dict(subset=list(subset), selection=score)
    return dict(**best, fits=len(records), importance_fits=paths*steps,
        selection_fits=len(cache), ledger=records,
        adaptation='fixed-K terminal CARS; normalized three-output mean abs coefficients')


def sparse_shared(x, y, xv, yv, *, k=48, alphas=(.01, .03, .1)):
    """Joint L21 elastic-net ranking, fixed-K extraction and PLS2 refit."""
    xm, xs = x.mean(0), x.std(0)
    ym, ys = y.mean(0), y.std(0)
    if min(xs) <= 1e-12 or min(ys) <= 1e-12:
        raise ValueError('Constant training column')
    rows, best = [], None
    for alpha in alphas:
        reg = MultiTaskElasticNet(alpha=alpha, l1_ratio=.5, fit_intercept=True,
            max_iter=50000, tol=1e-4, selection='cyclic')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            reg.fit((x-xm)/xs, (y-ym)/ys)
        if caught:
            raise RuntimeError('Elastic-net convergence warning: '+str(caught[0].message))
        importance = np.sqrt((reg.coef_**2).sum(0))
        subset = sorted(np.argsort(-importance, kind='stable')[:k].tolist())
        fitted = pls_fit(x, y, subset)
        row = dict(alpha=alpha, subset=subset, native_nonzero=int((importance>1e-12).sum()),
            zero_rank_padding=int((importance[subset] <= 1e-12).sum()),
            selection=metrics(yv, pls_predict(fitted, xv)), iterations=int(reg.n_iter_),
            dual_gap=float(np.max(np.asarray(reg.dual_gap_))))
        rows.append(row)
        if best is None or row['selection']['mean_r2'] > best['selection']['mean_r2']:
            best = row
    return dict(**best, fits=2*len(alphas), candidates=rows,
        adaptation='joint coefficient-L2 topK ranking, not native-support-size claim')


def select_mean_lead(actions, predictions, corr, rng, *, count=8, step=0):
    """Same diversity and random draw as legacy; change only guided slots."""
    if len({d for d, _ in actions}) < count:
        raise ValueError('This bounded policy requires K >= batch size')
    selected, thresholds = [], []
    random_order = rng.permutation(len(actions)).tolist()
    slots = ['mean', None, 'mean', None, step % 3, None, 'mean', None]
    for slot in slots[:count]:
        if predictions is None or slot is None:
            order = random_order
        else:
            values = predictions.mean(1) if slot == 'mean' else predictions[:, slot]
            order = np.argsort(-values, kind='stable').tolist()
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
                thresholds.append(threshold)
                break
        if chosen is None:
            raise RuntimeError('Mean-led action batch cannot be completed')
        selected.append(chosen)
    return selected, thresholds


class PolicySearch(CapacitySearch):
    """Versioned scoped policy adapter; no historical source is edited."""
    def one_batch(self):
        if getattr(self, 'proposal_policy', 'legacy') == 'legacy':
            return super().one_batch()
        original = boundary.variable_diverse
        step = len(self.decisions)-getattr(self, 'warm_decisions', 0)
        def adapted(actions, predictions, corr, rng, count=8):
            return select_mean_lead(actions, predictions, corr, rng, count=count, step=step)
        boundary.variable_diverse = adapted
        try:
            return super().one_batch()
        finally:
            boundary.variable_diverse = original

    def summary(self):
        return dict(**super().summary(), proposal_policy=getattr(self, 'proposal_policy', 'legacy'))


class MultiTaskMLP(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(inputs, 64), nn.GELU(), nn.Dropout(.05),
                                    nn.Linear(64, 32), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(32, 1) for _ in range(3)])

    def forward(self, x):
        state = self.shared(x)
        return torch.cat([h(state) for h in self.heads], dim=1)


class SharedSTG(nn.Module):
    """Global stochastic gates; hard fixed-K deployment is separately refitted."""
    def __init__(self, inputs, sigma=.5):
        super().__init__()
        self.mu = nn.Parameter(torch.full((inputs,), .5))
        self.predictor = MultiTaskMLP(inputs)
        self.sigma = sigma

    def forward(self, x):
        z = self.mu
        if self.training:
            z = z+self.sigma*torch.randn_like(z)
        z = torch.clamp(z, 0, 1)
        return self.predictor(x*z)

    def regularizer(self):
        return (.5*(1+torch.erf(self.mu/(self.sigma*math.sqrt(2))))).mean()

    def subset(self, k):
        if not 0 < k <= len(self.mu):
            raise ValueError('Invalid hard K')
        return np.sort(np.argsort(-self.mu.detach().cpu().numpy(), kind='stable')[:k]).tolist()


def nn_predict(model, norm, x, subset):
    model.eval()
    z = ((x[:, subset]-norm['xm'])/norm['xs']).astype('float32')
    with torch.no_grad():
        prediction = model(torch.from_numpy(z)).numpy()
    return prediction*norm['ys']+norm['ym']


def advance_patience(best_score, anchor, last_epoch, epoch, minimum_delta=1e-5):
    """Count cumulative small gains against the last reset, not last checkpoint."""
    if best_score > anchor+minimum_delta:
        return best_score, epoch
    return anchor, last_epoch


def train_nn(x, y, xv, yv, subset, seed, *, gated=False, penalty=0.,
             max_epochs=1200, min_epochs=400, patience=300, fixed_epochs=None, emit=None):
    subset = list(subset)
    norm = dict(xm=x[:, subset].mean(0), xs=x[:, subset].std(0), ym=y.mean(0), ys=y.std(0))
    if min(norm['xs']) <= 1e-12 or min(norm['ys']) <= 1e-12:
        raise ValueError('Constant NN training column')
    xt = torch.from_numpy(((x[:, subset]-norm['xm'])/norm['xs']).astype('float32'))
    yt = torch.from_numpy(((y-norm['ym'])/norm['ys']).astype('float32'))
    torch.manual_seed(seed+720000)
    rng = np.random.default_rng(seed+730000)
    model = SharedSTG(len(subset)) if gated else MultiTaskMLP(len(subset))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    history, best, best_score, last_improvement = [], None, -float('inf'), 0
    patience_anchor = -float('inf')
    updates, min_gate_gradient = 0, float('inf')
    target = max_epochs if fixed_epochs is None else int(fixed_epochs)
    start = time.perf_counter()
    for epoch in range(1, target+1):
        model.train()
        for ids in np.array_split(rng.permutation(len(x)), math.ceil(len(x)/128)):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(xt[ids])-yt[ids])**2).mean()
            if gated:
                loss = loss+penalty*model.regularizer()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite NN loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite NN gradient')
            if gated:
                gn = float(model.mu.grad.norm())
                if gn <= 0:
                    raise RuntimeError('Stochastic gate feedback disconnected')
                min_gate_gradient = min(min_gate_gradient, gn)
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite NN parameter')
            updates += 1
        if fixed_epochs is not None:
            continue
        if epoch % 10 == 0 or epoch == target:
            score = metrics(yv, nn_predict(model, norm, xv, subset))
            row = dict(epoch=epoch, **score)
            if gated:
                row.update(expected_active=float(model.regularizer().detach()*len(subset)),
                           deterministic_active=int((model.mu.detach()>0).sum()))
            history.append(row)
            if score['mean_r2'] > best_score:
                best_score = score['mean_r2']
                best = dict(state=copy.deepcopy(model.state_dict()), epoch=epoch, selection=score)
            patience_anchor, last_improvement = advance_patience(
                best_score, patience_anchor, last_improvement, epoch)
            if emit and epoch % 200 == 0:
                emit(f'NN epoch={epoch} best_inner={best_score:.7f} gated={gated} lambda={penalty}')
            if epoch >= min_epochs and epoch-last_improvement >= patience:
                break
    if fixed_epochs is not None:
        best = dict(state=copy.deepcopy(model.state_dict()), epoch=target, selection=None)
    if best is None:
        raise RuntimeError('No valid NN checkpoint')
    model.load_state_dict(best['state'])
    model.eval()
    if gated:
        best['hard_subset_local'] = model.subset(48 if len(subset) >= 48 else len(subset))
    return dict(model=model, norm=norm, subset=subset, epoch=best['epoch'],
        selection=best['selection'], history=history, epochs_run=epoch, updates=updates,
        min_gate_gradient=None if not gated else min_gate_gradient,
        seconds=time.perf_counter()-start, optimizer=optimizer.state_dict())
