"""One fixed12/24/12 allocation probe. Operator starts scientific training."""
from pathlib import Path
from datetime import datetime, timezone
from types import MethodType
import argparse
import json
import math
import signal
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run_panel_reranking_check as history
from empirical_budget_core import GenericPLSEvaluator
from hsi_e2e.pool_boundary import CapacitySearch, PaidReplayEvaluator
from hsi_e2e.framework_screen import metrics, advance_patience
from hsi_e2e.formal_runtime import read, write, sha, torch_atomic
from panel_reranking_core import strict_contrast

OUT = ROOT / 'results/ph_quota_12_24_12_001'
READINESS = ROOT / 'results/ph_quota_readiness_001'
DOC = ROOT / 'docs/pH配额单因素检查_协议与终端说明_2026-11-19.md'
PROPOSAL = ROOT / 'results/ph_selection_stability_review_001/next_step_proposal.json'
SPLITS = (7117, 8111, 9103); SEEDS = (17, 29, 43)
QUOTA = (12, 24, 12); BUDGET = 4096; COMPONENTS = 8; CAP = 48
CANDIDATE = 'quota_ph24'; REFERENCES = ('original', 'covsel', 'full')
METHODS = (CANDIDATE, *REFERENCES); WIDTHS = (24, 23)
BASE_PIDS = {'original': 'U3', 'covsel': 'COV', 'full': 'FULL'}
base = history.base
matched = history.previous
duration = history.duration


def cells(): return [(s, seed) for s in SPLITS for seed in SEEDS]
def profile(t): return history.profile(t)
def cached_choice(s, seed, method, t): return history.choice(s, seed, BASE_PIDS[method], t)
def folder(s, seed, t): return OUT / 'models' / str(s) / str(seed) / str(t)
def search_folder(s, seed, t): return OUT / 'search' / str(s) / str(seed) / str(t)
def stop_check():
    if base.STOP: raise InterruptedError('Operator stop; no automatic continuation')


def sources():
    return list(dict.fromkeys(history.sources() + [Path(__file__), ROOT / 'tests/test_ph_quota_check.py']))


def reference_rows():
    rows = [r for r in read(history.OUT / 'metrics.json') if r['method'] in REFERENCES]
    expected = {(s, seed, m) for s, seed in cells() for m in REFERENCES}
    if len(rows) != len(expected) or {(r['split'], r['seed'], r['method']) for r in rows} != expected:
        raise ValueError('Incomplete unique reference grid')
    return rows


def estimate_minutes(refs):
    search_seconds = selection_low = selection_high = refit_low = refit_high = 0.
    for s, seed in cells():
        for t in range(3):
            old_search = read(history.ledger_paths(s, seed)[t].parent / 'summary.json')
            search_seconds += old_search['seconds'] * (QUOTA[t] / 16)
            a = cached_choice(s, seed, 'original', t); cfg = profile(t)
            rate = a['seconds'] / a['epochs_run']
            selection_low += rate * cfg['min_epochs']; selection_high += rate * cfg['max_epochs']
            row = next(r for r in refs if (r['split'], r['seed'], r['method']) == (s, seed, 'original'))
            entry = read(row['refit_entries'][t]); rate = entry['seconds'] / entry['epoch']
            refit_low += rate * 10; refit_high += rate * cfg['max_epochs']
    return dict(search=search_seconds / 60, selection_low=selection_low / 60,
        selection_high=selection_high / 60, refit_high=refit_high / 60,
        total_low=(search_seconds + selection_low + refit_low) / 60,
        total_high=(search_seconds + selection_high + refit_high) / 60,
        scope='Extrapolated from frozen historical rates; selected refit epochs unknown; not guaranteed wall clock')


def lock():
    p = read(OUT / 'protocol.json')
    for key in ('source_sha256', 'input_sha256'):
        for path, digest in p[key].items():
            if sha(path) != digest: raise RuntimeError('Locked artifact changed: ' + path)
    if p['versions'] != duration.versions(): raise RuntimeError('Environment changed after preparation')
    if (p['quota'], p['query_budget_per_target'], p['components']) != (list(QUOTA), BUDGET, COMPONENTS):
        raise RuntimeError('Prepared configuration differs from running entry')
    history.lock(); history.seal()
    return p


def prepare():
    if OUT.exists(): raise RuntimeError('Output exists; no overwrite or automatic restart')
    history.lock(); history.seal()
    if read(history.OUT / 'status.json')['state'] != 'complete_verified':
        raise RuntimeError('Verified previous experiment required')
    for path, digest in read(history.OUT / 'delivery_manifest.json')['files'].items():
        if sha(path) != digest: raise RuntimeError('Historical result changed: ' + path)
    ready = read(READINESS / 'verification.json')
    if not ready['passed']: raise RuntimeError('Entry tests must pass')
    for path, digest in ready['source_sha256'].items():
        if sha(path) != digest: raise RuntimeError('Source changed after tests: ' + path)
    if read(PROPOSAL)['proposed_quota'] != list(QUOTA): raise ValueError('Accepted proposal mismatch')
    refs = reference_rows()
    inputs = [READINESS / 'verification.json', DOC, PROPOSAL,
              history.OUT / 'protocol.json', history.OUT / 'delivery_manifest.json',
              history.OUT / 'metrics.json', history.OUT / 'summary.json']
    for s in SPLITS:
        arrays = duration.selection_arrays(s)
        inputs += [matched.prior.OLD / str(s) / n for n in ('selection_data.npz', 'audit_data.npz')]
        for row in [r for r in refs if r['split'] == s]:
            matched.validate_selection(*arrays, row['subset'])
            if row['method'] != 'full' and len(row['subset']) > CAP: raise ValueError('Reference input cap')
            inputs.append(Path(row['prediction_file']))
            for t in range(3):
                a = cached_choice(s, row['seed'], row['method'], t)
                e = read(row['refit_entries'][t])
                if a['subset'] != row['subset'] or a['widths'] != list(WIDTHS):
                    raise ValueError('Cached input or network mismatch')
                if e['target'] != t or e['epoch'] != a['epoch'] or e['subset'] != a['subset'] or Path(e['selected_folder']) != Path(a['folder']):
                    raise ValueError('Cached refit identity mismatch')
                inputs += [Path(a['folder']) / n for n in ('selected.json', 'selected_model.pt', 'history.json')]
                inputs += [Path(row['refit_entries'][t]), Path(e['model_path'])]
    estimate = estimate_minutes(refs)
    OUT.mkdir(parents=True)
    write(OUT / 'references.json', refs)
    p = dict(created_utc=datetime.now(timezone.utc).isoformat(), purpose='One fixed per-target quota vector check',
        partitions=list(SPLITS), seeds=list(SEEDS), quota=list(QUOTA), target_order=['SSC', 'pH', 'firmness'],
        input_cap=CAP, components=COMPONENTS, query_budget_per_target=BUDGET,
        query_budget_per_cell=BUDGET * 3, new_search_fits=len(cells()) * 3 * BUDGET,
        new_nn_selections=len(cells()) * 3, new_nn_refits=len(cells()) * 3,
        cached_reference_selections=len(refs) * 3, cached_reference_refits=len(refs) * 3,
        pipeline_cases=len(cells()) * len(METHODS), methods=list(METHODS), widths=list(WIDTHS),
        profiles={str(t): profile(t) for t in range(3)},
        selection='Unchanged greedy CapacitySearch,seed unchanged for each target,PLS8single-responseR2,4096unique paid queries each;strict best final endpoints;union<=48;no final-panel reranking',
        predictor='Three independent MLP24/23,each reads entire union;unchanged frozen AdamWlr.001/decay.001,batch128,dropout.05,clip5,eval10',
        seal='All27new searches,27newNN choices,and81cached choices globally sealed before current audit loader',
        evaluation='Fixed chosen-epoch refit on fit+reward,then existing336historical audit;all27reference cases byte-reused',
        gate='Candidate versus original andCovSel:mean>0,all3partitionmeans>0,>=6/9wins,alltargetmeans>=0;full also reported with identical rule;descriptive only',
        primary_references=['original', 'covsel'], full_precision_reference='full',
        fairness='Same input upper cap and selection-query cap versus original union,not guaranteed same actual union size,time or total compute;CovSel is a different cheaper selector',
        stopping='Any warning,nonfinite,integrity error,operator stop or exception stops;one quota only;no retries,quotasweep,reweighting,extraepochs,seeds or automatic promotion',
        interpretation='Historically adapted overlapping development;not independent validation or optimal quota identification;failure does not rule out all allocations',
        runtime='CPU actualTorch4/BLAS1/interop1', versions=duration.versions(), estimate_minutes=estimate,
        source_sha256={str(f): sha(f) for f in sources()},
        input_sha256={str(f): sha(f) for f in dict.fromkeys(inputs)})
    write(OUT / 'protocol.json', p)
    write(OUT / 'status.json', dict(state='prepared_not_started', audit_opened=False, new_models_trained=0))
    lock(); print(dict(prepared_only=True, new_search_fits=p['new_search_fits'], new_nn_selections=p['new_nn_selections'],
                       new_nn_refits=p['new_nn_refits'], estimate_minutes=estimate), flush=True)


def emit(message):
    line = datetime.now(timezone.utc).isoformat(timespec='seconds') + ' ' + message
    print(line, flush=True)
    with (OUT / 'run.log').open('a', encoding='utf-8') as f: f.write(line + '\n')


def search(x, y, xv, yv, seed, k, output):
    ev = GenericPLSEvaluator(x, y, xv, yv, BUDGET, k=k, components=COMPONENTS)
    state = CapacitySearch(ev, seed, 'greedy'); start = time.perf_counter()
    try:
        stop_check(); state.initialize()
        while ev.fit_count < BUDGET:
            stop_check(); state.one_batch()
            if ev.fit_count % 1024 == 0: emit(f'SEARCH seed={seed} K={k} fits={ev.fit_count}/{BUDGET}')
        if ev.fit_count != BUDGET or ev.cache_hits: raise AssertionError('Paid query accounting')
        return state
    finally:
        state.elapsed_seconds = time.perf_counter() - start
        write(output / 'ledger.json', ev.ledger); write(output / 'decisions.json', state.decisions)
        write(output / 'boundary_events.json', getattr(state, 'boundary_events', []))
        write(output / 'summary.json', state.summary() if ev.best_subset is not None else
              dict(actual_fits=ev.fit_count, cache_hits=ev.cache_hits, incomplete=True))


def search_all():
    panels = {}
    for s in SPLITS:
        x, y, xv, yv = duration.selection_arrays(s)
        for seed in SEEDS:
            own = []
            for t, k in enumerate(QUOTA):
                emit(f'START SEARCH split={s} seed={seed} target={t} K={k}')
                state = search(x, y[:, t:t+1], xv, yv[:, t:t+1], seed, k, search_folder(s, seed, t))
                own.append(list(state.ev.best_subset))
            union = sorted(set().union(*map(set, own)))
            if not max(QUOTA) <= len(union) <= CAP: raise AssertionError('Union cardinality')
            panels.setdefault(str(s), {})[str(seed)] = dict(individual=own, union=union, quota=list(QUOTA))
            write(OUT / 'panels.json', panels)
    return panels


class Progress:
    def __enter__(self):
        self.original = base.metrics; self.n = 0; self.best = -float('inf'); base.metrics = self.score
        return self
    def __exit__(self, *args): base.metrics = self.original
    def score(self, y, prediction):
        result = self.original(y, prediction); self.n += 1; self.best = max(self.best, result['mean_r2'])
        if self.n % 20 == 0: emit(f'EPOCH {self.n * 10} best_reward={self.best:.7f}')
        return result


def select_all(panels):
    count = 0
    for s, seed in cells():
        x, y, xv, yv = duration.selection_arrays(s); subset = panels[str(s)][str(seed)]['union']
        for t in range(3):
            stop_check(); emit(f'SELECT split={s} seed={seed} target={t} columns={len(subset)}')
            with Progress(): a = base.train(x, y[:, t:t+1], xv, yv[:, t:t+1], subset, seed, WIDTHS, **profile(t))
            f = folder(s, seed, t); f.mkdir(parents=True, exist_ok=True)
            torch_atomic(f / 'selected_model.pt', dict(state=a['model'].state_dict(), norm=a['norm'], subset=subset))
            write(f / 'history.json', a['history'])
            write(f / 'selected.json', dict(split=s, seed=seed, method=CANDIDATE, target=t, subset=subset,
                widths=list(WIDTHS), parameters=sum(p.numel() for p in a['model'].parameters()),
                **{k: a[k] for k in ('epoch', 'epochs_run', 'updates', 'seconds', 'selection')}))
            count += 1
            write(OUT / 'status.json', dict(state='selecting', audit_opened=False, new_models_trained=count))
    seal(create=True); lock(); emit('ALL SEARCH ENDPOINTS AND NN CHOICES SEALED;current audit arrays unopened')


def choice_files():
    paths = [OUT / 'references.json', OUT / 'panels.json']
    for s, seed in cells():
        for t in range(3):
            paths += [search_folder(s, seed, t) / n for n in ('ledger.json', 'decisions.json', 'summary.json', 'boundary_events.json')]
            paths += [folder(s, seed, t) / n for n in ('selected.json', 'selected_model.pt', 'history.json')]
    for row in read(OUT / 'references.json'):
        for t in range(3):
            a = cached_choice(row['split'], row['seed'], row['method'], t)
            paths += [Path(a['folder']) / n for n in ('selected.json', 'selected_model.pt', 'history.json')]
    return list(dict.fromkeys(paths))


def seal(create=False):
    values = {str(f): sha(f) for f in choice_files()}; marker = OUT / 'choices_locked.json'
    if marker.exists():
        if read(marker)['files'] != values: raise RuntimeError('Sealed choices changed')
    elif create: write(marker, dict(utc=datetime.now(timezone.utc).isoformat(), files=values))
    else: raise RuntimeError('Global choices not sealed')


def load_audit(s):
    if not (OUT / 'choices_locked.json').exists(): raise RuntimeError('Audit blocked before global seal')
    seal(); lock()
    with np.load(matched.prior.OLD / str(s) / 'audit_data.npz', allow_pickle=False) as z:
        return tuple(z[n].copy() for n in ('X_audit', 'y_audit', 'audit_ids'))


def evaluate_all(panels):
    seal(); lock(); rows = []
    write(OUT / 'status.json', dict(state='refitting_and_evaluating', audit_opened=True, new_models_trained=len(cells()) * 3))
    refs = read(OUT / 'references.json')
    for s in SPLITS:
        xf, yf, xr, yr = duration.selection_arrays(s); x, y = np.vstack([xf, xr]), np.vstack([yf, yr])
        xa, ya, ids = load_audit(s)
        with np.load(matched.prior.OLD / str(s) / 'selection_data.npz', allow_pickle=False) as z:
            matched.prior.pilot.prior.assert_partition(z['fit_ids'], z['reward_ids'], ids)
        for seed in SEEDS:
            subset = panels[str(s)][str(seed)]['union']; preds = []; entries = []
            for t in range(3):
                stop_check(); a = read(folder(s, seed, t) / 'selected.json')
                emit(f'REFIT split={s} seed={seed} target={t} epochs={a["epoch"]}')
                trained = base.train(x, y[:, t:t+1], None, None, subset, seed, WIDTHS, fixed_epochs=a['epoch'])
                f = OUT / 'refits' / str(s) / str(seed) / str(t); f.mkdir(parents=True, exist_ok=True); path = f / 'model.pt'
                torch_atomic(path, dict(state=trained['model'].state_dict(), norm=trained['norm'], subset=subset))
                write(f / 'entry.json', dict(split=s, seed=seed, method=CANDIDATE, target=t, subset=subset,
                    epoch=a['epoch'], updates=trained['updates'], seconds=trained['seconds'], model_path=str(path), selected_folder=str(folder(s, seed, t))))
                preds.append(history.predict_saved(path, xa)); entries.append(str(f / 'entry.json'))
            prediction = np.hstack(preds); score = metrics(ya, prediction); score['mae'] = abs(ya - prediction).mean(0).tolist()
            f = OUT / 'predictions' / str(s) / f'{seed}.npz'; f.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, prediction=prediction, target=ya, ids=ids)
            rows.append(dict(split=s, seed=seed, method=CANDIDATE, subset=subset, audit=score, prediction_file=str(f), refit_entries=entries))
            emit(f'CASE split={s} seed={seed} method={CANDIDATE} meanR2={score["mean_r2"]:.7f}')
            rows += [r for r in refs if (r['split'], r['seed']) == (s, seed)]
    write(OUT / 'metrics.json', rows); write(OUT / 'summary.json', summarize(rows))
    write(OUT / 'costs.json', dict(new_search_pls_fits=len(cells()) * 3 * BUDGET,
        verification_pls_fits=0, new_nn_selections=len(cells()) * 3, new_nn_refits=len(cells()) * 3,
        cached_reference_selections=len(refs) * 3, cached_reference_refits=len(refs) * 3,
        scope='Logical and actual new costs reported separately; historical reference query costs retained in source records',
        strictly_equal_total_compute=False))
    write(OUT / 'complete.json', dict(files={n: sha(OUT / n) for n in ('protocol.json', 'metrics.json', 'summary.json', 'costs.json', 'choices_locked.json')}))


def summarize(rows):
    expected = {(s, seed, m) for s, seed in cells() for m in METHODS}
    if len(rows) != len(expected) or {(r['split'], r['seed'], r['method']) for r in rows} != expected:
        raise ValueError('Incomplete unique experiment grid')
    methods = {}
    for m in METHODS:
        group = [r for r in rows if r['method'] == m]
        methods[m] = dict(mean_r2=float(np.mean([r['audit']['mean_r2'] for r in group])),
            r2=np.mean([r['audit']['r2'] for r in group], axis=0).tolist(),
            rmse=np.mean([r['audit']['rmse'] for r in group], axis=0).tolist(),
            partition_means={str(s): float(np.mean([r['audit']['mean_r2'] for r in group if r['split'] == s])) for s in SPLITS})
    comparisons = {m: strict_contrast(rows, CANDIDATE, m, SPLITS, SEEDS) for m in REFERENCES}
    return dict(methods=methods, candidate_minus_reference=comparisons,
        primary_support=all(comparisons[m]['descriptive_support'] for m in ('original', 'covsel')),
        all_reference_descriptive_support=all(v['descriptive_support'] for v in comparisons.values()),
        independent_validation=False, configuration_promoted=False, global_convergence_claimed=False)


def validate_search(ledger, k):
    if len(ledger) != BUDGET: raise AssertionError('Exact search budget required')
    seen = set(); best = -float('inf'); winner = None
    for i, row in enumerate(ledger, 1):
        subset = tuple(row['subset'])
        if row['actual_fits'] != i or len(subset) != k or tuple(sorted(set(subset))) != subset or subset in seen:
            raise AssertionError('Unique canonical paid subsets required')
        if not np.isfinite(row['mean_r2']) or row['r2'] != [row['mean_r2']]: raise AssertionError('Single-target finite R2 required')
        if row['mean_r2'] > best: best, winner = row['mean_r2'], list(subset)
        if row['incumbent_mean_r2'] != best: raise AssertionError('Best paid prefix mismatch')
        seen.add(subset)
    return winner


def replay_search(x, y, xv, yv, seed, k, path):
    ledger = read(path / 'ledger.json'); expected = validate_search(ledger, k)
    ev = GenericPLSEvaluator(x, y, xv, yv, BUDGET, k=k, components=COMPONENTS)
    ev.replay_log = ledger; ev.replay_target = BUDGET
    # The paid ledger stores R2 and normalized MSE, not unsaved error vectors.
    # Greedy replay uses only the recorded target R2 and mean acceptance score.
    ev.replay_records = {tuple(r['subset']): dict(r2=np.array(r['r2']), nmse=r['nmse'], mean_r2=r['mean_r2']) for r in ledger}
    ev.evaluate = MethodType(PaidReplayEvaluator.evaluate, ev)
    state = CapacitySearch(ev, seed, 'greedy'); state.initialize()
    while ev.fit_count < BUDGET: stop_check(); state.one_batch()
    old = read(path / 'summary.json')
    canonical = lambda v: json.dumps(v, sort_keys=True, allow_nan=False)
    if canonical(state.decisions) != canonical(read(path / 'decisions.json')) or canonical(getattr(state, 'boundary_events', [])) != canonical(read(path / 'boundary_events.json')):
        raise AssertionError('Search-path replay mismatch')
    if list(ev.best_subset) != expected or old['best']['subset'] != expected or old['cache_hits'] or ev.cache_hits or old['context_sha256'] != ev.context_hash:
        raise AssertionError('Search endpoint/accounting/context mismatch')
    return expected


def verify():
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    lock(); seal()
    for n, digest in read(OUT / 'complete.json')['files'].items():
        if sha(OUT / n) != digest: raise AssertionError('Completed output changed')
    panels = read(OUT / 'panels.json'); rows = read(OUT / 'metrics.json'); refs = reference_rows()
    if refs != read(OUT / 'references.json'): raise AssertionError('Cached reference rows changed')
    for ref in refs:
        actual = next(r for r in rows if (r['split'], r['seed'], r['method']) == (ref['split'], ref['seed'], ref['method']))
        if actual != ref: raise AssertionError('Reference predictions/metrics were not reused exactly')
    count = 0; max_error = 0.; searches = 0
    for s in SPLITS:
        xf, yf, xr, yr = duration.selection_arrays(s); x, y = np.vstack([xf, xr]), np.vstack([yf, yr]); xa, ya, ids = load_audit(s)
        for seed in SEEDS:
            row = panels[str(s)][str(seed)]
            own = [replay_search(xf, yf[:, t:t+1], xr, yr[:, t:t+1], seed, QUOTA[t], search_folder(s, seed, t)) for t in range(3)]
            searches += 3
            if row != dict(individual=own, union=sorted(set().union(*map(set, own))), quota=list(QUOTA)) or len(row['union']) > CAP:
                raise AssertionError('Quota provenance or union mismatch')
            for method in METHODS:
                actual = next(r for r in rows if (r['split'], r['seed'], r['method']) == (s, seed, method)); outputs = []
                for t in range(3):
                    f = folder(s, seed, t) if method == CANDIDATE else Path(cached_choice(s, seed, method, t)['folder'])
                    a = read(f / 'selected.json') if method == CANDIDATE else cached_choice(s, seed, method, t)
                    cfg = profile(t); h = read(f / 'history.json'); subset = actual['subset']
                    if a['subset'] != subset or a['widths'] != list(WIDTHS): raise AssertionError('Selection input/width mismatch')
                    if not cfg['min_epochs'] <= a['epochs_run'] <= cfg['max_epochs'] or a['updates'] != math.ceil(len(xf) / 128) * a['epochs_run']:
                        raise AssertionError('Selection training duration/update mismatch')
                    if [v['epoch'] for v in h] != list(range(10, a['epochs_run'] + 1, 10)) or a['epoch'] != max(h, key=lambda v: v['mean_r2'])['epoch']:
                        raise AssertionError('Best reward checkpoint/history mismatch')
                    anchor, last, best = -float('inf'), 0, -float('inf')
                    for point in h:
                        best = max(best, point['mean_r2']); anchor, last = advance_patience(best, anchor, last, point['epoch'])
                    if a['epochs_run'] != cfg['max_epochs'] and a['epochs_run'] - last < cfg['patience']: raise AssertionError('Patience mismatch')
                    saved = torch.load(f / 'selected_model.pt', map_location='cpu', weights_only=False)
                    norm = dict(xm=xf[:, subset].mean(0), xs=xf[:, subset].std(0), ym=yf[:, t:t+1].mean(0), ys=yf[:, t:t+1].std(0))
                    if saved['subset'] != subset or any(not np.array_equal(saved['norm'][k], v) for k, v in norm.items()): raise AssertionError('Fit-only scales mismatch')
                    reward = r2_score(yr[:, t:t+1], history.predict_saved(f / 'selected_model.pt', xr))
                    if abs(reward - a['selection']['mean_r2']) > 1e-13: raise AssertionError('Selected reward reload mismatch')
                    e = read(actual['refit_entries'][t])
                    if (e['split'], e['seed'], e['target'], e['epoch'], e['subset']) != (s, seed, t, a['epoch'], subset): raise AssertionError('Refit identity mismatch')
                    if Path(e['selected_folder']) != f or e['updates'] != math.ceil(len(x) / 128) * a['epoch']: raise AssertionError('Refit selected-folder/update mismatch')
                    saved = torch.load(e['model_path'], map_location='cpu', weights_only=False)
                    norm = dict(xm=x[:, subset].mean(0), xs=x[:, subset].std(0), ym=y[:, t:t+1].mean(0), ys=y[:, t:t+1].std(0))
                    if saved['subset'] != subset or any(not np.array_equal(saved['norm'][k], v) for k, v in norm.items()): raise AssertionError('Fit+reward refit scales mismatch')
                    outputs.append(history.predict_saved(e['model_path'], xa)); count += 1
                prediction = np.hstack(outputs)
                with np.load(actual['prediction_file'], allow_pickle=False) as z:
                    if not np.array_equal(z['ids'], ids) or not np.array_equal(z['target'], ya): raise AssertionError('Audit sample IDs/targets mismatch')
                    error = float(np.max(abs(prediction - z['prediction']))); max_error = max(error, max_error)
                    if error: raise AssertionError('Checkpoint prediction reload mismatch')
                for key, value in [('r2', r2_score(ya, prediction, multioutput='raw_values')),
                                   ('rmse', np.sqrt(mean_squared_error(ya, prediction, multioutput='raw_values'))),
                                   ('mae', mean_absolute_error(ya, prediction, multioutput='raw_values'))]:
                    if not np.allclose(value, actual['audit'][key], atol=1e-12, rtol=0): raise AssertionError('Independent metric mismatch')
                if abs(float(np.mean(actual['audit']['r2'])) - actual['audit']['mean_r2']) > 1e-13: raise AssertionError('Mean R2 mismatch')
    if summarize(rows) != read(OUT / 'summary.json'): raise AssertionError('Summary mismatch')
    cost = read(OUT / 'costs.json')
    if (cost['new_search_pls_fits'], cost['new_nn_selections'], cost['new_nn_refits']) != (searches * BUDGET, searches, searches):
        raise AssertionError('New cost accounting mismatch')
    lock(); seal()
    proof = dict(passed=True, pipeline_cases=len(rows), models=count, replayed_searches=searches,
        paid_search_fits=searches * BUDGET, verification_pls_fits=0, max_prediction_reload_error=max_error,
        cached_reference_cases=len(refs), independent_validation=False, configuration_promoted=False)
    write(OUT / 'verification.json', proof); return proof


def report(proof):
    summary = read(OUT / 'summary.json')
    labels = {CANDIDATE: '配额12/24/12并集', 'original': '原16/16/16并集', 'covsel': 'CovSel48', 'full': '全谱641'}
    lines = ['# pH配额单因素检查结果', '', '历史开发比较；只改变来源配额，所有目标仍读取完整并集。不是独立来源验证。', '',
             '|方案|平均R²|SSC|pH|硬度|', '|---|---:|---:|---:|---:|']
    for m, v in summary['methods'].items(): lines.append('|' + labels[m] + '|' + '|'.join(f'{x:.6f}' for x in [v['mean_r2'], *v['r2']]) + '|')
    lines += ['', '|候选减参照|平均差|SSC差|pH差|硬度差|胜出单元|描述性条件|', '|---|---:|---:|---:|---:|---:|---|']
    for m, v in summary['candidate_minus_reference'].items():
        lines.append('|' + labels[m] + '|' + '|'.join(f'{x:+.6f}' for x in [v['mean_delta'], *v['target_delta']])
                     + f'|{v["wins"]}/{len(cells())}|' + ('满足' if v['descriptive_support'] else '未满足') + '|')
    lines += ['', f'核验{proof["models"]}个选择模型和全部流程，预测重载误差{proof["max_prediction_reload_error"]}。'
              f'精确重放{proof["replayed_searches"]}条付费搜索，核验阶段不新增PLS拟合。',
              '保留全部单元和逐目标代价。一个配额方案的成败不证明最佳配额或结构上限。触顶不证明充分收敛。',
              '不自动继续扫配额、改权重、加训练或开启独立来源实验，不自动晋级主方法。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def configure_runtime():
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)


def run():
    lock()
    if read(OUT / 'status.json')['state'] != 'prepared_not_started': raise RuntimeError('Already started; no automatic retry')
    with (OUT / 'operator_start.txt').open('x', encoding='utf-8') as f: f.write(datetime.now(timezone.utc).isoformat() + '\n')
    old_signal = signal.getsignal(signal.SIGINT)
    def interrupt(*args):
        base.STOP = True; raise InterruptedError('Operator stopped this run')
    try:
        signal.signal(signal.SIGINT, interrupt); base.STOP = False; configure_runtime()
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter('error'); environment = matched.version.pin_and_record()
            if environment['actual_interop_threads'] != 1: raise RuntimeError('Interop1 required')
            environment['python_executable'] = sys.executable; write(OUT / 'environment.json', environment)
            write(OUT / 'status.json', dict(state='searching', audit_opened=False, new_models_trained=0))
            panels = search_all(); matched.version.pin_and_record()
            select_all(panels); matched.version.pin_and_record()
            evaluate_all(panels); matched.version.pin_and_record(); proof = verify(); report(proof)
        write(OUT / 'status.json', dict(state='complete_verified', audit_opened=True, new_models_trained=len(cells()) * 3))
        emit('COMPLETE quota12/24/12 verified;no quota sweep,new-source experiment or parameter changes started')
        write(OUT / 'delivery_manifest.json', dict(files={str(f): sha(f) for f in OUT.rglob('*') if f.is_file() and f.name != 'delivery_manifest.json'}))
    except BaseException as error:
        state = read(OUT / 'status.json')
        write(OUT / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        write(OUT / 'status.json', dict(state='stopped_on_problem', audit_opened=state.get('audit_opened', False)))
        emit('STOPPED;retain files for diagnosis;no automatic retries or changes'); raise
    finally: signal.signal(signal.SIGINT, old_signal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'status']); args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'run': run()
    else: print(read(OUT / 'status.json'))


if __name__ == '__main__': main()
