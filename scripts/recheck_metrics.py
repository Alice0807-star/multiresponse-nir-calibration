"""Recalculate V5 archived-prediction metrics; never search, fit or use network.

Reference responses must first be imported locally from the official data files.
The public prediction files contain IDs and model predictions, not reference data.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

FIELDS = ('r2', 'mean_r2', 'rmse', 'mae', 'normalized_rmse')
ARMS = ('A', 'B', 'C', 'D_low', 'D_wide')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def score(y, prediction, training_sd):
    if y.shape != prediction.shape or y.ndim != 2 or y.shape[1] != 4:
        raise ValueError('Prediction/reference shape mismatch')
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('Non-finite reference or prediction')
    if training_sd.shape != y.shape or np.any(training_sd <= 0) or not np.isfinite(training_sd).all():
        raise ValueError('Invalid training SD')
    error = prediction-y
    denominator = np.square(y-y.mean(axis=0)).sum(axis=0)
    if np.any(denominator <= 1e-12):
        raise ValueError('Constant/degenerate evaluation response')
    r2 = 1-np.square(error).sum(axis=0)/denominator
    return dict(r2=r2.tolist(), mean_r2=float(r2.mean()),
                rmse=np.sqrt(np.square(error).mean(axis=0)).tolist(),
                mae=np.abs(error).mean(axis=0).tolist(),
                normalized_rmse=np.sqrt(np.square(error/training_sd).mean(axis=0)).tolist())


def average(values):
    if not values:
        raise ValueError('Cannot average an empty workflow collection')
    return {field: (float(np.mean([v[field] for v in values])) if field == 'mean_r2'
                    else np.mean([v[field] for v in values], axis=0).tolist()) for field in FIELDS}


def validate_reference_data(path):
    if not path.is_file():
        raise FileNotFoundError('Reference responses are not bundled for redistribution. '
                                'Import the official data first; see data/README.md. Missing: '+str(path))
    result = {}
    with np.load(path, allow_pickle=False) as z:
        for source, n in [('cgl', 231), ('corn', 80)]:
            key = source+'_y'
            if key not in z:
                raise ValueError('Imported reference data missing key '+key)
            y = np.asarray(z[key], dtype=np.float64)
            if y.shape != (n, 4) or not np.isfinite(y).all():
                raise ValueError('Invalid imported reference shape/values for '+source)
            result[source] = y.copy()
    return result


def validate_cells(cells):
    if len(cells) != 28 or len({c['id'] for c in cells}) != 28:
        raise ValueError('Expected exactly 28 unique evaluation cells')
    for c in cells:
        fit, validation, test = [set(c[field]) for field in ('fit_ids', 'validation_ids', 'test_ids')]
        expected_sizes = (122, 31, 78) if c['source'] == 'cgl' else (48, 16, 16)
        if tuple(map(len, (fit, validation, test))) != expected_sizes:
            raise ValueError('Wrong fit/validation/test partition sizes: '+c['id'])
        if fit & validation or fit & test or validation & test:
            raise ValueError('Overlapping partitions: '+c['id'])
        if sorted(fit | validation | test) != list(range(231 if c['source'] == 'cgl' else 80)):
            raise ValueError('Incomplete source IDs: '+c['id'])
    grouped = defaultdict(list)
    for c in cells:
        grouped[(c['source'], c['cohort'], c['selector_seed'])].append(c)
    for (source, cohort, seed), group in grouped.items():
        if source == 'corn':
            ids = [i for c in group for i in c['test_ids']]
            if len(group) != 5 or sorted(ids) != list(range(80)):
                raise ValueError('Corn OOF groups must cover each sample exactly once: '+cohort+'/'+str(seed))


def prediction_record(root, entry, cell, reference):
    path = (root/entry['prediction_file']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Prediction path escapes repository')
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != entry['prediction_sha256']:
        raise ValueError('Prediction file checksum mismatch: '+entry['prediction_file'])
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {'ids', 'prediction'}:
            raise ValueError('Unexpected public prediction fields: '+entry['prediction_file'])
        ids = np.asarray(z['ids'], dtype=np.int64).copy()
        pred = np.asarray(z['prediction'], dtype=np.float64).copy()
    if len(ids) != len(set(ids.tolist())) or ids.tolist() != cell['test_ids']:
        raise ValueError('Archived prediction ID order differs from its declared test partition')
    y = reference[cell['source']][ids]
    training_ids = cell['fit_ids']+cell['validation_ids']
    sd = reference[cell['source']][training_ids].std(axis=0, ddof=1)
    training_sd = np.broadcast_to(sd, y.shape).copy()
    return dict(ids=ids, prediction=pred, target=y, training_sd=training_sd,
                score=score(y, pred, training_sd))


def pool(records, expected_ids):
    ids = np.concatenate([r['ids'] for r in records])
    if len(ids) != len(set(ids.tolist())) or sorted(ids.tolist()) != expected_ids:
        raise ValueError('Duplicate or missing OOF sample IDs')
    order = np.argsort(ids)
    y, pred, sd = [np.vstack([r[field] for r in records])[order]
                   for field in ('target', 'prediction', 'training_sd')]
    return score(y, pred, sd)


def summarize(index, records):
    cells = index['cells']
    groups = defaultdict(list)
    for c in cells:
        groups[(c['source'], c['cohort'])].append(c)
    result = {}; workflows = []
    entries = index['pipelines']
    for (source, cohort), group in sorted(groups.items()):
        seeds = sorted({c['selector_seed'] for c in group})
        expected_ids = sorted({i for c in group for i in c['test_ids']})
        per_seed = {}
        for seed in seeds:
            per_seed[str(seed)] = {}
            for arm in ARMS:
                for rep in ([None] if arm in ARMS[:3] else range(1, 11)):
                    selected = [e for e in entries if (e['source'], e['cohort'], e['selector_seed'], e['arm'], e['replicate']) ==
                                (source, cohort, seed, arm, rep)]
                    if len(selected) != len([c for c in group if c['selector_seed'] == seed]):
                        raise ValueError('Missing evaluation cell in pooled workflow')
                    metric = pool([records[e['id']] for e in selected], expected_ids)
                    key = arm if rep is None else f'{arm}:{rep}'
                    per_seed[str(seed)][key] = metric
                    workflows.append(dict(source=source, cohort=cohort, selector_seed=seed,
                                          arm=arm, replicate=rep, rows=len(expected_ids), **metric))
        abc = {a:average([per_seed[str(seed)][a] for seed in seeds]) for a in ARMS[:3]}
        draws = {a:[dict(replicate=r, **average([per_seed[str(seed)][f'{a}:{r}'] for seed in seeds]))
                    for r in range(1, 11)] for a in ARMS[3:]}
        methods = {**abc, **{a:average(v) for a,v in draws.items()}}
        contrasts = {}
        for c,a in [('B','A'), ('C','B'), ('B','D_low'), ('C','D_wide')]:
            contrasts[c+'_minus_'+a] = dict(mean_r2=methods[c]['mean_r2']-methods[a]['mean_r2'],
                target_r2=(np.asarray(methods[c]['r2'])-methods[a]['r2']).tolist(),
                rmse_delta=(np.asarray(methods[c]['rmse'])-methods[a]['rmse']).tolist(),
                mae_delta=(np.asarray(methods[c]['mae'])-methods[a]['mae']).tolist(),
                normalized_rmse_delta=(np.asarray(methods[c]['normalized_rmse'])-methods[a]['normalized_rmse']).tolist())
        contrasts['range_difference'] = contrasts['C_minus_B']['mean_r2']-(methods['D_wide']['mean_r2']-methods['D_low']['mean_r2'])
        ranks = {}
        for arm, values in draws.items():
            anchor = abc['B' if arm == 'D_low' else 'C']['mean_r2']
            ranks[arm] = dict(B_or_C_better_count=sum(anchor > v['mean_r2'] for v in values), draws=10,
                              min=float(min(v['mean_r2'] for v in values)), max=float(max(v['mean_r2'] for v in values)))
        result[source+'/'+cohort] = dict(methods=methods, per_seed=per_seed, D_draws=draws, contrasts=contrasts,
            random_reference_position=ranks, sample_count_per_seed=len(expected_ids), algorithm_seeds=seeds,
            D_global_union_counts=[e['global_input_union_count'] for e in entries if e['source']==source and e['cohort']==cohort and e['arm']=='D_low'])
    sensitivity = []
    for key, cohort in result.items():
        if key.startswith('corn/'):
            ss = cohort['per_seed']['17']
            sensitivity.append(dict(cohort=key.split('/')[1], selector_seed=17,
                A=ss['A']['mean_r2'], B=ss['B']['mean_r2'], C=ss['C']['mean_r2'],
                D_low=float(np.mean([ss[f'D_low:{r}']['mean_r2'] for r in range(1,11)])),
                D_wide=float(np.mean([ss[f'D_wide:{r}']['mean_r2'] for r in range(1,11)])),
                B_minus_A=ss['B']['mean_r2']-ss['A']['mean_r2'], C_minus_B=ss['C']['mean_r2']-ss['B']['mean_r2']))
    return dict(cohorts=result, split_sensitivity_fixed_seed=sensitivity), workflows


def table_statistics(summary):
    random = {}; relative = []
    for key, cohort in summary['cohorts'].items():
        random[key] = {}
        for arm in ARMS[3:]:
            values = [v['mean_r2'] for v in cohort['D_draws'][arm]]
            random[key][arm] = dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)),
                                    ddof=1, n_random_workflows=10, values=values)
        if key.endswith('/original'):
            names = ['Casein', 'Glucose', 'Lactate', 'Moisture'] if key.startswith('cgl') else ['Moisture', 'Oil', 'Protein', 'Starch']
            methods = cohort['methods']
            for t,name in enumerate(names):
                relative.append(dict(source=key, target=name,
                    B_vs_A_percent=100*(methods['B']['rmse'][t]/methods['A']['rmse'][t]-1),
                    C_vs_A_percent=100*(methods['C']['rmse'][t]/methods['A']['rmse'][t]-1)))
    return dict(random_workflow_sd=random, relative_rmse=relative)


def full_spectrum(index, records):
    result = {}
    for source in ('cgl', 'corn'):
        entries = [e for e in index['full_spectrum'] if e['source']==source]
        ids = sorted({i for e in entries for i in records[e['id']]['ids'].tolist()})
        result[source] = pool([records[e['id']] for e in entries], ids)
    return result


def compare(actual, expected, label, maximum, atol):
    if isinstance(expected, dict):
        for key,value in expected.items():
            if key not in actual:
                raise AssertionError(f'Missing result key: {label}/{key}')
            compare(actual[key], value, label+'/'+key, maximum, atol)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError('Result-list length mismatch: '+label)
        for i,(a,b) in enumerate(zip(actual,expected)):
            compare(a,b,label+'/'+str(i),maximum,atol)
    elif isinstance(expected, (float,int)) and not isinstance(expected,bool):
        error = abs(float(actual)-float(expected))
        maximum[0] = max(maximum[0],error); maximum[1] += 1
        if not np.isfinite(error) or error > atol:
            raise AssertionError(f'Numerical mismatch at {label}: {actual!r} vs {expected!r}; abs error {error}')
    elif actual != expected:
        raise AssertionError('Result identity/value mismatch: '+label)


def run(root, reference_file, output, tolerance):
    root = root.resolve(); output = output.resolve()
    protected = [root/name for name in ('configs', 'results', 'scripts', 'data', 'figures', 'docs', 'archive')]
    if output == root or any(output == p or output.is_relative_to(p) for p in protected):
        raise ValueError('Output directory must not overwrite repository inputs. Use generated/ or a separate directory.')
    index = read(root/'configs/evaluation_index.json')
    if len(index['pipelines']) != 644 or len({e['id'] for e in index['pipelines']}) != 644:
        raise ValueError('Expected exactly 644 unique pipeline records')
    reference = validate_reference_data(reference_file)
    validate_cells(index['cells'])
    cells = {c['id']:c for c in index['cells']}
    records = {}; pipeline_metrics = []
    for entry in index['pipelines']+index['full_spectrum']:
        cell = cells[entry['cell_id']]
        record = prediction_record(root, entry, cell, reference)
        records[entry['id']] = record
        if entry.get('arm') != 'full_pls':
            pipeline_metrics.append(dict(id=entry['id'], cell_id=entry['cell_id'], source=entry['source'],
                cohort=entry['cohort'], fold=entry['fold'], selector_seed=entry['selector_seed'],
                arm=entry['arm'], replicate=entry['replicate'], score=record['score']))
    summary, workflows = summarize(index, records)
    if len(workflows) != 184:
        raise ValueError('Expected exactly 184 seed/arm/replicate complete workflows')
    derived = table_statistics(summary)
    full = full_spectrum(index, records)
    maximum = [0., 0]
    compare(pipeline_metrics, read(root/'results/expected/pipeline_metrics.json'), 'pipelines', maximum, tolerance)
    compare(summary, read(root/'results/expected/metrics_v5.json'), 'summary', maximum, tolerance)
    compare(derived, read(root/'results/expected/table_statistics.json'), 'table_statistics', maximum, tolerance)
    compare(full, read(root/'results/expected/full_spectrum.json'), 'full_spectrum', maximum, tolerance)
    output.mkdir(parents=True, exist_ok=True)
    for name,value in [('pipeline_metrics.json',pipeline_metrics),('pooled_metrics.json',workflows),
                       ('summary.json',summary),('table_statistics.json',derived),('full_spectrum_summary.json',full)]:
        write(output/name,value)
    report = dict(passed=True, pipeline_cases=644, logical_target_models=2576,
                  complete_seed_arm_replicate_workflows=184, full_spectrum_unique_prediction_files=6,
                  numeric_values_compared=maximum[1], max_absolute_difference=maximum[0], tolerance=tolerance,
                  training_sd_recomputed_from_local_reference_labels=True,
                  public_prediction_files_do_not_contain_reference_labels=True,
                  new_fits=0, model_loading=False, network_required=False,
                  scope='Archived-prediction metric recalculation only; not full training reproduction or independent validation.')
    write(output/'metrics_verification.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--reference-data', type=Path, default=None)
    parser.add_argument('--output-root', type=Path, default=None)
    parser.add_argument('--tolerance', type=float, default=1e-12)
    args = parser.parse_args(); root=args.root.resolve()
    if not np.isfinite(args.tolerance) or not 0 < args.tolerance <= 1e-8:
        parser.error('Tolerance must be finite and in (0, 1e-8].')
    reference = args.reference_data or root/'data/local/reference_data.npz'
    output = args.output_root or root/'generated'
    try:
        report = run(root,reference,output,args.tolerance)
    except (ValueError, FileNotFoundError, KeyError, AssertionError) as exc:
        print('RECHECK FAILED: '+str(exc),file=sys.stderr)
        return 2
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
