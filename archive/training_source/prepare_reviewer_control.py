"""Prepare only;real-data fit entry points are blocked throughout preparation."""
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import io
import json
import sys
import unittest
import numpy as np
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import reviewer_control_core as c
import run_reviewer_control as r


def protected_inputs():
    path=ROOT/'results/manuscript_v4_revision_001/delivery_manifest.json'
    manifest=c.read(path); protected={str(path):c.sha(path)}
    for key in ('immutable_files','protected_input_hashes','workspace_pointer_hashes'):
        for file,digest in manifest[key].items():
            if file in protected and protected[file]!=digest: raise RuntimeError('Conflicting historical digest')
            protected[file]=digest
    c.check_hashes(protected)
    return protected


def synthetic_tests():
    stream=io.StringIO(); suite=unittest.TestSuite()
    for pattern in ('test_reviewer_control.py','test_pool_boundary.py','test_public_validation.py','test_input_reuse_control.py'):
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    return result,stream.getvalue()


def original_cells(audit,raw_x,raw_y):
    ab=c.read(ROOT/'results/input_reuse_control_001/metrics.json'); cells=[]
    for source in ('cgl','corn'):
        info=audit[source]; panels=c.read(ROOT/'results/public_validation_001'/source/'panels.json')
        cc=c.read(ROOT/'results/public_validation_001'/source/'metrics.json')
        parts={'official':info['split']} if source=='cgl' else info['fold_records']
        for fold,part in parts.items():
            selection=ROOT/'data/public_validation_v1'/source/fold/'selection.npz'
            with np.load(selection,allow_pickle=False) as z:
                fi,vi=z['fit_ids'].copy(),z['reward_ids'].copy()
                if fi.tolist()!=part['fit_ids'] or vi.tolist()!=part['reward_ids']: raise AssertionError('Original sample order differs')
                if source=='corn':
                    for key,expected in [('X_fit',raw_x[fi]),('X_reward',raw_x[vi]),('y_fit',raw_y[fi]),('y_reward',raw_y[vi])]:
                        if not np.array_equal(z[key],expected): raise AssertionError('Raw Corn index/target mapping differs')
            for seed in c.SPEC['original_selector_seeds']:
                panel=panels[fold][str(seed)]; own,union=panel['individual'],panel['union']
                if sorted(set().union(*map(set,own)))!=union: raise AssertionError('Archived union differs')
                archives={}
                for arm in ('A','B','C'):
                    rows=[z for z in (cc if arm=='C' else ab) if z['fold']==fold and z['seed']==seed and
                        (z.get('method')=='union_pls' if arm=='C' else z['source']==source and z['arm']==('own' if arm=='A' else 'union'))]
                    if len(rows)!=1: raise AssertionError('Archived metric record not unique')
                    row=rows[0]; refs=row['refit_entries']
                    if len(refs)!=4: raise AssertionError('Missing archived target')
                    for target,path in enumerate(refs):
                        entry=c.read(path); expected=own[target] if arm=='A' else union
                        if entry['subset']!=expected: raise AssertionError('Archived ABC input mismatch')
                        cap=15 if arm=='C' else c.SPEC['quota'][source]
                        if not 1<=entry['components']<=cap: raise AssertionError('Archived candidate rule mismatch')
                    archives[arm]={k:row[k] for k in ('prediction_file','refit_entries')}
                cell=dict(id=c.cell_id(source,'original',fold,seed),source=source,cohort='original',outer_seed=0,
                    fold=fold,selector_seed=seed,mode='original',bands=info['input_columns'],quota=c.SPEC['quota'][source],
                    targets=info['targets'],fit_ids=part['fit_ids'],validation_ids=part['reward_ids'],test_ids=part['test_ids'],
                    selection_file=str(selection),own=own,union=union,archived=archives,
                    test_file=str(selection.parent/('test_features.npz' if source=='cgl' else 'test.npz')))
                if source=='cgl': cell['test_labels_from_prediction']=archives['A']['prediction_file']
                cells.append(cell)
    return cells


def validate_cells(cells):
    if len(cells)!=28 or len({z['id'] for z in cells})!=28: raise AssertionError('Cell count/identity mismatch')
    for cell in cells:
        fi,vi,ti=[cell[k] for k in ('fit_ids','validation_ids','test_ids')]
        if any(len(v)!=len(set(v)) for v in (fi,vi,ti)) or set(fi)&set(vi) or (set(fi)|set(vi))&set(ti):
            raise AssertionError('Duplicate/overlapping sample IDs')
        expected=(122,31,78) if cell['source']=='cgl' else (48,16,16)
        if tuple(map(len,(fi,vi,ti)))!=expected: raise AssertionError('Frozen cell sizes changed')
        if len(cell['targets'])!=4: raise AssertionError('Target count differs')
        if cell['mode']=='original':
            for own in cell['own']: c.validate_panel(own,cell['union'],cell['bands'],cell['quota'])
    for cohort in ('original','repeat_31001','repeat_31002'):
        cs=[z for z in cells if z['source']=='corn' and z['cohort']==cohort and z['selector_seed']==17]
        if len(cs)!=5 or sorted(i for z in cs for i in z['test_ids'])!=list(range(80)): raise AssertionError('Corn OOF coverage differs')


def prepare():
    if c.OUT.exists() or c.READY.exists(): raise RuntimeError('Preparation destination already exists;do not overwrite')
    if not c.DOC.is_file(): raise RuntimeError('Written protocol missing')
    protected=protected_inputs(); result,log=synthetic_tests()
    if not result.wasSuccessful():
        print(log); raise RuntimeError('Synthetic suite failed;no real-data preparation started')
    # A forbidden-fit trap stays active while every real-data record is assembled.
    def no_fit(*args,**kwargs): raise AssertionError('Real-data fitting forbidden during preparation')
    with ExitStack() as stack:
        stack.enter_context(patch.object(r,'pls_fit',side_effect=no_fit))
        stack.enter_context(patch.object(r.GenericPLSEvaluator,'evaluate',side_effect=no_fit))
        from sklearn.cross_decomposition import PLSRegression
        stack.enter_context(patch.object(PLSRegression,'fit',side_effect=no_fit))
        audit=c.read(ROOT/'data/public_validation_v1/source_audit.json')
        raw=Path(audit['corn']['raw_mat'])
        if c.sha(raw)!=audit['corn']['raw_sha256']: raise AssertionError('Corn source hash differs')
        mat=loadmat(raw,squeeze_me=True,struct_as_record=False,variable_names=['m5spec','propvals'])
        x,y=np.asarray(mat['m5spec'].data,dtype=float),np.asarray(mat['propvals'].data,dtype=float)
        if x.shape!=(80,700) or y.shape!=(80,4) or not np.isfinite(x).all() or not np.isfinite(y).all(): raise AssertionError('Corn source dimensions/values differ')
        cells=original_cells(audit,x,y)
        for source in ('cgl','corn'):
            old=c.read(ROOT/'results/public_validation_001'/source/'protocol.json')
            if old['versions']!=r.versions(): raise AssertionError('Scientific library versions changed from original')
        old_parts=[dict(test_ids=z['test_ids']) for z in audit['corn']['fold_records'].values()]
        signatures=[c.split_signature(old_parts)]; split_records={}
        for outer in c.SPEC['new_outer_seeds']:
            parts=c.make_partitions(80,outer); signature=c.split_signature(parts)
            if signature in signatures: raise AssertionError('Repeated outer partition')
            signatures.append(signature); split_records[str(outer)]=parts
        # Only after all preflight and source checks pass are new output folders made.
        c.OUT.mkdir(parents=True); c.READY.mkdir(parents=True)
        for outer in c.SPEC['new_outer_seeds']:
            cohort=f'repeat_{outer}'
            for part in split_records[str(outer)]:
                fi,vi,ti=[np.array(part[k],dtype=int) for k in ('fit_ids','validation_ids','test_ids')]
                folder=c.OUT/'data'/cohort/part['fold'];folder.mkdir(parents=True)
                # Held-out responses are copied by fixed ID only;no summaries,choices or metrics.
                np.savez_compressed(folder/'selection.npz',X_fit=x[fi],y_fit=y[fi],X_reward=x[vi],y_reward=y[vi],fit_ids=fi,reward_ids=vi)
                np.savez_compressed(folder/'test.npz',X_test=x[ti],y_test=y[ti],test_ids=ti)
                cells.append(dict(id=c.cell_id('corn',cohort,part['fold'],c.SPEC['new_selector_seed']),source='corn',cohort=cohort,
                    outer_seed=outer,fold=part['fold'],selector_seed=c.SPEC['new_selector_seed'],inner_seed=part['inner_seed'],
                    mode='new',bands=700,quota=11,targets=audit['corn']['targets'],fit_ids=fi.tolist(),validation_ids=vi.tolist(),test_ids=ti.tolist(),
                    selection_file=str(folder/'selection.npz'),test_file=str(folder/'test.npz')))
        validate_cells(cells)
        checker=object.__new__(r.Runner);checker.data_cache={}
        for cell in cells:checker.selection(cell)
        initial=[]
        for cell in cells:
            if cell['mode']!='original': continue
            for target,own in enumerate(cell['own']):
                for rep in range(1,11):
                    panel,entropy=c.random_panel(cell,own,cell['union'],target,rep)
                    initial.append(dict(cell_id=cell['id'],target=target,replicate=rep,subset=panel,own=own,union=cell['union'],rng_entropy=entropy,
                        other_target_columns_in_draw=len((set(panel)-set(own))&set(cell['union'])),per_model_input_count=len(panel),B_input_count=len(cell['union'])))
        if len(initial)!=720: raise AssertionError('Original random-panel count differs')
        budget=dict(new_search_candidate_fits_exact=122880,new_calibration_candidate_fits_upper=17840,total_candidate_fits_upper=140720,
            final_refits_upper=2360,new_searches=40,new_target_panel_records=1200,random_target_panels=1120,
            expected_pipeline_cases=644,expected_logical_target_models_including_archived=2576,
            historical_ABC_target_records_reused=216,component_candidate_reuse='Same random panel,shared candidate cache for both ranges')
        c.write(c.OUT/'cells.json',cells);c.write(c.OUT/'initial_random_panels.json',initial)
        c.write(c.OUT/'budget_plan.json',budget);c.write(c.OUT/'new_partition_ids.json',split_records)
        c.write(c.OUT/'data_preparation_audit.json',dict(no_real_data_fits=True,no_new_test_metrics=True,
            original_test_predictions_opened=False,new_heldout_labels='Copied by IDs only;not an unseen independent data source',
            raw_corn_sha256=c.sha(raw),old_training_arrays_equal_raw_indexed=True,new_partitions_distinct=True,
            original_cells=18,new_cells=10,original_random_panels=720,original_ABC_not_retrained=True))
        (c.READY/'tests.log').write_text(log,encoding='utf-8')
        c.check_hashes(protected)
        sources=[ROOT/n for n in ('reviewer_control_core.py','run_reviewer_control.py','prepare_reviewer_control.py','tests/test_reviewer_control.py','empirical_budget_core.py')]
        sources+=[c.DOC]; sources+=sorted((ROOT/'src/hsi_e2e').glob('*.py'))
        proof=dict(passed=True,synthetic_tests=result.testsRun,synthetic_failures=len(result.failures),synthetic_errors=len(result.errors),
            historical_files_unchanged=len(protected),original_cells=18,new_cells=10,original_random_panels=720,
            real_data_fit_entry_points_blocked=True,new_real_data_fits=0,new_test_metrics=0,experiment_started=False,
            fixed_id_isolation_passed=True,raw_index_alignment_passed=True,budget=budget)
        c.write(c.READY/'verification.json',proof)
        c.write(c.OUT/'protocol.json',dict(spec=c.SPEC,versions=r.versions(),source_sha256={str(p):c.sha(p) for p in sources},
            protected_input_hashes=protected,readiness_verification=str(c.READY/'verification.json'),budget=budget,
            hardware_policy='CPU;Torch4,interop1,BLAS1;matches original PLS runtime;no GPU model training',
            fitting_authority='Only CLI run from user terminal;preparation never fits research data'))
        files=[p for p in c.OUT.rglob('*') if p.is_file()]+[c.READY/'tests.log',c.READY/'verification.json']
        c.write(c.OUT/'prepared_manifest.json',dict(files={str(p):c.sha(p) for p in sorted(files)},state='Prepared,not trained'))
        c.write(c.OUT/'status.json',dict(state='prepared_not_started',new_test_scoring=False))
        runner=r.Runner();runner.lock()
        # ID/shape/finite checks use selection files only;test loader must remain sealed.
        for cell in cells: runner.selection(cell)
        try:runner.test(cells[0])
        except RuntimeError as exc:
            if 'not sealed' not in str(exc):raise
        else:raise AssertionError('Test loader accessible before global seal')
        c.check_hashes(protected)
        print(json.dumps(proof,ensure_ascii=False,indent=2))
        print('PREPARED reviewer_control_v1;no research fitting or new test scoring started')


if __name__=='__main__':prepare()
