"""Cross-check archived figure, selection, table and cost records without labels.

This checks the consistency of saved records. It is not model reloading, training,
an independent dataset test or a reconstruction of historical model execution.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import importlib.util
import json
import math

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'evidence'

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def check(output=None):
    # The independent figure checker uses standard-library arithmetic and CSVs.
    path=ROOT/'scripts/figures/verify_numbers.py'
    spec=importlib.util.spec_from_file_location('figure_numbers',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.main()
    figure_report=read(ROOT/'generated/figure_qa/numeric_verification.json')
    curves=read(EVIDENCE/'figure_data/candidate_curves.json')
    pairs=read(EVIDENCE/'historical_ABC/component_pair_records.json')
    assert len(curves)==72 and len(pairs)==72
    by_key={(r['source'],r['fold'],int(r['seed']),int(r['target'])):r for r in curves}
    for row in pairs:
        rec=by_key[(row['source'],row['fold'],int(row['seed']),int(row['target']))]
        assert all(int(row[arm+'_components'])==int(rec[arm+'_components']) for arm in 'ABC')
        assert row['A_input_count']==(12 if row['source']=='cgl' else 11)
        assert row['B_input_count']==row['C_input_count']==row['global_input_union_count']
    boundary={}
    for source,denominator in [('cgl',12),('corn',60)]:
        part=[r for r in curves if r['source']==source]
        assert len(part)==denominator
        boundary[source]=dict(target_pairs=denominator,
            B_at_cap=sum(r['B_components']==r['B_upper'] for r in part),
            C_at_15=sum(r['C_components']==15 for r in part),
            C_exceeds_B_cap=sum(r['C_components']>r['B_upper'] for r in part),
            union_15_above_14=sum(r['union_trials'][14]['mean_r2']>r['union_trials'][13]['mean_r2'] for r in part),
            union_11_to_15_every_step_up=sum(all(r['union_trials'][j]['mean_r2']>r['union_trials'][j-1]['mean_r2'] for j in range(11,15)) for r in part))
    assert sum(r['B_at_cap'] for r in boundary.values())==43
    assert sum(r['C_at_15'] for r in boundary.values())==32
    assert sum(r['C_exceeds_B_cap'] for r in boundary.values())==61
    assert boundary['corn']['union_15_above_14']==35 and boundary['corn']['union_11_to_15_every_step_up']==20
    directions=read(EVIDENCE/'historical_ABC/changed_component_score_directions.json')
    assert sum(v['improved'] for v in directions.values())==52
    assert sum(v['worse'] for v in directions.values())==9

    columns=read(EVIDENCE/'components/selected_columns.json')['records']
    assert len(columns)==72
    lookup={(r['source'],r['fold'],int(r['seed']),int(r['target'])):r for r in columns}
    overlaps=read(EVIDENCE/'components/exact_column_overlap.json');assert len(overlaps)==90
    jaccard=defaultdict(list)
    for record in overlaps:
        target=record.get('target') if record['scope']=='own' else 0
        a=lookup[(record['source'],record['fold'],int(record['seed_1']),int(target))]
        b=lookup[(record['source'],record['fold'],int(record['seed_2']),int(target))]
        name='subset' if record['scope']=='own' else 'global_input_union'
        a,b=set(a[name]),set(b[name]);value=len(a&b)/len(a|b)
        assert abs(value-record['jaccard'])<1e-15
        assert len(a&b)==record['intersection_size'] and len(a|b)==record['union_size']
        jaccard[record['source']+'/'+record['scope']].append(value)
    jaccard={k:sum(v)/len(v) for k,v in jaccard.items()}
    expected_jaccard={'cgl/own':.086081,'corn/own':.016190,'cgl/union':.216854,'corn/union':.038731}
    for key,value in expected_jaccard.items():assert round(jaccard[key],6)==value

    searches=read(EVIDENCE/'components/original_target_search_counts.json')
    assert len(searches)==72 and sum(v['recorded_fits'] for v in searches)==221184
    old_cost=read(EVIDENCE/'historical_ABC/cost_accounting.json')
    assert (old_cost['new_candidate_fits'],old_cost['new_final_refits'],old_cost['reused_final_models'])==(804,133,11)
    new_cost=read(EVIDENCE/'new_phase_costs.json')
    assert new_cost==dict(search_candidate_fits=122880,calibration_candidate_fits=17840,final_refits=2229,
                         final_models_reused_within_panel=131,historical_ABC_target_records_reused=216)
    execution=read(EVIDENCE/'archived_execution_verification.json')
    assert execution['pipeline_cases']==644 and execution['logical_target_models']==2576
    assert 28*23==644 and 644*4==2576 and 2576-216-131==2229
    # Historical reload statements are retained as records, not rerun here.
    assert execution['max_prediction_reload_error']==0 and not execution['D_global_budget_matched']

    panels=read(EVIDENCE/'input_panels/panels.json')
    draws=read(EVIDENCE/'input_panels/random_panels.json')
    assert len(panels)==28 and len(draws)==1120
    for row in draws:
        panel=panels[row['cell_id']]
        own=set(row['own']);union=set(row['union']);chosen=set(row['subset'])
        assert set(panel['own'][int(row['target'])])==own
        assert set(panel['union'])==union and own<=chosen
        assert len(chosen)==len(row['subset'])==len(union)==row['per_model_input_count']==row['B_input_count']
        assert row['other_target_columns_in_draw']==len((chosen-own)&union)

    tables=read(EVIDENCE/'tables/published_tables.json')
    assert len(tables['tables'])==34
    assert sum(r['document']=='main' for r in tables['tables'])==6
    assert sum(r['document']=='supplement' for r in tables['tables'])==28
    summary=read(EVIDENCE/'figure_data/summary.json')
    added=read(EVIDENCE/'added_table_statistics.json')
    for cohort,methods in added['random_workflow_sd'].items():
        for method,record in methods.items():
            values=[r['mean_r2'] for r in summary['cohorts'][cohort]['D_draws'][method]]
            average=sum(values)/10;sd=math.sqrt(sum((v-average)**2 for v in values)/9)
            assert abs(average-record['mean'])<1e-14 and abs(sd-record['sd'])<1e-14
    for record in added['relative_rmse']:
        co=summary['cohorts'][record['source']]['methods']
        names=['Casein','Glucose','Lactate','Moisture'] if record['source'].startswith('cgl') else ['Moisture','Oil','Protein','Starch']
        target=names.index(record['target'])
        for arm in ['B','C']:
            value=100*(co[arm]['rmse'][target]/co['A']['rmse'][target]-1)
            assert abs(value-record[arm+'_vs_A_percent'])<1e-12
    report=dict(passed=True,scope='Saved-record arithmetic and identity checks; not a saved-model reload or model-fitting audit.',
        figures=figure_report,original_target_pairs=72,component_boundaries=boundary,
        exact_column_jaccard=jaccard,table_snapshots=34,main_tables=6,supplementary_tables=28,
        candidate_search_counts_verified=True,november24_SD_cells=8,november24_B_A_cells=8,
        original_changed_score_improved=52,original_changed_score_declined=9,
        original_and_new_input_cells=28,shared_dual_range_random_target_panels=1120,
        archived_execution_report_consistent=True,raw_reference_labels_loaded=False,new_model_fits=0)
    output=Path(output) if output else ROOT/'generated/evidence_report.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    args=parser.parse_args();print(json.dumps(check(args.output),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
