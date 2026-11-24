"""Trace and check every published V5 table without fitting models.

With --use-recomputed, generated/summary.json is used. A successful companion
metrics_verification.json is required before that route is labelled as derived
from archived-prediction recalculation. Otherwise numerical checks are explicitly
saved-record consistency checks. Literature table S13 is scope/text checking,
not an independent literature review. No raw labels or models are loaded here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
from statistics import mean, stdev
import sys

ROOT = Path(__file__).resolve().parents[1]
COHORTS = ('cgl/original', 'corn/original', 'corn/repeat_31001', 'corn/repeat_31002')
LABELS = dict(zip(COHORTS, ('CGL 原划分', 'Corn 原划分', 'Corn 31001', 'Corn 31002')))
ARMS = ('A', 'B', 'C', 'D_low', 'D_wide')
TARGETS = {'cgl': ('Casein', 'Glucose', 'Lactate', 'Moisture'),
           'corn': ('Moisture', 'Oil', 'Protein', 'Starch')}
SNAPSHOT = 'evidence/tables/published_tables.json'
EVAL = 'configs/evaluation_index.json'
METRICS = 'results/expected/metrics_v5.json'
PAIRS = 'evidence/historical_ABC/component_pair_records.json'
CURVES = 'evidence/figure_data/candidate_curves.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def f(value, decimals=6):
    return f'{float(value):.{decimals}f}'


def signed(value, decimals=6):
    return f'{float(value):+.{decimals}f}'


def extent(values):
    lo, hi = min(values), max(values)
    return str(lo) if lo == hi else f'{lo}—{hi}'


def norm(value):
    # Logical table text, independent of layout-only line breaks and minus glyphs.
    return re.sub(r'\s+', '', str(value)).replace('−', '-').replace('\u2060', '')


def coverage_definitions():
    """Static, public provenance index; does not claim snapshot presence is proof."""
    definitions = []
    def add(doc, number, label, preferred, fallback, level, definition, limits=''):
        definitions.append(dict(id=f'{doc}:{number}', document=doc, table_index=number,
            table_label=label, preferred_sources=preferred, fallback_sources=fallback,
            check_class=level, definition=definition, limitations=limits))
    metric_preferred = ['generated/summary.json', 'generated/metrics_verification.json']
    metric_fallback = [METRICS]
    add('main',1,'Table 1',[EVAL],[],'protocol_definition',
        'Source size, band count, per-cell partition sizes, and selector seed lists from evaluation cells.')
    add('main',2,'Table 2',[EVAL,PAIRS],[],'protocol_definition',
        'A reads Sj; B/C read U; matched random panels retain Sj. Narrow caps are source quotas; wide cap is 15.')
    for n,label,definition in [
        (3,'Table 3','A/B/C macro R2, and D workflow mean plus sample SD (ddof=1) across ten complete draws.'),
        (4,'Table 4','B-Dlow, C-Dwide, difference of range differences, and strictly higher draw counts.'),
        (5,'Table 5','Corn selector seed 17 only, complete OOF metrics and paired differences across three partitions.'),
        (6,'Table 6','Original-partition target RMSE and B/A, C/A percentage changes from unrounded aggregated metrics.')]:
        add('main',n,label,metric_preferred,metric_fallback,'metric_summary',definition,
            'Generated metrics are prior archived-prediction recalculation, not newly trained models or independent validation.')
    add('supplement',1,'Table S1',[PAIRS],[],'saved_selection_arithmetic',
        'Per-target component minima/maxima and C choices above the source-specific B cap.')
    for n,label,definition in [(2,'Table S2','All original target metrics including training-SD-normalized RMSE.'),
                              (3,'Table S3a','Original target R2, RMSE, MAE differences and aggregated-RMSE percentage change.')]:
        add('supplement',n,label,metric_preferred,metric_fallback,'metric_summary',definition)
    add('supplement',4,'Table S3b',['generated/resampling_report.json'],
        ['evidence/fixed_prediction_resampling/expected.json'],'fixed_prediction_resampling',
        '2.5 and 97.5 percentiles from the archived 10000 paired row resamples.',
        'Fixed predictions only; does not cover new selection/training uncertainty or independent samples.')
    for n,label in [(5,'Table S4a'),(6,'Table S4b')]:
        add('supplement',n,label,[PAIRS],[],'saved_selection_arithmetic',
            'Every original fold/seed/target input count, actual component choice, and C-above-cap flag.')
    for n,label in [(7,'Table S7a'),(8,'Table S7b')]:
        add('supplement',n,label,[CURVES],[],'saved_candidate_arithmetic',
            'Actual choices and boundary/slope counts recomputed from the archived candidate arrays.',
            'Checks saved candidate scores, not their refitting or a proof of optimality.')
    add('supplement',9,'Table S7c',metric_preferred,metric_fallback,'metric_summary',
        'Original per-selector-seed macro R2 and B-A, C-B; nine decimal display precision.')
    add('supplement',10,'Table S8',['evidence/components/selected_columns.json'],[],
        'saved_index_jaccard','Exact set Jaccard recomputed for each seed pair within each fold and target.',
        'No chance correction or wavelength-neighborhood clustering; not a complete stability assessment.')
    for n,label in [(11,'Table S10a'),(12,'Table S10b')]:
        add('supplement',n,label,metric_preferred+['generated/full_spectrum_summary.json'],
            [METRICS,'results/expected/full_spectrum.json'],'metric_summary_with_full_spectrum',
            'Original C and full-spectrum background metrics and their differences.',
            'Full-spectrum is an original-partition background; no additional-partition full models are inferred.')
    add('supplement',13,'Table S12a',['evidence/components/original_target_search_counts.json',
        'evidence/historical_ABC/cost_accounting.json',PAIRS,'evidence/components/full_spectrum_conditions.json'],[],
        'archived_cost_accounting','Historical search/candidate/refit/reuse counts and stage arithmetic.',
        'Archived execution counts; historical searches are not rerun.')
    add('supplement',14,'Table S12b',['evidence/new_phase_costs.json',EVAL],[],
        'archived_cost_accounting','New search, calibration, refit/reuse counts and their sum.',
        'Counts refer to distinct stages and are not independent experiments.')
    add('supplement',15,'Table S13',[SNAPSHOT],[],'text_scope_only',
        'Checks column definitions, named comparison rows, nonempty protocol descriptions, and this-study scope.',
        'Literature details are not independently rechecked in this reproduction package; not a novelty or priority proof.')
    for n,label in [(16,'Table S14a'),(17,'Table S14b'),(18,'Table S14c'),(19,'Table S14d')]:
        add('supplement',n,label,metric_preferred,metric_fallback,'metric_summary',
            'Original D metrics or all new-partition arm/target metrics; each D averaged over ten workflows.')
    add('supplement',20,'Table S14e',[EVAL,'evidence/input_panels/panels.json',
        'evidence/input_panels/random_panels.json'],[],'saved_input_index_arithmetic',
        'Per-model input quotas and ranges of U and the four-target D global unions reconstructed from saved indices.',
        'D only matches each model column count; the global acquisition budget is not matched.')
    add('supplement',21,'Table S15',metric_preferred,metric_fallback,'metric_summary',
        'Per-seed B-A, C-B, B-Dlow and C-Dwide; D averages ten complete workflows.')
    for n,label in [(22,'Table S16a'),(23,'Table S16b'),(24,'Table S16c'),(25,'Table S16d')]:
        add('supplement',n,label,metric_preferred,metric_fallback,'metric_summary',
            'Every one of ten D workflow macro scores and its paired B/C difference, without selection of draws.')
    for n,label in [(26,'Table S17a'),(27,'Table S17b')]:
        add('supplement',n,label,[EVAL],[],'saved_selection_arithmetic',
            'New-partition fold/target U size and A/B/C component choices from archived pipeline index.')
    add('supplement',28,'Table S19',['evidence/archived_execution_verification.json',
        'evidence/new_phase_costs.json',EVAL],[],'archived_execution_record',
        'Archived execution report values, counts, and boolean scope statements.',
        'A saved report consistency check; this does not independently rerun the original candidate models or establish blind validation.')
    return {'schema_version':1,'published_version':'V5 2026-11-24','table_count':34,
        'policy':'All tables are tracked, but verification levels differ. Generated files are used only with --use-recomputed; snapshot presence alone is not verification.',
        'rounding':'Numerical cells are reconstructed at their stated display precision; merged logical cells repeat.',
        'tables':definitions}


class Audit:
    def __init__(self, root, generated, use_recomputed=False):
        self.root, self.generated = root, generated
        snapshot = read(root/SNAPSHOT)
        self.tables = {(t['document'],int(t['table_index'])):t for t in snapshot['tables']}
        if len(self.tables)!=34 or snapshot['table_count']!=34:
            raise ValueError('Published table snapshot must contain 6 main and 28 supplementary tables')
        self.results=[]
        self.summary_path = generated/'summary.json' if use_recomputed else root/METRICS
        self.summary = read(self.summary_path)
        report = read(generated/'metrics_verification.json') if use_recomputed else {}
        self.attested = (self.summary_path==generated/'summary.json' and report.get('passed') is True
                         and report.get('training_sd_recomputed_from_local_reference_labels') is True)
        if use_recomputed and not self.attested:
            raise ValueError('--use-recomputed requires a successful generated metrics_verification.json')
        self.metric_level = 'archived_prediction_derived_metrics' if self.attested else 'saved_metric_record_consistency'
        self.index=read(root/EVAL); self.cells=self.index['cells']; self.pipelines=self.index['pipelines']
        self.pairs=read(root/PAIRS); self.curves=read(root/CURVES)
        self.definitions={(r['document'],r['table_index']):r for r in coverage_definitions()['tables']}

    def relative(self,path):
        try:return Path(path).relative_to(self.root).as_posix()
        except ValueError:return str(Path(path))

    def metric(self,key,arm):return self.summary['cohorts'][key]['methods'][arm]

    def compare(self,doc,index,expected,headers=1,level=None,sources=None,definition_checks=None):
        table=self.tables[(doc,index)]; actual=table['rows']; failures=[]
        if len(actual)!=headers+len(expected):
            failures.append(dict(kind='row_count',actual=len(actual),expected=headers+len(expected)))
        checked=0
        for i,(a,b) in enumerate(zip(actual[headers:],expected),headers+1):
            if len(a)!=len(b):failures.append(dict(row=i,kind='column_count',actual=len(a),expected=len(b)))
            for j,(av,bv) in enumerate(zip(a,b),1):
                checked+=1
                if norm(av)!=norm(bv):failures.append(dict(row=i,column=j,actual=av,expected=str(bv)))
        for description,condition in definition_checks or []:
            if not condition:failures.append(dict(kind='definition',description=description))
        definition=self.definitions[(doc,index)]
        self.results.append(dict(id=definition['id'],caption=table['caption'],passed=not failures,
            verification_level=level or definition['check_class'],source_files=sources or definition['preferred_sources'],
            checked_body_cells=checked,header_rows=headers,definition_checks=len(definition_checks or []),
            limitations=definition['limitations'],failures=failures))

    def metrics_table(self,doc,index,expected,headers=1,extra_sources=None):
        self.compare(doc,index,expected,headers,self.metric_level,
                     [self.relative(self.summary_path)]+(extra_sources or []))


def run(root=ROOT, generated_root=None, use_recomputed=False):
    root=Path(root).resolve(); generated=Path(generated_root or root/'generated').resolve()
    if generated==root or any(generated==root/p or generated.is_relative_to(root/p)
                              for p in ['evidence','configs','results','scripts','data','figures']):
        raise ValueError('Generated output must not overwrite repository inputs')
    a=Audit(root,generated,use_recomputed); defs=coverage_definitions()
    recorded=read(root/'evidence/tables/coverage_index.json')
    if recorded!=defs:raise ValueError('Coverage index does not match the table checker definitions')
    source_name=lambda s:'CGL' if s=='cgl' else 'Corn'

    # Protocol tables have no independent model-performance claim.
    rows=[]
    for key in COHORTS:
        cells=[c for c in a.cells if c['source']+'/'+c['cohort']==key]; c=cells[0]
        n=len(c['fit_ids'])+len(c['validation_ids'])+len(c['test_ids'])
        label={'cgl/original':'CGL 官方划分','corn/original':'Corn 原五折',
               'corn/repeat_31001':'Corn 新五折31001','corn/repeat_31002':'Corn 新五折31002'}[key]
        rows.append([label,('同' if 'repeat_' in key else '')+f"{n}／{c['bands']}",
            ('每折' if c['source']=='corn' else '')+'／'.join(str(len(c[k])) for k in ['fit_ids','validation_ids','test_ids']),
            '、'.join(map(str,sorted({c['selector_seed'] for c in cells})))])
    a.compare('main',1,rows,definition_checks=[('four source-partition settings',len(rows)==4)])
    a.compare('main',2,[['A 目标专用','Sj','1—12','1—11'],['B 并集窄范围','U','1—12','1—11'],
        ['C 并集宽范围','同一U','1—15','1—15'],['Dlow 随机窄范围','Sj加随机列至K列','与B相同','与B相同'],
        ['Dwide 随机宽范围','与Dlow同一面板','与C相同','与C相同']],headers=2,
        definition_checks=[('dual header names source-specific component ranges',
            norm(a.tables[('main',2)]['rows'][0][2])=='PLS成分候选范围' and
            a.tables[('main',2)]['rows'][1][2:]==['CGL','Corn'])])
    rows=[]
    for key in COHORTS:
        cohort=a.summary['cohorts'][key]
        row=[LABELS[key]]+[f(a.metric(key,arm)['mean_r2']) for arm in 'ABC']
        for arm in ['D_low','D_wide']:
            values=[r['mean_r2'] for r in cohort['D_draws'][arm]]
            if len(values)!=10:raise ValueError('Expected ten complete D workflows')
            row.append(f(mean(values))+' ± '+f(stdev(values)))
        rows.append(row)
    a.metrics_table('main',3,rows,2)
    rows=[]
    for key in COHORTS:
        b,c,lo,wi=[a.metric(key,x)['mean_r2'] for x in ['B','C','D_low','D_wide']]
        cohort=a.summary['cohorts'][key]
        counts=[str(sum(base>d['mean_r2'] for d in cohort['D_draws'][arm]))+'/10'
                for base,arm in [(b,'D_low'),(c,'D_wide')]]
        rows.append([LABELS[key],signed(b-lo),signed(c-wi),signed((c-b)-(wi-lo))]+counts)
    a.metrics_table('main',4,rows,2)
    rows=[]
    for i,cohort in enumerate(['original','repeat_31001','repeat_31002']):
        v=a.summary['cohorts']['corn/'+cohort]['per_seed']['17']
        scores=[v[arm]['mean_r2'] for arm in 'ABC']; dw=mean(v[f'D_wide:{r}']['mean_r2'] for r in range(1,11))
        rows.append([['原五折','新五折31001','新五折31002'][i]]+[f(x) for x in scores]+
                    [signed(scores[1]-scores[0]),signed(scores[2]-scores[1]),signed(scores[2]-dw)])
    a.metrics_table('main',5,rows,2)
    rows=[]
    for source in ['cgl','corn']:
        key=source+'/original'
        for target,name in enumerate(TARGETS[source]):
            vals=[a.metric(key,arm)['rmse'][target] for arm in ARMS]
            rows.append([source_name(source),name]+[f(x) for x in vals]+
                        [signed(100*(vals[j]/vals[0]-1),2)+'%' for j in [1,2]])
    a.metrics_table('main',6,rows,2)

    rows=[]
    for source in ['cgl','corn']:
        for target,name in enumerate(TARGETS[source]):
            part=[p for p in a.pairs if p['source']==source and p['target']==target]
            rows.append([source_name(source),name+(' (wt %)' if source=='cgl' else ''),str(len(part))]+
                [extent([p[arm+'_components'] for p in part]) for arm in 'ABC']+
                [str(sum(p['C_components']>p['B_upper'] for p in part))+'/'+str(len(part))])
    a.compare('supplement',1,rows)
    rows=[]
    for source in ['cgl','corn']:
        for arm in 'ABC':
            v=a.metric(source+'/original',arm)
            for target,name in enumerate(TARGETS[source]):
                rows.append([source_name(source),arm,name+(' (wt %)' if source=='cgl' else '')]+
                    [f(v[field][target]) for field in ['r2','rmse','mae','normalized_rmse']])
    a.metrics_table('supplement',2,rows)
    rows=[]
    for source in ['cgl','corn']:
        for num,den in [('B','A'),('C','B'),('C','A')]:
            x,y=[a.metric(source+'/original',arm) for arm in [num,den]]
            for t,name in enumerate(TARGETS[source]):
                rows.append([source_name(source),num+'−'+den,name+(' (wt %)' if source=='cgl' else ''),
                    signed(x['r2'][t]-y['r2'][t]),signed(x['rmse'][t]-y['rmse'][t]),
                    signed(100*(x['rmse'][t]/y['rmse'][t]-1),2)+'%',signed(x['mae'][t]-y['mae'][t])])
    a.metrics_table('supplement',3,rows)
    rp=generated/'resampling_report.json'
    if use_recomputed:
        res=read(rp)
        if res.get('passed') is not True:raise ValueError('Generated resampling verification is not successful')
        intervals={s:res['sources'][s]['macro_r2_percentiles'] for s in ['cgl','corn']}
        rlevel='archived_prediction_fixed_resampling'; rsource=a.relative(rp)
    else:
        rsource='evidence/fixed_prediction_resampling/expected.json'; intervals=read(root/rsource)['macro_r2']
        rlevel='saved_resampling_record_consistency'
    rows=[]
    for source in ['cgl','corn']:
        for name in ['B_minus_A','C_minus_B','C_minus_A']:
            lo,hi=intervals[source][name]
            rows.append([source_name(source),name.replace('_minus_','−'),'['+signed(lo)+', '+signed(hi)+']'])
    a.compare('supplement',4,rows,level=rlevel,sources=[rsource])
    for table,source in [(5,'cgl'),(6,'corn')]:
        rows=[]
        for p in a.pairs:
            if p['source']!=source:continue
            rows.append([p['fold'],str(p['seed']),p['target_name'].replace(' (wt %)',''),str(p['A_input_count']),
                str(p['B_input_count'])]+[str(p[x+'_components']) for x in 'ABC']+
                ['是' if p['C_components']>p['B_upper'] else '否'])
        a.compare('supplement',table,rows)
    rows=[]
    for source in ['cgl','corn']:
        part=[v for v in a.curves if v['source']==source]
        values=[len(part),sum(v['B_components']==v['B_upper'] for v in part),sum(v['C_components']==15 for v in part),
                sum(v['C_components']>v['B_upper'] for v in part),
                sum(v['union_trials'][14]['mean_r2']>v['union_trials'][13]['mean_r2'] for v in part),
                sum(all(v['union_trials'][j]['mean_r2']>v['union_trials'][j-1]['mean_r2'] for j in range(11,15)) for v in part)]
        rows.append([source_name(source)]+list(map(str,values)))
    a.compare('supplement',7,rows)
    rows=[]
    for source in ['cgl','corn']:
        for target,name in enumerate(TARGETS[source]):
            part=[v for v in a.curves if v['source']==source and v['target']==target]
            values=[len(part),sum(v['B_components']==v['B_upper'] for v in part),sum(v['C_components']==15 for v in part),
                    sum(v['C_components']>v['B_upper'] for v in part)]
            rows.append([source_name(source),name]+list(map(str,values)))
    a.compare('supplement',8,rows)
    rows=[]
    for source in ['cgl','corn']:
        for seed,v in a.summary['cohorts'][source+'/original']['per_seed'].items():
            x,y,z=[v[arm]['mean_r2'] for arm in 'ABC']
            rows.append([source_name(source),seed,f(x,9),f(y,9),f(z,9),signed(y-x,9),signed(z-y,9)])
    a.metrics_table('supplement',9,rows)
    records=read(root/'evidence/components/selected_columns.json')['records']; groups=defaultdict(dict)
    for r in records:groups[(r['source'],r['fold'],r['target'])][r['seed']]=r
    overlaps=defaultdict(list); unions_done=set()
    for (source,fold,target),seeds in groups.items():
        for sa,sb in itertools.combinations(sorted(seeds),2):
            x,y=[set(seeds[s]['subset']) for s in [sa,sb]]
            overlaps[(source,'own')].append(len(x&y)/len(x|y))
            key=(source,fold,sa,sb)
            if key not in unions_done:
                unions_done.add(key); x,y=[set(seeds[s]['global_input_union']) for s in [sa,sb]]
                overlaps[(source,'union')].append(len(x&y)/len(x|y))
    a.compare('supplement',10,[[source_name(s),f(mean(overlaps[(s,'own')])),f(mean(overlaps[(s,'union')])),
        str(len(overlaps[(s,'own')])),str(len(overlaps[(s,'union')]))] for s in ['cgl','corn']])
    fullpath=generated/'full_spectrum_summary.json' if use_recomputed else root/'results/expected/full_spectrum.json'
    full=read(fullpath); level=a.metric_level if fullpath==generated/'full_spectrum_summary.json' else 'saved_metric_record_consistency'
    rows=[[source_name(s),f(a.metric(s+'/original','C')['mean_r2']),f(full[s]['mean_r2']),
           signed(a.metric(s+'/original','C')['mean_r2']-full[s]['mean_r2'])] for s in ['cgl','corn']]
    sources=[a.relative(a.summary_path),a.relative(fullpath)]
    a.compare('supplement',11,rows,level=level,sources=sources)
    rows=[]
    for source in ['cgl','corn']:
        c=a.metric(source+'/original','C'); fullm=full[source]
        for t,name in enumerate(TARGETS[source]):
            rows.append([source_name(source),name+(' (wt %)' if source=='cgl' else '')]+
                        [f(v[field][t]) for field in ['r2','rmse','mae'] for v in [c,fullm]])
    a.compare('supplement',12,rows,level=level,sources=sources)

    # Actual historical execution is not reconstructed from these cost records.
    searches=read(root/'evidence/components/original_target_search_counts.json')
    old=read(root/'evidence/historical_ABC/cost_accounting.json'); new=read(root/'evidence/new_phase_costs.json')
    counts={s:sum(v['recorded_fits'] for v in searches if v['source']==s) for s in ['cgl','corn']}
    npairs=Counter(p['source'] for p in a.pairs)
    acond={s:sum(p['A_input_count'] for p in a.pairs if p['source']==s) for s in ['cgl','corn']}
    conditions=read(root/'evidence/components/full_spectrum_conditions.json')
    nfull=Counter(c['source'] for c in conditions); fullc={s:sum(len(c['full_candidate_grid']) for c in conditions if c['source']==s) for s in ['cgl','corn']}
    rows=[['历史目标子集生成','每目标／划分／种子3072次',str(counts['cgl']),str(counts['corn']),
            f"{sum(counts.values())}次；{len(searches)}个原搜索，未重跑"],
          ['历史C成分候选','每目标配对1—15',str(npairs['cgl']*15),str(npairs['corn']*15),f"{len(a.pairs)*15}次历史拟合；B从中选择"],
          ['追加A成分候选','CGL每配对12次；Corn11次',str(acond['cgl']),str(acond['corn']),f"{sum(acond.values())}次新增拟合"],
          ['追加B成分候选','读取历史C候选','0','0','没有新增候选拟合'],
          ['A/B最终模型',f"{old['new_logical_target_model_records']}条逻辑记录",'—','—',
           f"{old['new_final_refits']}次新增重拟合＋{old['reused_final_models']}次历史复用"],
          ['历史全谱模型','每折每目标15个候选；一份最终模型',f"{fullc['cgl']}候选＋{nfull['cgl']}重拟合",
           f"{fullc['corn']}候选＋{nfull['corn']}重拟合",'确定性结果跨种子复用；本轮零拟合']]
    a.compare('supplement',13,rows,definition_checks=[('A/B accounting sums',
        sum(acond.values())==old['new_candidate_fits'] and old['new_final_refits']+old['reused_final_models']==old['new_logical_target_model_records'])])
    a.compare('supplement',14,[['新增目标子集搜索',str(new['search_candidate_fits']),'40次搜索，每目标3072个唯一候选'],
        ['新增成分校准候选',str(new['calibration_candidate_fits']),'D的窄／宽范围共享候选缓存'],
        ['两类候选拟合合计',str(new['search_candidate_fits']+new['calibration_candidate_fits']),'分别列阶段，不等于独立实验'],
        ['新增最终重拟合',str(new['final_refits']),'按实际选中成分重拟合'],
        ['同面板最终模型复用',str(new['final_models_reused_within_panel']),'两个范围选中相同成分时复用'],
        ['历史A/B/C目标记录复用',str(new['historical_ABC_target_records_reused']),'不计作本轮新增拟合']],
        definition_checks=[('new refit arithmetic',len(a.pipelines)*4-new['historical_ABC_target_records_reused']-new['final_models_reused_within_panel']==new['final_refits'])])
    # This row-structure check deliberately does not reassert literature claims.
    literature=a.tables[('supplement',15)]['rows']
    literature_ok=(len(literature)==5 and literature[0]==['工作','数据与预处理','评价设计','输入及校准规则']
        and all(len(r)==4 and all(str(c).strip() for c in r) for r in literature[1:])
        and all(literature[i][0].startswith(prefix) for i,prefix in enumerate(['MCSWS','Li等','Wan等','本研究'],1)))
    a.compare('supplement',15,[],headers=len(literature),level='text_scope_only',
              definition_checks=[('literature rows and complete protocol fields only',literature_ok)])
    for index,key in zip([16,17,18,19],COHORTS):
        rows=[]; source=key.split('/')[0]
        arms=['D_low','D_wide'] if key.endswith('/original') else ARMS
        for arm in arms:
            for t,name in enumerate(TARGETS[source]):
                rows.append([arm.replace('_',''),name]+[f(a.metric(key,arm)[field][t]) for field in ['r2','rmse','mae','normalized_rmse']])
        a.metrics_table('supplement',index,rows)
    panels=read(root/'evidence/input_panels/panels.json'); draws=read(root/'evidence/input_panels/random_panels.json')
    rows=[]
    for key in COHORTS:
        cells=[c for c in a.cells if c['source']+'/'+c['cohort']==key]; ids={c['id'] for c in cells}
        unions=[len(panels[c['id']]['union']) for c in cells]
        grouped=defaultdict(set)
        for d in draws:
            if d['cell_id'] in ids:grouped[(d['cell_id'],d['replicate'])].update(d['subset'])
        rows.append([LABELS[key],str(cells[0]['quota']),extent(unions),extent([len(s) for s in grouped.values()])])
    a.compare('supplement',20,rows)
    rows=[]
    for key in COHORTS:
        for seed,v in a.summary['cohorts'][key]['per_seed'].items():
            b,c,base=[v[arm]['mean_r2'] for arm in ['B','C','A']]
            lo=mean(v[f'D_low:{r}']['mean_r2'] for r in range(1,11)); wi=mean(v[f'D_wide:{r}']['mean_r2'] for r in range(1,11))
            rows.append([LABELS[key],seed]+[signed(x) for x in [b-base,c-b,b-lo,c-wi]])
    a.metrics_table('supplement',21,rows)
    for index,key in zip([22,23,24,25],COHORTS):
        rows=[]; cohort=a.summary['cohorts'][key]
        for lo,wi in zip(cohort['D_draws']['D_low'],cohort['D_draws']['D_wide']):
            if lo['replicate']!=wi['replicate']:raise ValueError('D draw identities do not match')
            rows.append([str(lo['replicate']),f(lo['mean_r2']),f(wi['mean_r2']),
                signed(a.metric(key,'B')['mean_r2']-lo['mean_r2']),signed(a.metric(key,'C')['mean_r2']-wi['mean_r2'])])
        a.metrics_table('supplement',index,rows)
    lookup={(p['cell_id'],p['arm']):p for p in a.pipelines if p['arm'] in 'ABC'}
    for table,outer in [(26,31001),(27,31002)]:
        rows=[]
        for cell in a.cells:
            if cell['outer_seed']!=outer:continue
            for t,target in enumerate(cell['targets']):
                selected=[lookup[(cell['id'],arm)] for arm in 'ABC']
                rows.append([cell['fold'],target,str(selected[1]['input_counts'][t])]+[str(p['components'][t]) for p in selected])
        a.compare('supplement',table,rows)
    report=read(root/'evidence/archived_execution_verification.json')
    a.compare('supplement',28,[['流程记录',f"{report['pipeline_cases']}条，每条含4目标"],
        ['逻辑目标记录',f"{report['logical_target_models']}条，其中{new['historical_ABC_target_records_reused']}条历史A/B/C复用"],
        ['新增保存面板校准记录',f"{report['stored_model_records']}条"],
        ['逐候选重载验证分数',f"{report['candidate_records_replayed']}条"],
        ['最终模型重载最大预测差',str(report['max_prediction_reload_error'])],
        ['全部新增选择与最终模型先锁定再测试','通过' if report['all_new_choices_sealed_before_test'] else '未通过'],
        ['D每模型输入列数匹配','通过' if report['D_per_model_counts_matched'] else '未通过'],
        ['D全局采集并集匹配','未匹配，预设设计限制' if not report['D_global_budget_matched'] else '通过'],
        ['核验阶段新拟合',str(report['new_fits_during_verification'])+'次'],
        ['新的独立盲测来源','无' if not report['independent_validation_claimed'] else '有']],
        definition_checks=[('archived pipeline index count',len(a.pipelines)==report['pipeline_cases']
            and len(a.pipelines)*4==report['logical_target_models'])])
    if len(a.results)!=34 or {(r['id']) for r in a.results}!={d['id'] for d in defs['tables']}:
        raise ValueError('Not all 34 tables were checked exactly once')
    levels=dict(Counter(r['verification_level'] for r in a.results))
    source_paths={root/SNAPSHOT,root/'evidence/tables/coverage_index.json',a.summary_path}
    source_paths.update(root/p for d in defs['tables'] for p in d['preferred_sources']+d['fallback_sources'] if not p.startswith('generated/'))
    result=dict(passed=all(r['passed'] for r in a.results),table_count=34,main_tables=6,supplementary_tables=28,
        tables_checked=34,logical_body_cells_checked=sum(r['checked_body_cells'] for r in a.results),
        verification_levels=levels,generated_metric_recalculation_used=a.attested,
        summary_source=a.relative(a.summary_path),raw_reference_labels_loaded_by_this_script=False,
        model_files_loaded=False,new_model_fits=0,full_training_reproduction_claimed=False,
        all_tables_independently_recalculated_from_raw_data=False,
        scope='Published-table values/definitions traced to saved records; --use-recomputed explicitly uses prior generated archived-prediction recalculation. Verification levels are table-specific.',
        limitations=['S13 literature descriptions receive scope/structure checking only.',
                     'Saved candidates, costs, selection indices and historical execution reports are not rerun training.',
                     'Fixed-prediction resampling is not full-process uncertainty or independent validation.'],
        input_sha256={a.relative(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_paths) if p.is_file()},
        tables=a.results)
    write(generated/'tables_verification.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--generated-root',type=Path,default=None)
    parser.add_argument('--use-recomputed',action='store_true',
                        help='Use successful generated archived-prediction and resampling outputs; default checks expected records only.')
    args=parser.parse_args()
    try: report=run(args.root,args.generated_root,args.use_recomputed)
    except (ValueError,KeyError,FileNotFoundError,TypeError) as exc:
        print('TABLE RECHECK FAILED: '+str(exc),file=sys.stderr); return 2
    print(json.dumps({k:v for k,v in report.items() if k not in ['tables','input_sha256']},ensure_ascii=False,indent=2))
    if not report['passed']:
        print(json.dumps([t for t in report['tables'] if not t['passed']],ensure_ascii=False,indent=2),file=sys.stderr)
    return 0 if report['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
