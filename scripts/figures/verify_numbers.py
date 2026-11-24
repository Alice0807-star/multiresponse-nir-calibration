"""Independent CSV/record cross-checks. No experiment or model execution."""
from pathlib import Path
import csv
import hashlib
import json
import math
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"evidence"/"figure_data"
QA=ROOT/"generated"/"figure_qa"
QA.mkdir(parents=True,exist_ok=True)


def rows(name):
    with (DATA/(name+".csv")).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def average(values):return sum(values)/len(values)


def main():
    summary=json.loads((DATA/"summary.json").read_text(encoding="utf-8"))
    curves=json.loads((DATA/"candidate_curves.json").read_text(encoding="utf-8"))
    members=json.loads((DATA/"figure1_membership.json").read_text(encoding="utf-8"))
    errors=[]
    def same(a,b):
        error=abs(float(a)-float(b));errors.append(error);assert error<1e-12,(a,b)
    effects=rows("Figure_2_paired_differences")
    assert len(effects)==80
    for row in effects:
        cohort=summary["cohorts"][row["cohort"]]
        seeds=cohort["algorithm_seeds"]
        union=average([cohort["per_seed"][str(s)][row["union_method"]]["mean_r2"] for s in seeds])
        random=average([cohort["per_seed"][str(s)][row["random_method"]+":"+row["workflow"]]["mean_r2"] for s in seeds])
        same(row["delta_macro_r2"],union-random)
    for row in rows("Figure_2_effect_summary"):
        values=[float(x["delta_macro_r2"]) for x in effects if x["cohort"]==row["cohort"] and
                x["union_method"]==row["contrast"].split(" minus ")[0]]
        same(row["mean_delta_macro_r2"],average(values))
        assert int(row["positive"])==sum(v>0 for v in values)
        assert int(row["equal"])==sum(v==0 for v in values)
        assert int(row["negative"])==sum(v<0 for v in values)
    for row in rows("Figure_3_partition_differences"):
        assert int(row["selection_seed"])==17
        cohort=summary["cohorts"]["corn/"+row["cohort"]]["per_seed"]["17"]
        if row["contrast"]=="B − A":expect=cohort["B"]["mean_r2"]-cohort["A"]["mean_r2"]
        elif row["contrast"]=="C − B":expect=cohort["C"]["mean_r2"]-cohort["B"]["mean_r2"]
        else:expect=cohort["C"]["mean_r2"]-average([cohort[f"D_wide:{i}"]["mean_r2"] for i in range(1,11)])
        same(row["delta_macro_r2"],expect)
    targets={"cgl":["Casein","Glucose","Lactate","Moisture"],"corn":["Moisture","Oil","Protein","Starch"]}
    for row in rows("Figure_4_rmse_ratios"):
        i=targets[row["source"]].index(row["target"])
        co=summary["cohorts"][row["source"]+"/original"]
        def mean_rmse(method):
            return average([co["per_seed"][str(s)][arm]["rmse"][i]
                            for s in co["algorithm_seeds"] for arm in
                            ([f"{method}:{r}" for r in range(1,11)] if method.startswith("D_") else [method])])
        same(row["ratio"],mean_rmse(row["numerator"])/mean_rmse(row["denominator"]))
    assert len(rows("Figure_4_rmse_ratios"))==24
    assert len(rows("Figure_4_absolute_rmse"))==40
    assert sorted(set(v for own in members["own"] for v in own))==members["union"]
    shared=Counter(v for own in members["own"] for v in own)
    assert sum(n==2 for n in shared.values())==6
    assert len(rows("Figure_1_observed_membership"))==42
    choices=rows("Figure_S3_selected_components");assert len(choices)==216
    for row in choices:
        r=next(c for c in curves if c["source"]==row["source"] and c["fold"]==row["fold"] and
               c["target_name"]==row["target"] and c["seed"]==int(row["seed"]))
        assert int(row["selected_components"])==r[row["arm"]+"_components"]
        key="A_trials" if row["arm"]=="A" else "union_trials"
        allowed=[v for v in r[key] if v["components"]<=int(row["allowed_max"])]
        best=max(sorted(allowed,key=lambda v:v["components"]),key=lambda v:v["mean_r2"])
        assert best["components"]==int(row["selected_components"])
    coverage=json.loads((DATA/"curve_coverage_and_limits.json").read_text(encoding="utf-8"))
    assert sum(v["curves"] for v in coverage)==144
    assert sum(v["distinct_scores"] for v in coverage)==1884
    negative=sum(v["mean_r2"]<0 for r in curves for key in ["A_trials","union_trials"] for v in r[key])
    report=dict(passed=True,paired_effects=80,partition_differences=9,rmse_ratios=24,absolute_rmse_values=40,
       membership_columns=42,shared_columns=6,distinct_validation_curves=144,distinct_validation_scores=1884,
       negative_validation_scores_retained=negative,actual_component_choices=216,max_absolute_numeric_error=max(errors),
       new_model_fits=0)
    (QA/"numeric_verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
