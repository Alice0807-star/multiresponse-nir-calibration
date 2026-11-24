# Reproduction scope and numerical rules

The purpose of this package is to let a reader trace and check the reported fifth-version results. The study's model searches are complete and frozen. None of the default verification entry points selects wavelengths, adjusts a hyperparameter, fits a model or opens a new test experiment.

## Levels of verification

### Package and evidence integrity

`python scripts/verify.py` checks the distributed files and archived evidence without requiring the source benchmark arrays. Hashes can detect missing or changed files. Record-level checks can test identities, counts, candidate choices and consistency of retained summaries. They do not independently recompute errors when the reference labels are absent.

### Saved-prediction metrics

After local data import, `python scripts/recheck_metrics.py --help` describes the metric-recalculation entry point. The full combined entry point is `python scripts/verify.py --all`.

Predictions are paired with local source reference values by source, sample ID and response identity. The evaluation index preserves the partition, fold, selector seed, input configuration and random-workflow number. A recheck must not infer row identity from incidental file order or use rounded Word-table values as its source observations.

### Saved-model prediction

The saved-model check uses archived numerical model parameters and locally imported spectra to recover predictions. Its supported entry point is documented by `python scripts/recheck_models.py --help`. It performs prediction only. Model prediction agreement and saved-prediction metric agreement are reported as separate checks.

These checks are narrower than fitting a fresh PLS model or repeating a complete wavelength search. A fitted parameter archive is not itself a claim that every training trajectory has been reproduced on a new device.

### Historical training implementation

The historical implementation, frozen configuration and recorded scientific environment support provenance and inspection. The package does not claim that a fresh end-to-end training run was performed while assembling this release, or that cross-version/cross-device training equivalence was established.

The original scientific environment was Python 3.10.21, NumPy 2.1.2, scikit-learn 1.4.2 and PyTorch 2.6.0+cu124. The PLS computations used the CPU. The lighter verification environment is declared separately in the requirement files; a CUDA installation is not a prerequisite for the no-fit checks.

## Predictive metrics

For each response, R², RMSE and MAE use the original response scale. RMSE is the square root of the mean squared residual. MAE is the mean absolute residual. The R² denominator uses the reference values for the complete applicable evaluation set.

Training-SD-normalized RMSE divides each residual by the corresponding response's SD in that prediction's outer training partition, including fitting F and validation V rows. This SD uses `ddof=1`. The squared normalized residuals are then pooled over the complete evaluation rows before taking their mean and square root. Test-set SD is not substituted for training SD.

CGL uses its 78 official test rows. Corn uses all 80 out-of-fold predictions for each complete partition/selector-seed/random-workflow combination. Test IDs must cover 0–79 once in that combination. Fold R² values are not averaged to obtain the Corn R².

## Aggregation order

The following order is part of the protocol:

1. Compute each response's metrics from the complete test or OOF predictions for one algorithm seed and one random workflow where applicable.
2. On original CGL and Corn settings, average the response metrics over selector seeds 17, 29 and 43. The new Corn partitions 31001 and 31002 use seed 17 only.
3. Macro-average R² by giving each of the four responses equal weight.
4. For D, retain all ten complete random-workflow results and then average those metrics across the ten workflows. A workflow spans all responses and all outer folds in its setting.

Predictions are not first averaged to form an ensemble. The 50 Corn fold/workflow combinations are not concatenated and counted as new independent samples. The three selector seeds and the ten random workflows are not pooled as 30 independent data-source repetitions.

## Added table statistics

Table 3 reports the sample SD of the ten D workflow-level macro R² values, with denominator nine (`ddof=1`). For the original settings, each workflow-level value already contains the three-seed metric average. For new Corn partitions it uses seed 17.

This SD describes variation among the fixed random-extension workflows conditional on the retained data, partitions and selection rules. It is not a standard error, confidence interval, model-search uncertainty estimate or measure of new-source generalization. A/B/C seed means do not have the same variability interpretation as the D workflow SD.

At a fixed setting, B or C is constant across the ten D workflows. Consequently, the sample SD of B−Dlow equals the sample SD of Dlow; the corresponding C−Dwide identity also holds. Table 4 does not repeat the same SD as a new source of uncertainty evidence.

Table 6 calculates relative RMSE change as:

```text
100 * (aggregated_RMSE_of_B_or_C / aggregated_RMSE_of_A - 1)
```

The ratio is calculated after the specified seed/workflow metric aggregation, using unrounded values. Positive changes indicate higher error. They are not the average of per-run percentage ratios and do not represent significance tests. RMSE ratios in Figure 4 use the same aggregation-before-ratio rule.

## Paired comparisons and saved resampling

The principal random-reference differences are B−mean(Dlow) and C−mean(Dwide). Each of the ten D values represents a complete workflow, not an independent material source. Counts of positive paired differences are descriptive counts, not p-values or independent-source win rates.

The descriptive range difference is `(C−B) − mean(Dwide−Dlow)`. It is not a causal contribution fraction or a unique mechanism decomposition.

The retained 10,000 fixed-prediction resamples apply to the original A/B/C analysis only. They do not reselect features or refit a model, do not remove cross-fold training overlap and are not advertised as full-pipeline 95% confidence intervals. They are not silently extended to D or the new Corn partitions.

The independent entry point is `python scripts/recheck_resampling.py`. It accepts `--reference-data` for an alternative local imported NPZ and `--output` for a report path. Without those overrides it reads `data/local/reference_data.npz` and writes `generated/resampling_report.json`.

## Data and index checks

The local importer checks the values, dimensions and row order of the source arrays against their declared identities. CGL uses full `Spectra`/`PropVals` row IDs. Corn uses source m5 row IDs. Selected-column IDs and sample IDs are zero-based.

Fitting, validation and test IDs are retained separately. New Corn training partitions contain newly searched target subsets; the old selected subsets are not substituted into a new partition. Overlap between the different full Corn partitions is expected because they reuse the same 80 samples. That reuse is disclosed rather than interpreted as independent-source validation.

The frozen 644 pipeline cases and 2,576 logical target-model records include historical reuse. The 1,200 stored calibration-panel records, 17,840 saved candidate records, 2,229 new final refits and 131 within-panel final-model reuses are different counting units. File counts are not substituted for those protocol counts.

## Figures and tables

Published table values must be reproduced with their declared rounding, while numerical comparisons use the unrounded archive values and the tolerances stated by each check. Any failed or skipped check must appear separately from passed checks.

Plot regeneration uses the retained figure sources and archived plotting data. The exact fonts and PDF/SVG rendering can vary with the local platform. Numerical agreement does not imply byte-identical figure files after regeneration. Check the rendered labels, common axes, negative values, legends and captions after changing a plotting environment.

Install `requirements-figures.txt` and run `python scripts/figures/recreate_all.py` to regenerate the ten pages. The archived figures remain in `figures/`; new PDF/SVG outputs, PNG previews, plotting CSVs and the available figure checks are written under `generated/`. The standard entry point does not require the original author's local layout-helper installation. This portability is not a claim that every original author-layout audit has been independently repeated by every reader.

The component curves contain saved validation scores. A wider candidate range cannot lower the best retained validation score merely because it contains the narrower range, but this does not establish better test prediction or that 15 components is globally optimal. The component-choice and exact-column-overlap records are descriptive evidence, not proofs of search convergence, stable chemical wavelength identity or a new selection method.

## What successful execution does not establish

The comparisons were developed after the original test results were known. Both selection stages reuse the same validation partition. The two new Corn partitions do not add new samples, instruments or material sources. D controls per-model input count and candidate range, not correlation structure, effective degrees of freedom, global acquisition count or search cost.

Execution checks do not resolve those study-design limits. They also do not erase the retained adverse response-level results, provide reference-assay repeatability or establish an application-specific measurement tolerance.
