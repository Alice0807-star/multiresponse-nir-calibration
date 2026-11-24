# Target wavelength union reuse in multiresponse NIR calibration

This repository accompanies the fifth manuscript version of an empirical study of target-specific wavelength inputs, their union, and PLS component ranges. It contains the study's code, data-source index, saved predictions, model parameters, evidence records and figure sources. Original benchmark spectra and reference labels are obtained separately from their provider.

The package supports independent checks of the reported results without rerunning wavelength searches or fitting models. It distinguishes file-integrity checks, saved-prediction recalculation, saved-model prediction checks and the historical training-source archive. Passing one level does not establish all the others.

## Study scope

The data sources are CGL_NIR and Corn from the [Eigenvector Research archive](https://eigenvector.com/resources/data-sets/). Each has four responses. Corn uses the m5 instrument only; its extra partitions reuse the same 80 samples.

| Configuration | Input to response model j | PLS component candidates |
|---|---|---|
| A | Its selected subset S_j | CGL 1–12; Corn 1–11 |
| B | Union U of the four target subsets | Same narrow range as A |
| C | The same U | 1–15 |
| Dlow | S_j retained and randomly extended to the size of U | Same range as B |
| Dwide | The same random extension as Dlow | Same range as C |

The primary random-reference comparisons are B versus Dlow and C versus Dwide. A versus B addresses input visibility; B versus C addresses the allowed component range. Random extensions match each model's input count and the corresponding candidate range. They do not match the global acquisition union, correlation structure, effective degrees of freedom or search cost.

This is a bounded, post-specified analysis of two public sources, not a new wavelength-selection algorithm or an independent blind validation. Results include adverse response-level effects and partition sensitivity. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the scope of the checks.

## Quick start

Use Python 3.10 or later. Run these commands from the repository root. The commands do not start training, download data or require a GPU.

### 1. Check the distributed files and archived evidence

```console
python -m pip install -r requirements-recheck.txt
python scripts/verify.py
```

This level checks the package and the internal consistency of its archived evidence. It does **not** independently recompute predictive errors without locally imported reference labels.

### 2. Import the provider data locally

Obtain and extract the official source files following [DATA_SOURCES.md](DATA_SOURCES.md). The importer accepts either each extracted MAT file or its provider ZIP. For example, replace the two local paths below with your downloaded files:

```console
python -m pip install -r requirements-import.txt
python scripts/import_reference_data.py --cgl "../source-data/CGL_nir.mat" --corn "../source-data/corn.mat"
```

The importer checks source-array values and row order, then creates `data/local/reference_data.npz`. This local file contains source spectra and labels and is excluded from version control and the public release archive. Do not upload it. Importing the data does not select wavelengths or fit a model.

### 3. Run the full supported no-fit verification

```console
python scripts/verify.py --all
```

This level adds saved-prediction metric recalculation, prediction checks using saved model parameters, and the original fixed-prediction resampling checks. It requires the imported reference file. A missing file is reported as a missing prerequisite, not silently treated as a passed metric check. Generated reports are written under `generated/` and do not overwrite archived expectations.

For the arguments supported by each independent entry point:

```console
python scripts/recheck_metrics.py --help
python scripts/recheck_models.py --help
```

Individual evidence and original-resampling checks can also be run directly:

```console
python scripts/recheck_evidence.py
python scripts/recheck_resampling.py
```

The resampling command requires the imported local reference data and remains restricted to the original A/B/C analysis.

### 4. Optionally regenerate the ten figure pages

```console
python -m pip install -r requirements-figures.txt
python scripts/figures/recreate_all.py
```

The archived PDF/SVG figures are in `figures/`. Regeneration writes separate files under `generated/`, including vector figures, previews, plotting data and the available figure checks. It does not overwrite the archived figures. Numerical agreement and scientific content are checked separately from fonts and renderer-specific appearance; regenerated pages are not automatically asserted to have passed every original author-layout audit.

## Coverage

The evidence index links the six main tables, 28 supplementary tables, main Figures 1–4 and supplementary Figures S1–S3 to their package inputs. The [published table snapshot](evidence/tables/published_tables.json) records all 34 tables; a snapshot is distinguished from a metric recalculation from predictions. Figure S2 has four coordinated pages, so the seven logical figures occupy ten figure pages.

Coverage includes:

- Original CGL and Corn A/B/C records for selector seeds 17, 29 and 43.
- Dlow/Dwide from ten complete random-extension workflows per setting.
- Corn partitions 31001 and 31002, each using selector seed 17 and newly selected subsets within its training partitions.
- The original full-spectrum PLS background, component validation curves, actual component choices, exact-column overlap records and calculation-cost records.
- The original A/B/C fixed-prediction resampling results; these are not extended to D or the new partitions.
- The 24 November table additions: eight D-workflow sample SDs and eight B-versus-A RMSE percentage changes.

The frozen design contains 644 pipeline records and 2,576 logical response-model records. Those counts include reuse and dependent comparisons; they are not counts of independent experiments, physical model files or independent samples.

## How to interpret verification

| Check | What it establishes | What it does not establish |
|---|---|---|
| Package/evidence check | Distributed files are intact and archived records agree with the declared expectations. | Independent predictive-error recalculation without reference data. |
| Metric recalculation | Saved predictions, local source labels and the declared aggregation rules reproduce the reported numbers within the stated tolerances. | Wavelength-search or fitting reproducibility from scratch. |
| Saved-model prediction check | Archived model parameters reproduce the corresponding saved predictions on the imported inputs. | A new training run, independent data-source validation or unbiased study design. |
| Historical source archive | The retained implementation and environment are inspectable. | Verified complete retraining on every new device/software combination. |

## Data and software rights

See [DATA_SOURCES.md](DATA_SOURCES.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Availability from the provider is not assumed to grant this project permission to mirror spectra or reference labels. No repository-wide software/data license, author identity, public URL or DOI is invented by this preparation package. The author team must finalize those release details before public submission.

## Reproduction reports

Consult the bundled verification reports and their recorded scope for what was actually exercised during package acceptance. A successful report is an execution and numerical-consistency check, not evidence that union inputs universally improve every response or transfer to new instruments/materials.
