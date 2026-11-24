# Evidence records and their limits

These files contain derived evaluation records, selections, saved validation
scores, and manuscript table snapshots. They contain no raw spectra or per-sample
reference response values. The original public datasets remain separate local
inputs, subject to their provider's terms.

- `figure_data/`: the frozen inputs for Figures 1–4 and S1–S3, including all 1,884
  distinct recorded validation scores and all 216 original A/B/C component
  selections. Negative validation scores have not been removed.
- `input_panels/`: the 28 original/new evaluation cells' selected subsets and
  unions, plus the 1,120 per-target random panels shared by the narrow/wide
  comparisons. The 1,120 panels are not 1,120 independent datasets.
- `historical_ABC/`: original A/B/C metric, component, error and split records.
  Historical file references marked `archive-record:` are provenance identifiers;
  they are not promises that those physical model files are included here.
- `components/`: original selected columns, exact column-overlap records,
  full-spectrum background conditions/metrics, and search-count records.
- `fixed_prediction_resampling/`: the original 10,000 saved paired row-count
  vectors for each source, and the published percentile summaries. The weights
  contain row identifiers and multiplicities, not reference labels. Their scope
  is the original A/B/C saved predictions only. They do not quantify full-training
  uncertainty or provide intervals for D or the new Corn splits.
- `tables/published_tables.json`: the six main and 28 supplementary tables as
  editable text-cell snapshots from the November 24 manuscript package. A
  snapshot makes a reported result inspectable; it is not an independent
  recalculation. The evidence map states which checks support each table.
- `archived_execution_verification.json`: an archived report of the historical
  model-reloading audit. Running the public saved-record checker does not repeat
  that audit or establish a new independent validation result.

Run `python scripts/recheck_evidence.py` from the repository root to check the
internal arithmetic and identities of these saved records, without loading
reference labels or fitting models. With local reference data imported, run
`python scripts/recheck_resampling.py` to independently re-evaluate the original
fixed-prediction percentile summaries. See the root documentation for the unified
verification entry point and the distinction between metric, model, and training
reproduction.
