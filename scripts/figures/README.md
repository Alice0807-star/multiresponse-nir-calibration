# Figure regeneration

The source is the frozen Python/Matplotlib figure implementation, adjusted only
to use repository-relative paths and to write into `generated/`. It does not
search wavelengths, fit regressions, select new components, or generate new
experimental observations.

From the repository root:

```text
python scripts/figures/recreate_all.py
```

Required plotting packages: NumPy, Matplotlib, and Pillow. The input records are
in `evidence/figure_data/`. Outputs are PDF/SVG in `generated/figures/`, PNG
previews in `generated/figure_previews/`, and derived plotting CSVs in
`generated/figure_data/`. No frozen publication file under `figures/` is replaced.

The ten publication pages are four main figures, S1, four coordinated S2 pages,
and S3. S2 is one logical figure across four pages. The script also creates a
convenience PDF combining the four S2 pages. Repeating points in a zoom view does
not add observations. Shared vertical scales, negative validation scores, and
component choices follow the archived implementation.

The original publication exports underwent numerical and visual review. A new
font, Matplotlib version, or operating system may change layout. Regenerating a
plot is not a claim that the new rendering has repeated every original layout
audit. An optional author-side `NATURE_FIGURE_QA_DIR` may point to an existing
installation of the layout auditor; that external tool is not a dependency for
data or figure reproduction and is not redistributed here. The frozen figures
are the publication reference.

The saved-record numerical checker can be run without Matplotlib:

```text
python scripts/figures/verify_numbers.py
```
