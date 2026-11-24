# Source data stay local

This directory intentionally contains no original spectra or per-sample reference
responses. Obtain CGL and Corn from the official source described in
`../DATA_SOURCES.md`, and import the files using `scripts/import_reference_data.py`.

The resulting `local/reference_data.npz` and import report are private runtime
files, excluded from the repository and release ZIP. CGL IDs index the original
231-row `Spectra` / `PropVals` order. Corn IDs index the 80 rows of `m5spec` /
`propvals`. Do not concatenate CGL calibration and test arrays to reconstruct IDs.
