# Data sources and local import

The study uses two public archives distributed by Eigenvector Research. This repository indexes those archives and provides study-generated predictions and records. It does not mirror the original spectra or reference response arrays.

## Obtain the source archives

Use the provider's [Data Sets page](https://eigenvector.com/resources/data-sets/) as the primary entry point. The following download URLs were listed there when checked on 24 November 2026:

| Source | Provider download | Material used in this study |
|---|---|---|
| CGL_NIR | [CGL_nir.mat_.zip](https://eigenvector.com/wp-content/uploads/2021/04/CGL_nir.mat_.zip) | Four responses, 231 source rows and 117 spectral columns; the supplied 153/78 calibration/test partition is retained. |
| Corn | [corn.mat_.zip](https://eigenvector.com/wp-content/uploads/2019/06/corn.mat_.zip) | Four responses for 80 samples; only the `m5spec` instrument is used, with 700 spectral columns. |

Download through the provider and extract `CGL_nir.mat` and `corn.mat`. A changed or unavailable provider URL is not a reason to substitute a different data version silently.

## File identity

The study's source files have these SHA-256 checksums. The Corn checksum is for the extracted MAT file; no checksum is asserted here for a Corn ZIP that was not retained in the original study.

| File | SHA-256 |
|---|---|
| `CGL_nir.mat_.zip` | `d5306358cd8584f4d12bd48715ffe5bc2709a87cf1b1e7f4d0563a2729d8adc4` |
| `CGL_nir.mat` | `49b0de6188544f3e67ab7103608dc9d9e2e574d4d8e2d6b5989712f86befc9b8` |
| `corn.mat` | `e28fd4be274a54ca57b1f2c67ef5a8bf4981f8314bcc73e4c64836fe658c46b5` |

Keep the acquired source files outside the tracked repository. Import instructions and the tested command are in [the main README](README.md). The generated local reference file belongs under `data/local/`, which is excluded from version control and from the publication archive.

## Row, column and response identity

Sample IDs and selected-column indices are zero-based. CGL IDs refer to the rows of the full `Spectra`/`PropVals` blocks, rather than a new concatenation of `Ycal` and `Ytest`. The supplied calibration/test blocks must map to those full source rows. Corn IDs refer to the original row order of `m5spec` and `propvals`.

The response order is:

- CGL: Casein, Glucose, Lactate, Moisture. The source records these as wt%.
- Corn: Moisture, Oil, Protein, Starch. Errors retain the source numeric scale; this repository does not infer a dry-basis or wet-basis conversion or an application tolerance.

The CGL MAT spectral axis is retained as stored. Its final wavelength is 2496 nm in the retained file, whereas the provider's page describes an endpoint of 2495 nm. The code does not reconstruct or alter the axis from that web description.

Only one instrument's Corn measurements are used. Measurements of the same 80 samples on other instruments are not treated as additional independent datasets. The two extra Corn partitions, 31001 and 31002, reuse those same 80 samples.

## Availability and redistribution are different

The provider makes the archives downloadable. The checked data page did not identify an explicit license granting this project permission to redistribute the original spectra or reference labels. The CGL ZIP contained the MAT file and a macOS metadata file, without a separate data license. The Corn MAT information credits Cargill and acknowledges permission for Eigenvector to distribute it; that statement is not a grant of redistribution rights to this repository.

Accordingly, the publication-ready package excludes both raw spectra and reference labels. This is a conservative packaging decision, not a claim that redistribution has been legally prohibited. Users should check the provider's current terms and any conditions supplied with their download. Eigenvector's DataSet **software** license is not assumed to license the benchmark **data**.

Source reference values are required to independently recompute predictive errors. Without locally imported source values, the repository can check published-file integrity and archived evidence consistency, but cannot independently recompute R², RMSE or MAE from predictions. The instructions make this dependency explicit.

Do not commit local source files, imported reference arrays or temporary verification outputs to a public repository. See [third-party notices](THIRD_PARTY_NOTICES.md) for the scope of this release.
