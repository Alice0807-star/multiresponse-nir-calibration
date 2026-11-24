# Historical training implementation

`training_source/` contains unchanged historical entry points and their local
module dependencies, with file hashes in `source_manifest.json`. They are retained
for method inspection and traceability. Some dependencies also implement older
research branches; these branches are not part of the V5 reported comparisons.

The package verification commands never invoke these files. They are not a
turnkey, certified clean-start training pipeline: the historical drivers expect
their original staged inputs, historical results and protocol manifests. A new
full search/fitting reproduction would need a separately configured output and
an explicit reproduction protocol. No such new training was performed here.

For supported portable checks, use `../../scripts/verify.py`. Final numeric PLS
parameters, rather than pickle/Torch checkpoints, are distributed under
`../../results/models/`. Their predictions can be independently replayed on
locally imported source spectra.
