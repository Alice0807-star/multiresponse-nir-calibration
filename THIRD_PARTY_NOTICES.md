# Rights and third-party notices

This local repository is being prepared for public review alongside an empirical near-infrared spectroscopy manuscript. It has not been uploaded or published by this preparation task.

## Original study material

The scripts, study-generated predictions, selected-variable indices, summaries and figure source files assembled here are research outputs from this project. The authors have not yet selected a repository-wide public license. No MIT, CC-BY or other blanket license is asserted by this preparation task. Public access to files, once enabled by the authors, does not by itself specify permission for every form of reuse.

Before release, the author team should select a license appropriate to its own code and outputs, identify the corresponding copyright holder, and ensure that the license scope excludes third-party material for which it cannot grant rights. This decision does not change the scientific results or the ability to inspect the local package.

## Source benchmark data

The source CGL_NIR and Corn archives are distributed by [Eigenvector Research](https://eigenvector.com/resources/data-sets/). Provider credits and file identities are recorded in [DATA_SOURCES.md](DATA_SOURCES.md). No original spectral arrays, reference response arrays, source MAT files or provider ZIP files are included in the public package.

An explicit third-party data redistribution grant was not confirmed in the checked provider page and retained archives as of 24 November 2026. This is recorded as an unresolved permission scope, rather than a conclusion that the provider forbids every redistribution. Acquire the data from the provider and retain any applicable terms or permissions.

## Software dependencies

The package refers to external scientific Python libraries rather than copying their implementations. Install them from their normal distributions and consult each library's own license. Relevant libraries include NumPy, SciPy, Matplotlib, Pillow, scikit-learn and PyTorch. Not all are needed for the default evidence checks; requirements are separated by the supported reproduction task.

The scikit-learn and PyTorch implementations are not relicensed as this project's code. Historical model-loading or training environments, where documented, remain distinct from the lightweight saved-prediction evaluation environment.

## Related research

The manuscript's related-work references are scholarly attribution, not licenses to redistribute those papers' code, datasets or figures. This package does not include third-party reference-figure images or claim that the authors originated shared-input modeling, PLS regression or feature-selection stability methods.

## Release identity

Use an immutable Git tag or archived release when citing a public version. A real repository URL, release identifier or DOI should be inserted only after it exists. This preparation task has not invented a repository owner, citation DOI, author list or funding information.
