Spherical Microphone Array Processing Toolbox
=============================================

Summary
-------
* This repository contains a library for some of the features commonly used in spherical array microphone processing.
* These features are mostly used for **direction-of-arrival (DOA)** and **six-degrees of freedom (6DoF)** problems
* It is easy to add new features, datasets, microphones.
* This repository consists of two main interfaces: `features` as library and `extract` via main script and config files.
* Scripts are for extracting batch features from emulations of selected anechoic/music files on SMIR dataset over different positions and rooms.
* Batch extraction can easily be done for readily-prepared classes for datasets and microphones using integrated configuration system via `hydra`.


Supported
---------
### Features
- Spherical harmonic decomposition (SHD)
- Direct path dominance (DPD)
- Generalised Pareto distribution (GPD) fit over singular values from DPD
- *TODO: Steered response power (SRP)*
- *TODO: Sparse plane wave decomposition (PWD) via orthogonal matching pursuit (OMP) using dictionary of Legendre kernels over HEALPix*
- *TODO: Residual energy test (RENT)*

### SMIR datasets
- METU SPARG Air [reference here]
- *TODO: TAU-SRIR DB*

### Microphones
- mh Acoustics Eigenmike em32
- *TODO: Zylia ZM-1*


Available experiment parameters
-------------------------------
### `experiment.job`
* `extract.<job>: ` `shd`, `dpd_ratio`, `pareto_params`, `legendre_count`

### `experiment.positions`
* `'all'`, `'perpendicular'`, `[ [ 3, 2, 2 ], ... ]`
