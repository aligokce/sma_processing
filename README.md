Spherical Microphone Array Processing Toolbox
=============================================

Summary
-------
* This repository contains a library for some of the features commonly used in spherical array microphone processing.
* These features are mostly used for **direction-of-arrival (DOA)** and **six-degrees of freedom (6DoF)** problems
* It is easy to add new features, datasets, microphones.
* This repository consists of two main interfaces: `features` as library and main script as feature extractor using config files.
* `main.py` is for extracting batch features from emulations of selected anechoic/music files on SMIR dataset over different positions and rooms.
* Batch extraction can easily be done for readily-prepared classes for datasets and microphones using integrated configuration system via `hydra`.


Supported
---------
### Features
- Spherical harmonic decomposition (SHD) [(Rafaely, 2015)](https://link.springer.com/book/10.1007/978-3-662-45664-4)
- Direct path dominance (DPD) [(Nadiri and Rafaely, 2014)](https://ieeexplore.ieee.org/abstract/document/6851936)
- Generalised Pareto distribution (GPD) fit over singular values from DPD [(Olgun and Hacihabiboglu, 2019)](http://publications.rwth-aachen.de/record/769382)
- Sparse plane wave decomposition (PWD) via orthogonal matching pursuit (OMP) using dictionary of Legendre kernels over HEALPix [(Coteli and Hacihabiboglu, 2021)](https://ieeexplore.ieee.org/document/9463766)
- Residual energy test (RENT) [(Coteli and Hacihabiboglu, 2021)](https://ieeexplore.ieee.org/document/9463766)

### SMIR datasets
- METU SPARG AIR [(Zenodo)](https://zenodo.org/record/2635758)
- BBC Maida Vale Impulse Response Dataset [(Zenodo)](https://zenodo.org/record/7267562)
- *TODO: TAU-SRIR DB [(Zenodo)](https://zenodo.org/record/6408611)*

### Microphones
- mh Acoustics [Eigenmike em32](https://mhacoustics.com/products)
- *TODO: Zylia [ZM-1](https://www.zylia.co/zylia-zm-1-microphone.html)*


How to use
----------
```py
python main.py --help
```


Tasks
-----
- [ ] Add support for real SMA recordings
- [ ] Fix fimin/fimax passing everywhere and discard empty npy portions
- [ ] Build a pipeline system
- [ ] Implement analyse functions
- [ ] wandb integration for analyse?
- [ ] Integrate room simulations


References
----------
- B. Rafaely, Fundamentals of Spherical Array Processing, vol. 8. Berlin, Heidelberg: Springer Berlin Heidelberg, 2015. doi: 10.1007/978-3-662-45664-4.
- O. Nadiri and B. Rafaely, "Localization of Multiple Speakers under High Reverberation using a Spherical Microphone Array and the Direct-Path Dominance Test," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 22, no. 10, pp. 1494-1505, Oct. 2014, doi: 10.1109/TASLP.2014.2337846.
- O. Olgun, H. Hacihabiboglu, "Data-driven Threshold Selection for Direct Path Dominance Test, Proceedings of the 23rd International Congress on Acoustics, 2019, pp. 3313–3320.
- M. B. Coteli and H. Hacihabiboglu, “Sparse Representations With Legendre Kernels for DOA Estimation and Acoustic Source Separation,” IEEE/ACM Trans. Audio Speech Lang. Process., vol. 29, pp. 2296–2309, 2021, doi: 10.1109/TASLP.2021.3091845.
