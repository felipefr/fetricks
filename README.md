# fetricks
Author: Felipe Rocha, f.rocha.felipe@gmail.com, felipe.figueredo-rocha@u-pec.fr

Useful tricks and extensions for Fenics (in its newer version FenicX) and other FEM tools in Python

FE + tricks : where FE stands for Fenics and Finite Element.

This little project is born with the aim of assembling some tricks and extensions of FenicsX and other FEM-related routines I have been using in different codes. They are mostly concerning applications in Continuum Mechanics, but not only that. There are also data management functions wrapping HDF5 python implementation.

Some of the public projects that use fetricks are micmacsfenics (https://github.com/felipefr/micmacsFenics) and ddfenics (https://github.com/felipefr/micmacsFenics).

I should acknowledge the excellent tutorial of Jeremy Bleyer (https://comet-fenics.readthedocs.io/en/latest/), from which some functions have adapted/

## Installation (fetricks)
Install with : pip install . (origin directory where setup.py is located) . Don't run "python setup.py install", because it usually does not link correctly.

## Installation Fenics using conda or mamba
- basis fenicsx installation
```
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
```
- adding jupyter kernel (to be recognised by other full jupyter installation, e.g. in VScode)
```
conda install ipykernel
```

- adding some useful libraries (some libraries might include some previous others, but I want to list them for documentation purposes)
```
conda install -c conda-forge scipy scikit-learn
pip install meshio=5.3.4
```

## Citation
Please cite 
[![DOI](https://zenodo.org/badge/489339019.svg)](https://zenodo.org/badge/latestdoi/489339019) if this library has been useful in your research.

This tools have been developed to assist or during the developement of the following other libraries:
- micmacsfenics: [![DOI](https://zenodo.org/badge/341954015.svg)](https://zenodo.org/badge/latestdoi/341954015)
- deepbnd: [![DOI](https://zenodo.org/badge/341954015.svg)](https://zenodo.org/badge/latestdoi/341954015)
- ddfenics : [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7646226.svg)](https://doi.org/10.5281/zenodo.7646226)

Please consider in citing the following article if you use micmacsfenics (multimaterial, Gauss point based implicit material laws) or deepbnd (hdf5, data-management) related functions

@article{Rocha2023,
title = {DeepBND: A machine learning approach to enhance multiscale solid mechanics},
journal = {Journal of Computational Physics},
pages = {111996},
year = {2023},
issn = {0021-9991},
doi = {https://doi.org/10.1016/j.jcp.2023.111996},
url = {https://www.sciencedirect.com/science/article/pii/S0021999123000918},
author = {Felipe Rocha and Simone Deparis and Pablo Antolin and Annalisa Buffa}
}

