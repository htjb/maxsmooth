# maxsmooth: Derivative Constrained Function Fitting

## Introduction

**maxsmooth:** Derivative Constrained Function Fitting  
**Author:** Harry Thomas Jones Bevins  
**Version:** 2.0.0 
**Homepage:** https://github.com/htjb/maxsmooth  
**Documentation:** https://maxsmooth.readthedocs.io/

![github CI](https://github.com/htjb/maxsmooth/workflows/CI/badge.svg?event=push)  
![Test Coverage Status](https://codecov.io/gh/htjb/maxsmooth/branch/master/graph/badge.svg)  
![Documentation Status](https://readthedocs.org/projects/maxsmooth/badge/?version=latest)  
![License](https://img.shields.io/badge/license-MIT-blue.svg)  
![PyPI](https://pypi.in/v/maxsmooth/badge.svg)  
![ASCL](https://img.shields.io/badge/ascl-2008.018-blue.svg?colorB=262255)  
![JOSS](https://joss.theoj.org/papers/7f53a67e2a3e8f021d4324de96fb59c8/status.svg)  

## Installation

The software can be pip installed from the PYPI repository like so:

```bash
pip install maxsmooth
```

or alternatively it can be installed from the git repository via:

```bash
git clone https://github.com/htjb/maxsmooth.git
cd maxsmooth
pip install .
```

## Derivative Constrained Functions and `maxsmooth`

`maxsmooth` is open source Python software (Python 3+) for fitting derivative constrained functions (DCFs) such as Maximally Smooth Functions (MSFs) to data sets.

MSFs are functions with no zero crossings in derivatives of order m ≥ 2 within the domain of interest. More generally, for DCFs the minimum constrained derivative order m can take on any value or a set of specific higher order derivatives may be constrained.

They are designed to prevent the loss of signals when fitting out dominant smooth foregrounds or large magnitude signals that mask signals of interest. “Smooth” here refers to foregrounds that follow power-law structures in the band of interest. In some cases DCFs can highlight systematics in the data.

`maxsmooth` uses quadratic programming via `CVXOPT` to fit data subject to a fixed linear constraint, Ga ≤ 0, where Ga is a matrix of derivatives. The constraint on an MSF is not explicitly linear and each constrained derivative can be positive or negative. `maxsmooth` tests the ≤ 0 constraint multiplied by a positive or negative sign.

For an Nth-order polynomial `maxsmooth` can test every available sign combination, but by default it implements a sign navigating algorithm. This is detailed in the `maxsmooth` paper and summarized in the software documentation.

The available sign combinations act as discrete parameter spaces all with global minima, and `maxsmooth` is capable of finding the minimum of these by cascading followed by directional exploration. The searched region is limited to capture enough of the neighbourhood to confidently return the global minimum.

The method relies on the problem being “well defined”, but not all problems satisfy this. In such cases it is possible to test every available sign combination on the constrained derivatives. See the paper for definitions.

`maxsmooth` features a built-in library of DCFs and allows users to define their own. Inflection points and zero crossings in higher order derivatives can also be included.

## Licence and Citation

The software is free to use under the MIT open source license. For academic use please cite the `maxsmooth` papers:

**MNRAS Paper (the “maxsmooth paper”):**

> H. T. J. Bevins et al., *maxsmooth: Rapid maximally smooth function fitting with applications in Global 21-cm cosmology*, MNRAS, 2021, stab152. https://doi.org/10.1093/mnras/stab152

BibTeX:

```bibtex
@article{10.1093/mnras/stab152,
  author = {Bevins, H T J and Handley, W J and Fialkov, A and Acedo, E de Lera and Greenhill, L J and Price, D C},
  title = "{maxsmooth: rapid maximally smooth function fitting with applications in Global 21-cm cosmology}",
  journal = {Monthly Notices of the Royal Astronomical Society},
  year = {2021},
  month = {01},
  issn = {0035-8711},
  doi = {10.1093/mnras/stab152},
  url = {https://doi.org/10.1093/mnras/stab152},
  note = {stab152},
  eprint = {https://academic.oup.com/mnras/advance-article-pdf/doi/10.1093/mnras/stab152/35931358/stab152.pdf},
}
```

**JOSS Paper:**

> Bevins, H. T., (2020). *maxsmooth: Derivative Constrained Function Fitting*, Journal of Open Source Software, 5(54), 2596. https://doi.org/10.21105/joss.02596

BibTeX:

```bibtex
@article{Bevins2020,
  doi = {10.21105/joss.02596},
  url = {https://doi.org/10.21105/joss.02596},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {54},
  pages = {2596},
  author = {Harry T. j. Bevins},
  title = {maxsmooth: Derivative Constrained Function Fitting},
  journal = {Journal of Open Source Software}
}
```

## Contributing

Contributions to `maxsmooth` are welcome via:

- Opening an issue for new features or bugs.
- Submitting a pull request (ideally with prior discussion).

## Documentation

Documentation: https://maxsmooth.readthedocs.io/

You can also compile it locally by running

```bash
pip install ".[docs]"
mkdocs serve
```

