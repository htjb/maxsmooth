---
title: 'maxsmooth: Derivative Constrained Function Fitting'
tags:
  - Python
  - astrophysics
  - cosmology
authors:
  - name: Harry T. J. Bevins
    orcid: 0000-0002-4367-3550
    affiliation: "1"
affiliations:
 - name: Astrophysics Group, Cavendish Laboratory, J.J.Thomson Avenue, Cambridge, CB3 0HE, UK
   index: 1
date: 31 July 2020
bibliography: paper.bib

---

# Summary

``maxsmooth`` is an optimisation routine for fitting derivative constrained functions (DCFs)
to data sets. DCFs are a family of functions which have derivatives that do not cross
zero in the band of interest. Consequently, they can produce perfectly smooth fits to
data sets and reveal non-smooth signals of interest in the residuals. ``maxsmooth``
utilises the [CVXOPT](https://pypi.org/project/cvxopt/) implementation of
quadratic programming to perform the constrained fitting rapidly and efficiently.

# Foreground Modelling in 21-cm Cosmology

The development of ``maxsmooth`` has largely been motivated by the problem
of foreground modelling in Global 21-cm experiments. [@EDGES_LB; @LEDA; @SARAS; @REACH].
Global 21-cm cosmology is the study of the spin temperature of hydrogen gas and
its relative magnitude when compared to the Cosmic Microwave Background (CMB)
during the Cosmic Dawn and Epoch of Reionisation. During this period in the Universe's
history the first stars began to form and the properties of the gas in the
Universe changed as it interacted with radiation from the first luminous sources
[@Furlanetto2006; @Pritchard2012; @Barkana2016].

The goal of Global 21-cm experiments is to detect this
structure in the sky averaged radio spectrum between approximately $50$ and $200$ MHz.
However, the signal of interest is expected to be of order $\leq 250$ mK and masked by
foregrounds $10^4 - 10^5$ orders of magnitude brighter [@Cohen; @Cohen2019].

Modelling and removal of the high magnitude foregrounds is essential for detection of
the Global 21-cm signal. The foregrounds are predominantly composed of smooth
spectrum synchrotron and
free-free emission from the Galaxy and extragalactic radio sources. Consequently, DCFs
provide a powerful alternative to standard polynomials for accurately modelling the
foregrounds in such experiments.

While ``maxsmooth`` has been designed with Global 21-cm cosmology in mind it is
equally applicable to any experiment in which the signal of interest has to be separated
from comparatively high magnitude smooth signals or foregrounds.

# ``maxsmooth``

DCFs can be fitted with routines such as Basin-hopping [@Basinhopping] and
Nelder-Mead [@Nelder-Mead] and this has
been the practice when fitting Maximally Smooth Functions (MSFs), a specific form of
DCF, for 21-cm Cosmology [@MSFCD; @MSF-EDGES]. However, ``maxsmooth`` employs quadratic programming to
rapidly and efficiently fit DCFs which are constrained such that

$$  \frac{d^my}{dx^m}\geq0 ~~ \textnormal{or} ~~ \frac{d^my}{dx^m}~\leq0, $$

where for MSFs $m$, the derivative order, is $\geq 2$. An example DCF from the library of $7$
built-in to ``maxsmooth`` is given by

$$ y ~ = ~ \sum_{k=0}^{N} ~ a_{k} ~ x^k, $$

where $x$ and $y$ are the independent and dependent variables respectively and $N$
is the order of the fit with powers from $0 - (N-1)$. The library is not intended
to be complete and will be extended by future contributions from users.
We find that the use of quadratic programming makes ``maxsmooth``
approximately two orders of magnitude quicker than a Basin-hopping/Nelder-Mead approach.
We find that ``maxsmooth`` can fit higher order fits to a higher degree of quality, lower $\chi^2$,
without modification of algorithm parameters as is needed when using Basin-hopping.

``maxsmooth`` rephrases the above condition such that

$$ \pm_m ~ \frac{d^my}{dx^m} ~ \leq0, $$

where the $\pm$ applies to a given $m$. This produces a set of discrete sign spaces
with different combinations of constrained positive and negative derivatives. In each sign space
the associated minimum is found using quadratic programming and then ``maxsmooth``
identifies the optimum combination of signs, $\mathbf{s}$ on the derivatives. To
summarise the minimisation problem we have

$$\min_{a,~s}~~\frac{1}{2}~\mathbf{a}^T~\mathbf{Q}~\mathbf{a}~+~\mathbf{q}^T~\mathbf{a},$$
$$\mathrm{s.t.}~~\mathbf{G(s)~a} \leq \mathbf{0},$$

where we are minimising $\chi^2$, $\mathbf{G(s)a}$ is a stacked matrix of derivative evaluations and $\mathbf{a}$
is the matrix of parameters we are attempting to optimise for. $\mathbf{Q}$ and $\mathbf{q}$
are given by

$$\mathbf{Q}~=~ \mathbf{\Phi}^T~\mathbf{\Phi}~~\textnormal{and}~~ \mathbf{q}^T~=~-\mathbf{y}^T~\mathbf{\Phi},$$

here $\mathbf{\Phi}$ is a matrix of basis function evaluations and $\mathbf{y}$
is a column matrix of the dependent data points we are fitting.

The discrete spaces can be searched in their entirety quickly and efficiently or
alternatively a sign navigating algorithm can be invoked using ``maxsmooth`` which
reduces the runtime of the algorithm. Division of the parameter space into
discrete sign spaces allows for a more complete exploration of the parameter space
when compared to Basin-hopping/Nelder-Mead based algorithms.

The sign navigating approach uses a cascading algorithm to identify a candidate set of signs
and parameters for the global
minimum. The cascading algorithm starts with a randomly generated set of signs. Each
individual sign is then flipped, from the lowest order derivative first, until the
objective function decreases in value. The signs associated with the lower
$\chi^2$ value then become the optimum set and the processes is repeated until
the objective function stops decreasing in value. This is then followed by a limited exploration
of the neighbouring sign spaces to identify the true global minimum.

![The time taken to fit a polynomial data set following an approximate $x^a$ power law
using both ``maxsmooth`` quadratic programming methods and for comparison a method
based in Basin-hopping and Nelder-Mead routines. We show the results using the later method
up to $N = 7$ after which the method begins to fail without adjustments to the routine parameters.
For $N = 3 - 7$ we find a maximum difference of $0.04\%$ between the optimum ``maxsmooth`` $\chi^2$
values and the Basin-hopping results. Figure taken from @Bevins.](times.png)

Documentation for ``maxsmooth`` is available at [ReadTheDocs](maxsmooth.readthedocs.io/)
and the code can be
found on [Github](https://github.com/htjb/maxsmooth). Version 1.1.0 is available
as a [PyPI](https://pypi.org/project/maxsmooth/1.1.0/) package. Continuous
integration is performed with [Travis](https://travis-ci.com/github/htjb/maxsmooth) and the
associated code coverage can be found at [CodeCov](https://codecov.io/gh/htjb/maxsmooth).

# Acknowledgements

Discussions on the applications of the software were provided by Eloy de Lera Acedo,
Will Handley and Anastasia Fialkov. The author is supported by the Science and
Technology Facilities Council (STFC) via grant number ST/T505997/1.

# References
