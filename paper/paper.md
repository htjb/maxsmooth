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
 - name: Astrophysics Group, Cavendish Laboratory, J.J.Thomson Avenue, Cambridge, CB3 0HE, United Kingdom
   index: 1
date: 31 July 2020
bibliography: paper.bib

---

# Summary

``maxsmooth`` is an optimisation routine written in Python (supporting version $\geq 3.6$)
for fitting Derivative Constrained Functions (DCFs) to data.
DCFs are a family of functions which have derivatives that do not cross
zero in the band of interest. Two special cases of DCF are Maximally Smooth Functions
(MSFs) which have derivatives with order $m \geq 2$ constrained and Completely Smooth
Functions (CSFs) with $m \geq 1$ constrained. Alternatively, we can constrain an
arbitrary set of derivatives and we
refer to these models as Partially Smooth Functions.
Due to their constrained nature, DCFs can produce perfectly smooth fits to
data and reveal non-smooth signals of interest in the residuals.

# Statement of Need

The development of ``maxsmooth`` has been motivated by the problem
of foreground modelling in Global 21-cm experiments [@EDGES_LB; @LEDA; @SARAS; @REACH].
Global 21-cm cosmology is the study of the spin temperature of hydrogen gas and
its relative magnitude compared to the Cosmic Microwave Background
during the Cosmic Dawn (CD) and Epoch of Reionisation (EoR). During the CD and EoR
the first stars formed and the properties of the hydrogen gas
changed as it interacted with radiation from these stars
[@Furlanetto2006; @Pritchard2012; @Barkana2016].

The goal of Global 21-cm experiments is to detect this
structure in the sky averaged radio spectrum between $\nu = 50$ and $200$ MHz.
However, the signal of interest is expected to be approximately $\leq 250$ mK and masked by
foregrounds $10^4 - 10^5$ times brighter [@Cohen; @Cohen2019].

Modelling and removal of the foreground is essential for detection of
the Global 21-cm signal. DCFs provide a powerful alternative to unconstrained
polynomials for accurately modelling the smooth synchrotron/free-free emission
foreground from the Galaxy and extragalactic radio sources.

To illustrate the abilities of ``maxsmooth`` we produce mock
21-cm experiment data and model and remove the foreground using an MSF.
We add to a mock foreground, $\nu^{-2.5}$,
a Gaussian noise with standard deviation of $0.02$ K and a
Gaussian signal with amplitude $0.23$ K, central frequency of $100$ MHz
and standard deviation of $10$ MHz.

The figure below shows the residuals (bottom panel, green) when fitting
and removing an MSF from the data (top panel) compared to the injected signal
(bottom panel, red). While the removal of the foreground does not
directly recover the injected signal, rather a smooth baseline subtracted version,
we see the remnant of the signal in
the residuals (see @Bevins for more details and examples).

![**Top panel:** The mock 21-cm data used to illustrate the abilities of
``maxsmooth``. **Bottom Panel:** The residuals, green, when removing an MSF
model of the 21-cm foreground from the data showing a clear remnant of
the signal, red. When jointly fitting an MSF and signal model we find that
we can accurately recover the signal itself (see @Bevins).](example.png)

``maxsmooth`` is applicable to any experiment in which the signal of interest
has to be separated
from comparatively high magnitude smooth signals or foregrounds.

# ``maxsmooth``

DCFs can be fitted with routines such as Basin-hopping [@Basinhopping] and
Nelder-Mead [@Nelder-Mead] and this has
been the practice for 21-cm cosmology [@MSFCD; @MSF-EDGES].
However, ``maxsmooth`` employs quadratic programming via
[CVXOPT](https://pypi.org/project/cvxopt/) to
rapidly and efficiently fit DCFs which are constrained such that

$$  \frac{d^my}{dx^m}\geq0 ~~ \textnormal{or} ~~ \frac{d^my}{dx^m}~\leq0. $$

An example DCF from the ``maxsmooth`` library is given by

$$ y ~ = ~ \sum_{k=0}^{N} ~ a_{k} ~ x^k, $$

where $x$ and $y$ are the independent and dependent variables respectively and $N$
is the order of the fit with powers from $0 - (N-1)$. The library is intended
be extended by future contributions from users.
We find that the use of quadratic programming makes ``maxsmooth``
approximately two orders of magnitude quicker than a Basin-hopping/Nelder-Mead approach.

``maxsmooth`` rephrases the above condition such that

$$ \pm_m ~ \frac{d^my}{dx^m} ~ \leq0, $$

where the $\pm$ applies to a given $m$. This produces a set of sign spaces
with different combinations of constrained positive and negative derivatives. In each sign space
the associated minimum is found using quadratic programming and then ``maxsmooth``
identifies the optimum combination of signs, $\mathbf{s}$. To
summarise the minimisation problem we have

$$\min_{a,~s}~~\frac{1}{2}~\mathbf{a}^T~\mathbf{Q}~\mathbf{a}~+~\mathbf{q}^T~\mathbf{a},$$
$$\mathrm{s.t.}~~\mathbf{G(s)~a} \leq \mathbf{0},$$

where we are minimising $\chi^2$, $\mathbf{G(s)a}$ is a stacked matrix of derivative evaluations and $\mathbf{a}$
is the matrix of parameters we are optimising for. $\mathbf{Q}$ and $\mathbf{q}$
are given by

$$\mathbf{Q}~=~ \mathbf{\Phi}^T~\mathbf{\Phi}~~\textnormal{and}~~ \mathbf{q}^T~=~-\mathbf{y}^T~\mathbf{\Phi},$$

here $\mathbf{\Phi}$ is a matrix of basis function evaluations and $\mathbf{y}$
is a column matrix of the dependent data.

The discrete spaces can be searched in their entirety quickly and efficiently or
a sign navigating algorithm can be invoked using ``maxsmooth``
reducing the fitting time. Division of the parameter space into
sign spaces allows for a more complete exploration
when compared to Basin-hopping/Nelder-Mead based algorithms.

The sign navigating approach uses a cascading algorithm to identify a candidate
optimum $\mathbf{s}$ and $\mathbf{a}$. The algorithm starts with a randomly generated $\mathbf{s}$. Each
individual sign is then flipped, from the lowest order derivative first, until the
objective function decreases in value. The signs associated with the lower
$\chi^2$ value become the optimum set and the process is repeated until
$\chi^2$ stops decreasing. This is followed by a limited exploration
of the neighbouring sign spaces to identify the true global minimum.

![The time taken to fit polynomial data following an approximate $x^a$ power law
using both ``maxsmooth`` quadratic programming methods and for comparison a method
based in Basin-hopping and Nelder-Mead routines. We show the results using the later method
up to $N = 7$ after which the method begins to fail without adjustments to the routine parameters.
For $N = 3 - 7$ we find a maximum difference of $0.04\%$ between the optimum ``maxsmooth`` $\chi^2$
values and the Basin-hopping results. Figure taken from @Bevins.](times.png)

Documentation for ``maxsmooth`` is available at [ReadTheDocs](maxsmooth.readthedocs.io/)
and the code can be
found on [Github](https://github.com/htjb/maxsmooth). The code is also pip installable
([PyPI](https://pypi.org/project/maxsmooth/)). Continuous
integration is performed with [Travis](https://travis-ci.com/github/htjb/maxsmooth)
and [CircleCi](https://circleci.com/gh/htjb/maxsmooth). The
associated code coverage can be found at [CodeCov](https://codecov.io/gh/htjb/maxsmooth).

# Acknowledgements

Discussions on the applications of the software were provided by Eloy de Lera Acedo,
Will Handley and Anastasia Fialkov. The author is supported by the Science and
Technology Facilities Council (STFC) via grant number ST/T505997/1.

# References
