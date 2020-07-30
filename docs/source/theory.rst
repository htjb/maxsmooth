This section has been adapted from section 4 of the ``maxsmooth`` paper
in order to explain how the algorithm works. What follows is a discussion of
the fitting problem and the
``maxsmooth`` algorithm. To restate concisely the problem being fitted we have

.. math::

        &\min_{a,~s}~~\frac{1}{2}~\mathbf{a}^T~\mathbf{Q}~\mathbf{a}~+~\mathbf{q}^T~\mathbf{a}, \\
        &\mathrm{s.t.}~~\mathbf{G(s)~a} \leq \mathbf{0}.

where :math:`{\mathbf{s}}` are the ``maxsmooth`` signs corresponding to the
signs on the derivatives.
A `problem' in this context is the combination of the data, order, basis
function and constraints on the DCF.

With ``maxsmooth`` we can test all possible sign combinations on the constrained derivatives.
This is a
reliable method and, provided the problem can be solved with quadratic programming,
will always give the correct global minimum. When the problem we are interested
in is "well defined", we can develop a quicker algorithm that searches or navigates
through the discrete ``maxsmooth`` sign spaces to find the global minimum.
Each sign space is a discrete parameter space with its own global minimum.
Using quadratic programming on a fit with a specific sign combination will
find this global minimum, and we are interested in finding the minimum
of these global minima.

A "well defined" problem is one in which the discrete sign spaces have large
variance in their minimum :math:`{\chi^2}` values and the sign space for the
global minimum is easily identifiable. In contrast we can have an "ill defined"
problem in which the variance in minimum :math:`{\chi^2}` across all sign
combinations is small. This concept of "well defined" and "ill defined" problems
is explored further in the following two sections.

Well Defined Problems and Discrete Sign Space Searches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :math:`{\chi^2}` Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We investigate the distribution of :math:`{\chi^2}` values, shown in the figure below,
for a 10 :math:`{^{th}}` order y-log(x) space MSF fit to a :math:`{y = x^{-2.5}}`
power law plus gaussian noise.

In the figure, a combination of all positive derivatives~(negative signs) and
all negative derivatives~(positive signs) corresponds to sign combination numbers
255 and 0 respectively. Specifically, the ``maxsmooth`` signs, :math:`{\mathbf{s}}`,
are related to the sign combination number by its :math:`{C}` bit binary representation,
here :math:`{C = (N -2)}`. In binary the sign combination numbers run from
00000000 to 11111111. Each bit represents the sign on the :math:`{m^{th}}`
order derivative with a 1 representing a negative ``maxsmooth`` sign.

.. image:: https://github.com/htjb/maxsmooth/raw/master/docs/images/chi_dist_theory.png

The distribution appears to be composed of smooth steps or shelves; however,
when each shelf is studied closer, we find a series of peaks and troughs. This can
be seen in the subplot of the above figure which shows the distribution in the
neighbourhood of the global minimum found in the large or `global' well. This type
of distribution with a large variance in :math:`{\chi^2}` is characteristic of a "well defined"
problem. We use this example :math:`{\chi^2}` distribution to motivate the ``maxsmooth``
algorithm outlined in the following section.

The ``maxsmooth`` Sign Navigating Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Exploration of the discrete sign spaces for high :math:`{N}` can be achieved by
exploring the spaces around an iteratively updated optimum sign combination.
The ``maxsmooth`` algorithm begins with a randomly generated set of signs for
which the objective function is evaluated and the optimum parameters are found.
We flip each individual sign one at a time beginning with the lowest order
constrained derivative first. When the objective function is evaluated to be lower
than that for the optimum sign combination, we replace it with the new set and repeat
the process in a `cascading' routine until the objective function stops decreasing in value.

The local minima shown in the :math:`{\chi^2}` distribution above mean that the
cascading algorithm is not sufficient to consistently find the global minimum.
We can demonstrate this by performing 100 separate runs of the cascading
algorithm on :math:`{y = x^{-2.5} + \mathrm{noise}}`, and we use a y-log(x) space
:math:`{10^{th}}` order MSF again. We find the true global minimum 79
times and a second local minimum 21 times.

To prevent the routine terminating in a local minimum we perform a complete search
of the sign spaces surrounding the minimum found after the cascading routine.
We refer to this search as a directional exploration and impose limits on its
extent. In each direction we limit the number of sign combinations to explore and
we limit the maximum allowed increase in :math:`{\chi^2}` value. These limits can
be modified by the user. We prevent repeated calculations of the minimum for given
signs and treat the minimum of all tested signs as the global minimum.

We run the consistency test again, with the full ``maxsmooth`` algorithm, and find
that for all 100 trial fits we find the same :math:`{\chi^2}` found when testing
all sign combinations. In the figure below, the red arrows show the approximate path
taken through the discrete sign spaces against the complete distribution of :math:`{\chi^2}`.
Point (1a) shows the random starting point in the algorithm, and point (1b) shows a rejected sign
combination evaluated during the cascade from point (1a) to (2). Point (2), therefore,
corresponds to a step through the cascade. Point (3) marks the end of the cascade
and the start of the left directional exploration. Finally, point (4) shows the end
of the right directional exploration where the calculated :math:`{\chi^2}`
value exceeds the limit on the directional exploration.

.. image:: https://github.com/htjb/maxsmooth/raw/master/docs/images/routine.png

The global well tends to be associated with signs that are all positive,
all negative or alternating. We see this in the figure above where the minimum falls
at sign combination number 169 and number 170, characteristic of the derivatives for
a :math:`{x^{-2.5}}` power law, corresponds to alternating positive and negative
derivatives from order :math:`{m = 2}`. Standard patterns of derivative signs can be seen
for all data following approximate power laws. All positive derivatives, all negative
and alternating signs correspond to data following the approximate power laws
:math:`{y\approx x^{k}}`, :math:`{y\approx -x^{k}}`, :math:`{y\approx x^{-k}}` and
:math:`{y\approx -x^{-k}}`.

The ``maxsmooth`` algorithm assumes that the global well is present in the :math:`{\chi^2}`
distribution and this is often the case. The use of DCFs is primarily driven by a
desire to constrain previously proposed polynomial models to foregrounds. As a result
we would expect that the data being fitted could be described by one of the four
approximate power laws highlighted above and that the global minimum will fall
around an associated sign combination. In rare cases the global well is not clearly
defined and this is described in the following subsection.

Ill Defined Problems and their Identification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can illustrate an "ill defined" problem, with a small variation in
:math:`{\chi^2}` across the ``maxsmooth`` sign spaces, by adding a non-smooth signal
of interest into the foreground model, :math:`{x^{-2.5}}` and fitting this with
a 10 :math:`{^{th}}` order log(y)-log(x) space MSF. We add an additional noise of
:math:`{0.020}` to the mock data. The resultant :math:`{\chi^2}` distribution with its
global minimum is shown in the top panel of the figure below.

The global minimum, shown as a black data point, cannot be found using the
``maxsmooth`` algorithm. The cascading algorithm may terminate in any of the
approximately equal minima and the directional exploration will then quickly
terminate because of the limits imposed.

.. image:: https://github.com/htjb/maxsmooth/raw/master/docs/images/combined_chi.png

If we repeat the above fit and perform it with a y-x space MSF we find that the
problem is well defined with a larger :math:`{\chi^2}` variation across sign
combinations. This is shown in the bottom panel of the above figure. The results,
when using the log(y)-log(x) space MSF, are significantly better than when using
y-x space MSF meaning it is important to be able to solve "ill defined" problems.
This can be done by testing all ``maxsmooth`` signs but knowing when this is
necessary is important if you are expecting to run multiple DCF fits to the
same data set. We can focus on diagnosing whether a DCF fit to the data is
"ill defined" because a joint fit to the same data set of a DCF and signal
of interest will also feature an "ill defined" :math:`{\chi^2}` distribution.

We can identify an "ill defined" problem by producing the equivalent of
the above figure using ``maxsmooth`` and visually assessing the :math:`{\chi^2}`
distribution for a DCF fit. Alternatively, we can use the parameter space plots,
detailed in the ``maxsmooth`` paper and later in this documentation,
to identify whether the constraints are weak or not, and if a local minima is
returned from the sign navigating routine then the minimum in these plots
will appear off centre.

Assessment of the first derivative of the data can also help to identify an
"ill defined" problem. For the example problem this is shown in the figure below
where the derivatives have been approximated using :math:`{\Delta y/ \Delta x}`.
Higher order derivatives of the data will have similarly complex or simplistic
structures in the respective spaces. There are many combinations of parameters
that will provide smooth fits with similar :math:`{\chi^2}` values in logarithmic
space leading to the presence of local minima. This issue will also be present
in any data set where the noise or signal of interest are of a similar magnitude
to the foreground in y - x space.

.. image:: https://github.com/htjb/maxsmooth/raw/master/docs/images/Gradient_fits.png
