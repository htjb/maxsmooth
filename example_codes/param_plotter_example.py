"""
We can assess the parameter space around the optimum solution
found using ``maxsmooth`` with the param_plotter() function.
This can help us identify how well a problem can be solved using the
sign sampling approach employed by ``maxsmooth`` or simply
be used to identify correlations between the foreground parameters.
For more details on this see the ``maxsmooth`` paper.

We begin by importing and fitting the data as with the chi_plotter()
function illustrated above.
"""

import numpy as np

x = np.load('Data/x.npy')
y = np.load('Data/y.npy')

from maxsmooth.DCF import smooth

N = 5
result = smooth(x, y, N, base_dir='examples/', fit_type='qp')

"""
We have changed the order of the fit to 5 to illustrate that for
order :math:`{N \leq 5}` and fits with derivatives :math:`{m \geq 2}` constrained
the function will plot each region of the graph corresponding to
different sign functions in a different colourmap. If the constraints are
different or the order is greater than 5 then the viable regions will have
a single colourmap. Invalid regions are plotted as black shaded colourmaps
and the contour lines are contours of :math:`{\chi^2}`.

We can import the function like so,
"""

from maxsmooth.parameter_plotter import param_plotter

"""
and access it using,
"""

param_plotter(result.optimum_params, result.optimum_signs,
    x, y, N, base_dir='examples/')

"""
The function takes in the optimum parameters and signs found after the fit
aswell as the data and order of the fit. There are a number of keyword arguments
detailed in the following section and the resultant fit is shown below. The
function by default samples the parameter ranges 50% either side of the optimum
and calculates 50 spamples for each parameter. In each panel the two
labelled parameters are varied while the others are maintained at their optimum
values.
"""
