"""
This example will show you how to generate a plot of the :math:`{\chi^2}`
distribution as a function of the descrete sign combinations on the constrained
derivatives.

First you will need to import your data and fit this using ``maxsmooth`` as
was done in the simple example code.

"""

import numpy as np

x = np.load('Data/x.npy')
y = np.load('Data/y.npy')

from maxsmooth.DCF import smooth

N = 10
result = smooth(x, y, N, base_dir='examples/',
    data_save=True, fit_type='qp')

"""
Here we have used some additional keyword arguments for the 'smooth' fitting
function. 'data_save' ensures that the files containing the tested sign combinations
and the corresponding objective function evaluations exist in the base directory
which we have changed to 'base_dir='examples/''. These files are essential for
the plotting the :math:`{\chi^2}` distribution and are not saved by ``maxsmooth``
without 'data_save=True'. We have also set the 'fit_type' to 'qp' rather than the
default 'qp-sign_flipping'. This ensures that all of the available sign
combinations are tested rather than a sampled set giving us a full picture of the
distribution when we plot it. We have used the default DCF model to fit this data.

We can import the 'chi_plotter' like so,
"""

from maxsmooth.chidist_plotter import chi_plotter

"""
and produce the fit which gets placed in the base directory with the following
code,
"""

chi_plotter(N, base_dir='examples/', fit_type='qp')

"""
We pass the same 'base_dir' as before so that the plotter can find the correct output
files. We also give the function the same 'fit_type' used for the fitting which
ensures that the files can be read.
"""
