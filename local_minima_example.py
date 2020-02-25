"""
x and y data generated using,

x = np.linspace(50, 150, 100)
noise = np.random.normal(0, 0.5, len(x))
y = 1e8*x**(-2.5) + 1e8*x**(-3) + noise
np.savetxt('x_data.txt', x)
np.savetxt('y_data.txt', y)

I tried to create multimodal data that was approximately a
x^(-2.5) i.e. representative of the data from a 21-cm experiment
without a signal. It's interesting to note this was quite difficult!
The scale of the noise and the extra x^(-3) term effect the multimodality.
In a real experiment I imagine a signal or a systematic would have the
same effect. For example EDGES can be well approximated by a x^(-2.5)
power law but it is multimodal because of the systematics/signal or noise
I think.

This verision of the maxsmooth code starts with a random sign generation
and then uses a steepest descent algorithm changing the signs
as soon as it finds a step downward. See the definition of
'sign_transform' in the maxsmooth.msf script for the list of
'steps' or transforms being tested. These are tested in the
order in which they are defined which I think makes sense because
they have a form like,

    [-1 1 1 1]
    [1 -1 1 1]
    [1 1 -1 1]
    [1 1 1 -1]

i.e we are changing the 2nd order derivative first which you might
expect to have the most effect on the function.

Also on the note about the time taken to do one fit. Each actually iteration
of the quadratic programming routine is very fast but I think most of the time
is spent in the msf.py code which calls the fitting routine/data saving/model
making ect.

"""

import numpy as np
import pylab as pl
from maxsmooth.msf import smooth
from maxsmooth.settings import setting

x = np.loadtxt('x_data.txt')
y = np.loadtxt('y_data.txt')

pl.subplot(111)
pl.plot(x, y)
pl.xlabel('x "Frequency [MHz]"', fontsize=12)
pl.ylabel('y "Temperature [K]"', fontsize=12)
pl.show()

N = [12]

setting = setting()

runs = 20 # Call the code 20 times and record the chi value
chi = []
for i in range(runs):
    fit = smooth(x, y, N, setting)
    chi.append(fit.Optimum_chi)

run = np.arange(0, runs, 1)

pl.subplot(111)
pl.plot(run, chi)
pl.xlabel('Run Number', fontsize=12)
pl.ylabel(r'$X^2$', fontsize=12)
pl.xticks(run)
pl.tight_layout()
pl.show()
