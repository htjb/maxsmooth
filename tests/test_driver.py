import numpy as np
import matplotlib.pyplot as plt
from maxsmooth.DCP import smooth
from numpy.testing import assert_almost_equal

def test_api():
    # Check api
    np.random.seed(0)

    Ndat = 100
    x = np.linspace(-1,1,Ndat)
    y = 1 + x + x**2 + x**3 + np.random.normal(0, 0.05, 100)

    N = 4
    sol = smooth(x, y, N)
    # Check RMS/Chi/y_fit calculated/returned correctly
    assert_almost_equal(sol.rms, np.sqrt(np.sum((y - sol.y_fit)**2)/len(y)))
    assert_almost_equal( sol.Optimum_chi , ((y - sol.y_fit)**2).sum())

def test_smooth():
    # Check parameters of smooth function are correct
    np.random.seed(0)

    Ndat = 100
    x = np.linspace(0,1,Ndat)
    y = 1 + x + x**2 + x**3
    #y = x**2

    N = 4
    sol = smooth(x, y, N, model_type='polynomial')

    assert_almost_equal(sol.Optimum_params.T[0], [1, 1, 1, 1], decimal=3)
