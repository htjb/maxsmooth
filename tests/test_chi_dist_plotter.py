import numpy as np
from maxsmooth.chidist_plotter import chi_plotter
from maxsmooth.DCF import smooth
import pytest
import os
import shutil

def test_keywords():

    np.random.seed(0)

    Ndat = 100
    x = np.linspace(0,1,Ndat)
    y = 1 + x + x**2 + x**3

    N = 4

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    smooth(x, y, N, base_dir='new_dir/', data_save=True)

    with pytest.raises(Exception):
        chi_plotter(N='Banana', base_dir='new_dir/')
    with pytest.raises(Exception):
        chi_plotter(N=3.3, base_dir='new_dir/')
    with pytest.raises(Exception):
        chi_plotter(N, base_dir='new_dir')
    with pytest.raises(Exception):
        chi_plotter(N, base_dir=5)
    with pytest.raises(Exception):
        chi_plotter(N, color='pink', base_dir='new_dir/')
    with pytest.raises(Exception):
        chi_plotter(N, constraints=4.3, base_dir='new_dir/')
    with pytest.raises(Exception):
        chi_plotter(N, constraints=7, base_dir='new_dir/')
    with pytest.raises(Exception):
        chi_plotter(N, zero_crossings=[3.3], base_dir='new_dir/')
    with pytest.raises(Exception):
        chi_plotter(N, zero_crossings=[1], base_dir='new_dir/')
    with pytest.raises(Exception):
        chi_plotter(N, zero_crossings='string', base_dir='new_dir/')
    with pytest.raises(Exception):
        chi_plotter(N, fit_type='pink', base_dir='new_dir/')
    with pytest.raises(Exception):
        chi_plotter(N, chi_squared_limit='banana', base_dir='new_dir/')
    with pytest.raises(Exception):
        chi_plotter(N, cap=5.5, base_dir='new_dir/')
    with pytest.raises(Exception):
        chi_plotter(N, plot_limits='pink', base_dir='new_dir/')

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    with pytest.raises(Exception):
        chi_plotter(N, base_dir='new_dir/')

def test_directory():

    np.random.seed(0)

    Ndat = 100
    x = np.linspace(0,1,Ndat)
    y = 1 + x + x**2 + x**3

    N = 4

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    smooth(x, y, N, base_dir='new_dir/')

    with pytest.raises(Exception):
        chi_plotter(N)

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    smooth(x, y, N, base_dir='new_dir/', data_save=True)

    if os.path.isdir('new_dir/Output_Signs/'):
        shutil.rmtree('new_dir/Output_Signs/')

    with pytest.raises(Exception):
        chi_plotter(N, base_dir='new_dir/')

def test_files():

    Ndat = 100
    x = np.linspace(50, 150, Ndat)
    y = 5e7*x**(-2.5)

    N = 4

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    smooth(
        x, y, N, base_dir='new_dir/',
        data_save=True, model_type='log_polynomial')

    plot = chi_plotter(N, base_dir='new_dir/')

    assert(plot.chi is None)
    assert(plot.cap == ((2**(N-2))//N) + N)
    assert(os.path.exists('new_dir/chi_distribution.pdf') is True)

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    smooth(x, y, N, base_dir='new_dir/', data_save=True, fit_type='qp')

    plot = chi_plotter(N, base_dir='new_dir/', fit_type='qp')

    assert(plot.chi is None)
    assert(plot.cap == ((2**(N-2))//N) + N)
    assert(os.path.exists('new_dir/chi_distribution.pdf') is True)

def test_chi():

    np.random.seed(0)

    Ndat = 100
    x = np.linspace(50, 150, Ndat)
    y = 5e7*x**(-2.5)

    N = 4

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    smooth(
        x, y, N, base_dir='new_dir/', data_save=True,
        model_type='loglog_polynomial', fit_type='qp')

    def model(x, N, params):
        y_sum = 10**(np.sum([
            params[i]*np.log10(x)**i
            for i in range(N)],
            axis=0))
        return y_sum

    parameters = np.loadtxt(
        'new_dir/Output_Parameters/4_qp.txt')

    chi_out = []
    for i in range(len(parameters)):
        chi_out.append(np.sum((y - model(x, N, parameters[i]))**2))
    chi_out = np.array(chi_out)

    plot = chi_plotter(N, base_dir='new_dir/', chi=chi_out,
        fit_type='qp')

    assert(np.all(plot.chi == chi_out))
    assert(plot.chi_squared_limit == 2*min(chi_out))
    assert(os.path.exists('new_dir/chi_distribution.pdf') is True)

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    smooth(
        x, y, N, base_dir='new_dir/', data_save=True,
        model_type='loglog_polynomial')

    parameters = np.loadtxt(
        'new_dir/Output_Parameters/4_qp-sign_flipping.txt')

    chi_out = []
    for i in range(len(parameters)):
        chi_out.append(np.sum((y - model(x, N, parameters[i]))**2))
    chi_out = np.array(chi_out)

    plot = chi_plotter(N, base_dir='new_dir/', chi=chi_out)

    assert(np.all(plot.chi == chi_out))
    assert(plot.chi_squared_limit == 2*min(chi_out))
    assert(os.path.exists('new_dir/chi_distribution.pdf') is True)

def test_ifp():

    np.random.seed(0)

    Ndat = 100
    x = np.linspace(-1, 1, Ndat)
    y = 1 + x + x**2 + x**3 + np.random.normal(0, 0.05, 100)

    N = 10

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    sol = smooth(
        x, y, N, zero_crossings=[4, 5, 6], constraints=1, base_dir='new_dir/',
        data_save=True)

    chi_plotter(N, base_dir='new_dir/', zero_crossings=[4, 5, 6], constraints=1)
    assert(os.path.exists('new_dir/chi_distribution.pdf') is True)

def test_limits():

    np.random.seed(0)

    Ndat = 100
    x = np.linspace(-1, 1, Ndat)
    y = 1 + x + x**2 + x**3 + np.random.normal(0, 0.05, 100)

    N = 10

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    sol = smooth(x, y, N, base_dir='new_dir/', data_save=True)

    chi_plotter(
        N, base_dir='new_dir/', cap=10, chi_squared_limit=1e4,
        plot_limits=True)
    assert(os.path.exists('new_dir/chi_distribution.pdf') is True)
