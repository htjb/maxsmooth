import numpy as np
import math
import pytest
import os
import shutil
from maxsmooth.DCF import smooth
from maxsmooth.parameter_plotter import param_plotter

def test_keywords():

    np.random.seed(0)

    Ndat = 100
    x = np.linspace(0,1,Ndat)
    y = 1 + x + x**2 + x**3

    N = 4

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    res = smooth(x, y, N, base_dir='new_dir/', data_save=True)

    with pytest.raises(Exception):
        param_plotter(
            res.optimum_params, res.optimum_signs,
            x, y, N, color='pink')
    with pytest.raises(Exception):
        param_plotter(
            res.optimum_params, res.optimum_signs,
            x, y, 5.5)
    with pytest.raises(Exception):
        param_plotter(
            res.optimum_params, res.optimum_signs,
            x, y, N, model_type='banana')
    with pytest.raises(Exception):
        param_plotter(
            res.optimum_params, res.optimum_signs,
            x, y, N, pivot_point='string')
    with pytest.raises(Exception):
        param_plotter(
            res.optimum_params, res.optimum_signs,
            x, y, N, pivot_point=len(x)+10)
    with pytest.raises(Exception):
        param_plotter(
            res.optimum_params, res.optimum_signs,
            x, y, N, base_dir=5)
    with pytest.raises(Exception):
        param_plotter(
            res.optimum_params, res.optimum_signs,
            x, y, N, base_dir='string')
    with pytest.raises(Exception):
        param_plotter(
            res.optimum_params, res.optimum_signs,
            x, y, N, constraints=3.3)
    with pytest.raises(Exception):
        param_plotter(
            res.optimum_params, res.optimum_signs,
            x, y, N, constraints=20)
    with pytest.raises(Exception):
        param_plotter(
            res.optimum_params, res.optimum_signs,
            x, y, N, zero_crossings=[3.3])
    with pytest.raises(Exception):
        param_plotter(
            res.optimum_params, res.optimum_signs,
            x, y, N, zero_crossings=[1])
    with pytest.raises(Exception):
        param_plotter(
            res.optimum_params, res.optimum_signs,
            x, y, N, samples=50.2)
    with pytest.raises(Exception):
        param_plotter(
            res.optimum_params, res.optimum_signs,
            x, y, N, width='string')
    with pytest.raises(Exception):
        param_plotter(
            res.optimum_params, res.optimum_signs,
            x, y, N, warnings='string')
    with pytest.raises(Exception):
        param_plotter(
            res.optimum_params, res.optimum_signs,
            x, y, N, gridlines=9)

def test_files():

    np.random.seed(0)

    Ndat = 100
    x = np.linspace(0,1,Ndat)
    y = 1 + x + x**2 + x**3

    N = 6

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    res = smooth(x, y, N, zero_crossings=[4])

    param_plotter(
        res.optimum_params, res.optimum_signs,
        x, y, N, base_dir='new_dir/', samples=10, zero_crossings=[4])

    assert(os.path.exists('new_dir/') is True)
    assert(os.path.exists('new_dir/Parameter_plot.pdf') is True)

    N = 10

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    res = smooth(x, y, N)

    param_plotter(
        res.optimum_params, res.optimum_signs,
        x, y, N, base_dir='new_dir/', samples=10, gridlines=True)

    assert(os.path.exists('new_dir/') is True)
    assert(os.path.exists('new_dir/Parameter_plot.pdf') is True)

def test_new_basis():

    np.random.seed(0)

    Ndat = 100
    x = np.linspace(-1, 1, Ndat)
    y = 1 + x + x**2 + x**3 + np.random.normal(0, 0.05, 100)

    N = 4

    arguments = [x[-1]*10, y[-1]*10]

    def basis_functions(x, y, pivot_point, N, *args):

        phi = np.empty([len(x), N])
        for h in range(len(x)):
            for i in range(N):
                phi[h, i] = args[1]*(x[h]/args[0])**i

        return phi

    def model(x, y, pivot_point, N, params, *args):

        y_sum = args[1]*np.sum([
            params[i]*(x/args[0])**i
            for i in range(N)], axis=0)

        return y_sum

    def derivative(m, x, y, N, pivot_point, params, *args):
        mth_order_derivative = []
        for i in range(N):
            if i <= m - 1:
                mth_order_derivative.append([0]*len(x))
        for i in range(N - m):
                mth_order_derivative_term = args[1]*math.factorial(m+i) / \
                    math.factorial(i) * \
                    params[int(m)+i]*(x)**i / \
                    (args[0])**(i + 1)
                mth_order_derivative.append(
                    mth_order_derivative_term)

        return mth_order_derivative

    def derivative_pre(m, x, y, N, pivot_point, *args):

        mth_order_derivative = []
        for i in range(N):
            if i <= m - 1:
                mth_order_derivative.append([0]*len(x))
        for i in range(N - m):
                mth_order_derivative_term = args[1]*math.factorial(m+i) / \
                    math.factorial(i) * \
                    (x)**i / \
                    (args[0])**(i + 1)
                mth_order_derivative.append(
                    mth_order_derivative_term)

        return mth_order_derivative

    sol = smooth(x, y, N, basis_functions=basis_functions, model=model,
    derivatives=derivative, der_pres=derivative_pre, args=arguments)

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    param_plotter(
        sol.optimum_params, sol.optimum_signs,
        x, y, N, samples=10, base_dir='new_dir/',
        basis_functions=basis_functions,
        model=model,
        derivatives=derivative, der_pres=derivative_pre, args=arguments)

    assert(os.path.exists('new_dir/') is True)
    assert(os.path.exists('new_dir/Parameter_plot.pdf') is True)

    with pytest.raises(Exception):
        param_plotter(
            sol.optimum_params, sol.optimum_signs,
            x, y, N, samples=10, base_dir='new_dir/',
            basis_functions=basis_functions,
            model=None,
            derivatives=derivative, der_pres=derivative_pre, args=arguments)

def test_new_basis_without_args():

    np.random.seed(0)

    Ndat = 100
    x = np.linspace(-1, 1, Ndat)
    y = 1 + x + x**2 + x**3 + np.random.normal(0, 0.05, 100)

    N = 4

    def basis_functions(x, y, pivot_point, N, *args):

        phi = np.empty([len(x), N])
        for h in range(len(x)):
            for i in range(N):
                phi[h, i] = (x[h])**i

        return phi

    def model(x, y, pivot_point, N, params, *args):

        y_sum = np.sum([
            params[i]*(x)**i
            for i in range(N)], axis=0)

        return y_sum

    def derivative(m, x, y, N, pivot_point, params, *args):
        mth_order_derivative = []
        for i in range(N):
            if i <= m - 1:
                mth_order_derivative.append([0]*len(x))
        for i in range(N - m):
                mth_order_derivative_term = math.factorial(m+i) / \
                    math.factorial(i) * \
                    params[int(m)+i]*(x)**i
                mth_order_derivative.append(
                    mth_order_derivative_term)

        return mth_order_derivative

    def derivative_pre(m, x, y, N, pivot_point, *args):

        mth_order_derivative = []
        for i in range(N):
            if i <= m - 1:
                mth_order_derivative.append([0]*len(x))
        for i in range(N - m):
                mth_order_derivative_term = math.factorial(m+i) / \
                    math.factorial(i) * \
                    (x)**i
                mth_order_derivative.append(
                    mth_order_derivative_term)

        return mth_order_derivative

    sol = smooth(x, y, N, basis_functions=basis_functions, model=model,
    derivatives=derivative, der_pres=derivative_pre)

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    param_plotter(
        sol.optimum_params, sol.optimum_signs,
        x, y, N, samples=10, base_dir='new_dir/',
        basis_functions=basis_functions,
        model=model,
        derivatives=derivative, der_pres=derivative_pre)

    assert(os.path.exists('new_dir/') is True)
    assert(os.path.exists('new_dir/Parameter_plot.pdf') is True)

    with pytest.raises(Exception):
        param_plotter(
            sol.optimum_params, sol.optimum_signs,
            x, y, N, samples=10, base_dir='new_dir/',
            basis_functions=basis_functions,
            model=None,
            derivatives=derivative, der_pres=None)

def test_loglog():

    Ndat = 100
    x = np.linspace(50, 150, Ndat)
    y = 5e7*x**(-2.5)

    N = 4

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    sol = smooth(x, y, N, model_type='loglog_polynomial')

    param_plotter(
        sol.optimum_params, sol.optimum_signs,
        x, y, N, samples=10, base_dir='new_dir/',
        model_type='loglog_polynomial', data_plot=True, center_plot=True)

    assert(os.path.exists('new_dir/Parameter_plot.pdf') is True)
