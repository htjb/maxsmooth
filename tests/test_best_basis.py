import numpy as np
import os
import shutil
import pytest
from maxsmooth.best_basis import basis_test

def test_keywords():

    np.random.seed(0)

    Ndat = 100
    x = np.linspace(0, 1, Ndat)
    y = 1 + x + x**2 + x**3

    with pytest.raises(Exception):
        basis_test(x, y, colour='pink')
    with pytest.raises(Exception):
        basis_test(x, y, fit_type='pink')
    with pytest.raises(Exception):
        basis_test(x, y, base_dir='output_files')
    with pytest.raises(Exception):
        basis_test(x, y, base_dir=5)
    with pytest.raises(Exception):
        basis_test(x, y, N='banana')
    with pytest.raises(Exception):
        basis_test(x, y, N=[3.3])
    with pytest.raises(Exception):
        basis_test(x, y, pivot_point='banana')
    with pytest.raises(Exception):
        basis_test(x, y, pivot_point=112)
    with pytest.raises(Exception):
        basis_test(x, y, constraints='banana')
    with pytest.raises(Exception):
        basis_test(x, y, constraints=16)
    with pytest.raises(Exception):
        basis_test(x, y, zero_crossings=[6.6, 7.2])
    with pytest.raises(Exception):
        basis_test(x, y, constraints=5, zero_crossings=[3])
    with pytest.raises(Exception):
        basis_test(x, y, chi_squared_limit='banana')
    with pytest.raises(Exception):
        basis_test(x, y, cap=45.5)
    with pytest.raises(Exception):
        basis_test(x, y, cvxopt_maxiter=22.2)

def test_directory():

    np.random.seed(0)

    Ndat = 100
    x = np.linspace(0,1,Ndat)
    y = 1 + x + x**2 + x**3

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    basis_test(x, y, base_dir='new_dir/', N=[4, 5], pivot_point=0)

    assert(os.path.exists('new_dir/') is True)
    assert(os.path.exists('new_dir/Basis_functions.pdf') is True)

def test_loglog():

    Ndat = 100
    x = np.linspace(50, 150, Ndat)
    y = 5e7*x**(-2.5)

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    basis_test(x, y, base_dir='new_dir/', N=[4, 5], chi_squared_limit=200)

    assert(os.path.exists('new_dir/') is True)
    assert(os.path.exists('new_dir/Basis_functions.pdf') is True)
