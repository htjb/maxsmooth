import numpy as np
import os
import shutil
import pytest
from maxsmooth.best_basis import basis_test

def test_keywords():

    np.random.seed(0)

    Ndat = 100
    x = np.linspace(0,1,Ndat)
    y = 1 + x + x**2 + x**3

    N = 4

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

def test_directory():

    np.random.seed(0)

    Ndat = 100
    x = np.linspace(0,1,Ndat)
    y = 1 + x + x**2 + x**3

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    basis_test(x, y, base_dir='new_dir/', N=[4, 5])

    assert(os.path.exists('new_dir/') is True)
    assert(os.path.exists('new_dir/Basis_functions.pdf') is True)

def test_loglog():

    Ndat = 100
    x = np.linspace(50, 150, Ndat)
    y = 5e7*x**(-2.5)

    if os.path.isdir('new_dir/'):
        shutil.rmtree('new_dir/')

    basis_test(x, y, base_dir='new_dir/', N=[4, 5])

    assert(os.path.exists('new_dir/') is True)
    assert(os.path.exists('new_dir/Basis_functions.pdf') is True)
