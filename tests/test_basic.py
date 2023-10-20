import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

import kima


def test_extensions_exist():
    kima.distributions
    kima.RVData
    kima.RVmodel
    kima.GPmodel
    kima.RVFWHMmodel
    kima.run

def test_api():
    D = kima.RVData('tests/simulated1.txt')
    # print(D.N)
    m = kima.RVmodel(True, 0, D)
    m.trend = True
    m.degree = 2
    # print(m.trend)
    # print(help(kima.run))

def test_RVData():
    # one instrument
    D = kima.RVData('tests/simulated1.txt')
    assert_equal(D.N, 40)
    # two instruments
    D = kima.RVData(['tests/simulated1.txt', 'tests/simulated2.txt'])
    assert_equal(D.N, 80)
    # read indicators too
    D = kima.RVData('tests/simulated2.txt', indicators=['i', 'j'])
    assert_equal(D.N, 40)
    # fail for one character file name
    with pytest.raises(RuntimeError):
        D = kima.RVData('i')

def test_RVmodel():
    m = kima.RVmodel(True, 0, kima.RVData('tests/simulated1.txt'))

def test_GPmodel():
    m = kima.GPmodel(True, 0, kima.RVData('tests/simulated1.txt'))

def test_RVFWHMmodel():
    # should fail because it doesn't read 4th and 5th column
    with pytest.raises(RuntimeError):
        m = kima.RVFWHMmodel(True, 0, kima.RVData('tests/simulated1.txt'))
    # this one should work
    m = kima.RVFWHMmodel(True, 0, 
                         kima.RVData('tests/simulated2.txt', indicators=['i', 'j']))


def test_distributions():
    from kima import distributions
    from kima.distributions import Gaussian, Uniform
    u = Uniform()
    # print(u)