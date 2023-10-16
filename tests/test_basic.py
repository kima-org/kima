import pytest
import numpy as np
from numpy.testing import assert_allclose

import kima


def test_extensions_exist():
    kima.distributions
    kima.RVData
    kima.RVmodel
    kima.GPmodel
    kima.RVFWHMmodel
    kima.run

@pytest.mark.xfail
def test_api():
    D = kima.RVData('tests/simulated1.txt')
    # print(D.N)
    m = kima.RVmodel(True, 0, D)
    m.trend = True
    m.degree = 2
    # print(m.trend)
    # print(help(kima.run))

@pytest.mark.xfail
def test_RVData():
    D = kima.RVData('tests/simulated1.txt')
    D = kima.RVData(['tests/simulated1.txt', 'tests/simulated2.txt'])

@pytest.mark.xfail
def test_RVmodel():
    m = kima.RVmodel(True, 0, kima.RVData('tests/simulated1.txt'))

@pytest.mark.xfail
def test_GPmodel():
    m = kima.GPmodel(True, 0, kima.RVData('tests/simulated1.txt'))

@pytest.mark.xfail
def test_RVFWHMmodel():
    m = kima.RVFWHMmodel(True, 0, kima.RVData('tests/simulated1.txt'))


def test_distributions():
    from kima import distributions
    from kima.distributions import Gaussian, Uniform
    u = Uniform()
    # print(u)