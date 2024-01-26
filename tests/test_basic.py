import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

import kima


def test_extensions_exist():
    kima.distributions
    kima.RVData
    kima.RVmodel
    kima.GPmodel
    kima.BINARIESmodel
    kima.RVFWHMmodel
    kima.run

def test_simple_api():
    D = kima.RVData('tests/simulated1.txt')
    # print(D.N)
    m = kima.RVmodel(True, 0, D)
    m.trend = True
    m.degree = 2

def test_RVData():
    # one instrument
    D = kima.RVData('tests/simulated1.txt')
    assert_equal(D.N, 40)
    assert_equal(len(D.t), 40)
    assert_equal(np.array(D.obsi), 1)

    assert_allclose(D.M0_epoch, 0.16166)
    D.M0_epoch = 0.0
    assert_allclose(D.M0_epoch, 0.0)

    # two instruments
    D = kima.RVData(['tests/simulated1.txt', 'tests/simulated2.txt'])
    assert_equal(D.N, 80)

    # read indicators too
    D = kima.RVData('tests/simulated2.txt', indicators=['i', 'j'])
    assert_equal(D.N, 40)

    # fail for one character file name
    with pytest.raises(RuntimeError):
        D = kima.RVData('i')

    # max_rows
    D = kima.RVData('tests/simulated2.txt', max_rows=20)
    assert_equal(D.N, 20)

    # load from arrays
    t, y, sig = np.random.rand(3, 50)
    D = kima.RVData(t, y, sig, units='ms')
    assert_equal(D.N, 50)
    assert_equal(len(D.t), len(D.y))
    assert_equal(len(D.t), len(D.sig))
    assert_equal(np.array(D.obsi), 1)

    t2, y2, sig2 = np.random.rand(3, 20)
    D = kima.RVData([t, t2], [y, y2], [sig, sig2], units='ms', instruments=['I1', 'I2'])
    assert_equal(D.N, 70)
    assert_equal(len(D.t), len(D.y))
    assert_equal(len(D.t), len(D.sig))
    assert_equal(np.unique(D.obsi), [1, 2])


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


def test_run():
    m = kima.RVmodel(True, 0, kima.RVData('tests/simulated1.txt'))
    kima.run(m, steps=1)
    m = kima.GPmodel(True, 0, kima.RVData('tests/simulated1.txt'))
    kima.run(m, steps=1)
    m = kima.RVFWHMmodel(True, 0, kima.RVData('tests/simulated2.txt', indicators=['i', 'j']))
    kima.run(m, steps=1)


def test_distributions():
    from kima import distributions
    from kima.distributions import Gaussian, Uniform, Fixed
    u = Uniform(0, 2)
    assert str(u) == 'Uniform(0; 2)'
    assert str(Gaussian(0, 1)) == 'Gaussian(0; 1)'
    from kima.distributions import UniformAngle
    assert_equal(UniformAngle().logpdf(1.0), Uniform(0.0, 2*np.pi).logpdf(1.0))
