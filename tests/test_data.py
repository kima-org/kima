import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_equal

import kima


def test_RVData():
    # one instrument
    D = kima.RVData('tests/simulated1.txt')
    assert_equal(D.N, 40)
    assert_equal(len(D.t), 40)
    assert_equal(np.array(D.obsi), 1)

    assert_allclose(D.M0_epoch, 7.49793)
    D.M0_epoch = 0.0
    assert_allclose(D.M0_epoch, 0.0)

    # two instruments
    D = kima.RVData(['tests/simulated1.txt', 'tests/simulated2.txt'])
    assert_equal(D.N, 80)

    # two instruments but only one file
    D = kima.RVData('tests/simulated2.txt', multi=True)
    assert_equal(D.N, 40)
    assert(D.multi)
    assert_equal(len(D.obsi), 40)
    assert_equal((np.array(D.obsi) == 1).sum(), 21)
    assert_equal((np.array(D.obsi) == 2).sum(), 19)

    # should fail on a file that doesn't have the 4th column
    with pytest.raises(RuntimeError):
        D = kima.RVData('tests/simulated1.txt', multi=True)


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


def test_RVData_missing_indicators():
    # test the issue described in https://github.com/kima-org/kima/issues/22

    # simulated2.txt is missing the 7th column
    with pytest.raises(RuntimeError):
        _ = kima.RVData('tests/simulated2.txt',
                        indicators=['i', 'j', 'n', 'missing'])

    # simulated1.txt is missing the 3rd column
    with pytest.raises(RuntimeError):
        _ = kima.RVData(['tests/simulated2.txt', 'tests/simulated1.txt'],
                        indicators=['i', 'j'])

def test_delimiter():
    D = kima.RVData('tests/simulated2.txt', delimiter='\t')
    assert_equal(D.N, 40)


def test_RVData_skip_indicators():
    # skip the 4th column of simulated2.txt
    D = kima.RVData('tests/simulated2.txt', indicators=['', 'j'])
    assert_equal(D.N, 40)
    assert_equal(np.shape(D.actind), (1, D.N))
    assert_equal(D.indicator_names, ['j'])
    assert_allclose(D.actind[0][0], 0.43098)


def test_RVData_normalized_actind():
    D = kima.RVData('tests/simulated2.txt', indicators=['i'])
    assert_equal(np.shape(D.actind), (1, D.N))
    assert_equal(np.shape(D.normalized_actind), (1, D.N))
    assert_equal(np.min(D.normalized_actind), 0.0)
    assert_equal(np.max(D.normalized_actind), 1.0)
    ai = np.array(D.actind)
    assert_allclose(D.normalized_actind, (ai - np.min(ai)) / np.ptp(ai))


def test_bad_file():
    with pytest.raises(ValueError):
        D = kima.RVData('tests/bad_file.txt', skip=1)

    D = kima.RVData('tests/bad_file.txt', skip=1, delimiter='\t')
    assert_equal(D.N, 2)


def test_ETVData():
    D = kima.ETVData('tests/simulated3.txt')

def test_GaiaData():
    D = kima.GAIAdata('tests/simulated4.txt')