import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

import kima
from kima.distributions import Uniform, UniformAngle


def test_RVmodel_setup():
    m = kima.RVmodel(True, 0, kima.RVData('tests/simulated1.txt'))
    kima.run(m, steps=1)

    setup_file = open('kima_model_setup.txt').read()
    for p in ('Cprior', 'Jprior'):
        assert p in setup_file

    # with planets
    m = kima.RVmodel(True, 1, kima.RVData('tests/simulated1.txt'))
    kima.run(m, steps=1)

    setup_file = open('kima_model_setup.txt').read()
    for p in ('Cprior', 'Jprior'):
        assert p in setup_file
    for p in ('Pprior', 'Kprior', 'eprior', 'phiprior', 'wprior'):
        assert p in setup_file

    # with planets and known object
    m = kima.RVmodel(True, 1, kima.RVData('tests/simulated1.txt'))
    m.set_known_object(2)
    m.KO_Pprior = [Uniform(1,2), Uniform(2,3)]
    m.KO_Kprior = [Uniform(1,2), Uniform(2,3)]
    m.KO_eprior = [Uniform(0,1), Uniform(0,1)]
    m.KO_wprior = [UniformAngle(), UniformAngle()]
    m.KO_phiprior = [UniformAngle(), UniformAngle()]
    kima.run(m, steps=1)

    setup_file = open('kima_model_setup.txt').read()

    assert 'known_object: true' in setup_file
    assert 'known_object: 2' in setup_file

    for p in ('Cprior', 'Jprior'):
        assert p in setup_file
    for p in ('Pprior', 'Kprior', 'eprior', 'phiprior', 'wprior'):
        assert p in setup_file
    for p in ('Pprior_0', 'Kprior_0', 'eprior_0', 'phiprior_0', 'wprior_0'):
        assert p in setup_file
    for p in ('Pprior_1', 'Kprior_1', 'eprior_1', 'phiprior_1', 'wprior_1'):
        assert p in setup_file

    from kima.pykima.cli import cli_clean
    cli_clean(check=False, output=True)


def test_GPmodel_setup():
    m = kima.GPmodel(True, 0, kima.RVData('tests/simulated1.txt'))
    kima.run(m, steps=1)

    setup_file = open('kima_model_setup.txt').read()
    for p in ('Cprior', 'Jprior'):
        assert p in setup_file
    for p in ('eta1_prior', 'eta2_prior', 'eta3_prior', 'eta4_prior'):
        assert p in setup_file

    from kima.pykima.cli import cli_clean
    cli_clean(check=False, output=True)