import pytest
import numpy as np
from kima.examples import _51Peg
from kima.GP import QP


def test_51Peg(benchmark):
    _ = benchmark(_51Peg, run=True, steps=100)

@pytest.mark.parametrize("N", [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120])
def test_QP(benchmark, N):
    t = np.sort(np.random.uniform(0, 100, size=N))
    benchmark(lambda: QP(t, 1.0, 22, 27, 0.76))

@pytest.mark.parametrize("N", [10, 50, 100, 500])
def test_GP_likelihood(benchmark, N):
    from kima import RVData, GPmodel
    from kima.distributions import RNG
    t = np.sort(np.random.uniform(0, 100, size=N))
    y = np.random.normal(0, 1, size=t.size)
    yerr = np.random.uniform(0.1, 0.2, size=t.size)
    data = RVData(t, y, yerr, instrument='simulated')
    model = GPmodel(fix=True, npmax=0, data=data)
    model.from_prior(RNG(np.random.randint(1000)))
    benchmark(lambda: model.log_likelihood())