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