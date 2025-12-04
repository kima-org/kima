import pytest
from common import cleanup_after_running

def test_51Peg():
    from kima.examples import _51Peg

    model, res = _51Peg(run=True, load=True, steps=10)
    assert model is not None, res is not None

def test_BL2009():
    from kima.examples import BL2009

    model, res = BL2009(run=True, load=True, which=1, steps=10)
    assert model is not None, res is not None

    model, res = BL2009(run=True, load=True, which=2, steps=10)
    assert model is not None, res is not None

def test_multi_instruments():
    from kima.examples import multi_instruments

    model, res = multi_instruments(run=True, load=True, steps=20)
    assert model is not None, res is not None
