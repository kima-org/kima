import kima
from kima.examples import _51Peg

def test_51Peg():
    _, res = _51Peg(run=True, load=True, steps=100)
    res.plot_posterior_np()
    res.plot_posterior_PKE()
    res.plot_gp()
    res.plot_random_samples()
    _ = res.maximum_likelihood_sample()

def test_load_after():
    model = _51Peg(run=True, steps=100)
    res = kima.load_results(model)
