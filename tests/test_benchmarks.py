from kima.examples import _51Peg

def run_51Peg():
    """ Function that needs some serious benchmarking """
    _ = _51Peg(run=True, steps=100)

def test_my_stuff(benchmark):
    _ = benchmark(run_51Peg)
