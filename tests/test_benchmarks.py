import pytest 
import time

def something(duration=0.000001):
    """ Function that needs some serious benchmarking """
    time.sleep(duration)
    # You may return anything you want, like the result of a computation
    return 123

def test_my_stuff(benchmark):
    # benchmark something
    result = benchmark(something)

    # Extra code, to verify that the run completed correctly.
    assert result == 123
