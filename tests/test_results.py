# import pytest
# import numpy as np

import kima

def test_trend_degree_issues():
    m = kima.RVmodel(True, 0, kima.RVData('tests/simulated1.txt'))
    m.trend = True
    m.degree = 2
    kima.run(m, steps=100)

    res = kima.load_results(m)
    res.plot_random_samples()

    from kima.pykima.cli import cli_clean
    cli_clean(check=False, output=True)


def test_moved_datafile():
    import shutil, os, tempfile
    from kima.pykima.utils import chdir
    from kima.pykima.cli import cli_clean
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy('tests/simulated1.txt', tmpdir)
        with chdir(tmpdir):
            # build the model
            m = kima.RVmodel(True, 0, kima.RVData('simulated1.txt'))
            # run it
            kima.run(m, steps=100)
            # remove the datafile
            os.remove('simulated1.txt')
            assert 'simulated1.txt' not in os.listdir('.')
            # load the results
            res = kima.load_results(m)
            assert res.data.N == 40
            res.plot_random_samples()
            # cleanup
            cli_clean(check=False, output=True)
