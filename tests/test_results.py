from common import cleanup_after_running
import numpy as np

import kima

def test_trend_degree_issues(cleanup_after_running):
    m = kima.RVmodel(True, 0, kima.RVData('tests/simulated1.txt'))
    m.trend = True
    m.degree = 2
    kima.run(m, steps=100)

    res = kima.load_results(m)
    res.plot_random_samples()
    res.hist_trend(show_prior=True, show_title=False)


def test_attributes(cleanup_after_running):
    m = kima.RVmodel(True, 0, kima.RVData('tests/simulated1.txt'))
    m.trend = True
    m.degree = 2
    kima.run(m, steps=100)

    res = kima.load_results(m)
    res.model
    res.priors
    res.ESS
    res.evidence
    res.information
    res.data
    res.posteriors
    res.parameter_priors


def test_methods(cleanup_after_running):
    m = kima.RVmodel(True, 0, kima.RVData('tests/simulated1.txt'))
    m.trend = True
    m.degree = 2
    kima.run(m, steps=100)

    res = kima.load_results(m)

    res.log_prior(res.posterior_sample[0])
    res.log_likelihood(res.posterior_sample[0])
    res.log_posterior(res.posterior_sample[0])
    res.maximum_likelihood_sample()
    res.map_sample()
    res.median_sample()
    res.eval_model(res.posterior_sample[0])
    res.planet_model(res.posterior_sample[0])
    res.full_model(res.posterior_sample[0])
    res.stochastic_model(res.posterior_sample[0])
    res.burst_model(res.posterior_sample[0])


def test_pickling(cleanup_after_running):
    m = kima.RVmodel(True, 0, kima.RVData('tests/simulated1.txt'))
    m.trend = True
    m.degree = 2
    kima.run(m, steps=100)

    res = kima.load_results(m)
    pkl = res.save_pickle()
    print(pkl)

def test_log_posterior(cleanup_after_running):
    m = kima.RVmodel(True, 0, kima.RVData('tests/simulated1.txt'))
    m.trend = True
    m.degree = 2
    kima.run(m, steps=100)

    res = kima.load_results(m)

    logp = res.log_prior(res.posterior_sample[0])
    assert np.isfinite(logp)
    logl = res.log_likelihood(res.posterior_sample[0])
    assert np.isfinite(logl)
    np.testing.assert_allclose(logl, res.posterior_lnlike[0, 1])


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
