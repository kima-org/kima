import os
from common import cleanup_after_running, path_to_test_data, simulated1
import numpy as np

import kima

def test_trend_degree_issues(cleanup_after_running, simulated1):
    m = kima.RVmodel(True, 0, simulated1)
    m.trend = True
    m.degree = 2
    kima.run(m, steps=100)

    res = kima.load_results(m)
    res.plot_random_samples()
    res.hist_trend(show_prior=True, show_title=False)


def test_attributes(cleanup_after_running, simulated1):
    m = kima.RVmodel(True, 0, simulated1)
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


def test_methods(cleanup_after_running, simulated1):
    m = kima.RVmodel(True, 0, simulated1)
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


def test_pickling(cleanup_after_running, simulated1):
    m = kima.RVmodel(True, 0, simulated1)
    m.trend = True
    m.degree = 2
    kima.run(m, steps=100)

    res = kima.load_results(m)
    pkl = res.save_pickle()
    os.remove(pkl)


def test_log_posterior(cleanup_after_running, simulated1):
    m = kima.RVmodel(True, 0, simulated1)
    m.trend = True
    m.degree = 2
    kima.run(m, steps=100)

    res = kima.load_results(m)

    logp = res.log_prior(res.posterior_sample[0])
    assert np.isfinite(logp)
    logl = res.log_likelihood(res.posterior_sample[0])
    assert np.isfinite(logl)
    np.testing.assert_allclose(logl, res.posterior_lnlike[0, 1])


def test_moved_datafile(path_to_test_data):
    import shutil, os, tempfile
    from kima.pykima.utils import chdir
    from kima.pykima.cli import cli_clean
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy(path_to_test_data('simulated1.txt'), tmpdir)
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


def test_np_bayes_factor_threshold():
    from dataclasses import dataclass
    from kima.pykima.analysis import np_bayes_factor_threshold

    @dataclass
    class FakeKimaResults:
        max_components: int
        Np: np.ndarray

    # no planets, 10 samples
    res = FakeKimaResults(0, np.zeros(10))
    np.testing.assert_equal(np_bayes_factor_threshold(res), 0)

    # up to 1 planet, 10 samples with 0 planets
    res = FakeKimaResults(1, np.zeros(10))
    np.testing.assert_equal(np_bayes_factor_threshold(res), 0)

    # up to 1 planet, 10 samples with 1 planet
    res = FakeKimaResults(1, np.ones(10))
    np.testing.assert_equal(np_bayes_factor_threshold(res), 1)

    # up to 2 planets, 10 samples with 1 planet
    res = FakeKimaResults(2, np.ones(10))
    np.testing.assert_equal(np_bayes_factor_threshold(res), 1)

    # probabilities 0, 1 = 0.9, 0.1
    res = FakeKimaResults(1, np.random.choice([0, 1], size=100, p=[0.9, 0.1]))
    np.testing.assert_equal(np_bayes_factor_threshold(res), 0)
