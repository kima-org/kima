import os
import sys

import numpy as np
import kima
from kima import RVdata, RVmodel
from kima.RVmodelC import RVmodelC
from kima import distributions
from kima.examples.gp_example import TRUTHS
from kima.pykima.utils import chdir

__all__ = ['trend']

here = os.path.dirname(__file__) # cwd

TRUTH = [2.0, -0.1, 0.002]

def create_data():
    np.random.seed(41)
    t = np.linspace(0, 180, 67)
    t_middle = t.min() + np.ptp(t)/2
    v = np.random.normal(0, 2, t.size)
    v += TRUTH[0] + TRUTH[1] * (t - t_middle) + TRUTH[2] * (t - t_middle)**2
    v_err = np.random.uniform(0.5, 1, t.size)
    np.savetxt(os.path.join(here, 'data.txt'), np.c_[t, v, v_err], fmt='%.5f')

create_data()
data = RVdata(os.path.join(here, 'data.txt'), skip=0)


def trend(run=False, load=False, degree=1, **kwargs):
    """
    Simple example of using `RVmodel` to analyse simulated data containing a
    long-term (quadratic) trend.
    """
    # load the right data file
    data = RVdata(os.path.join(here, 'data.txt'), skip=0)
    # create the model
    model = RVmodelC(fix=True, npmax=0, data=data)

    model.Jprior = distributions.Uniform(0, 10)
    # model.conditional.Pprior = distributions.LogUniform(1, 10*data.get_timespan())

    model.trend = True
    model.degree = degree

    kwargs.setdefault('steps', 5000)
    kwargs.setdefault('num_threads', 1)
    kwargs.setdefault('num_particles', 1)
    kwargs.setdefault('new_level_interval', 1000)
    kwargs.setdefault('save_interval', 100)

    with chdir(here):
        if run:
            kima.run(model, **kwargs)
        if load:
            res = kima.load_results(model, diagnostic='diagnostic' in sys.argv)
            return model, res
    return model

if __name__ == '__main__':
    create_data()

    model, res = trend(
        run='run' in sys.argv, load=True, 
        degree=1,  # data has a quadratic trend, model as a linear one
        steps=5_000
    )

    res.plot_random_samples()

    print('simulated data has a quadratic trend')
    if res.trend_degree == 1:
        print('model has a linear trend: parameters should not match')
    elif res.trend_degree == 2:
        print('model has a quadratic trend: parameters should match')
    elif res.trend_degree == 3:
        print('model has a cubic trend: parameters should still match')

    fig, _ = res.hist_vsys(show_offsets=False, show_prior=True)
    fig.axes[0].axvline(x=TRUTH[0], ls='--', color='r', alpha=1)

    fig = res.hist_trend(per_year=False, show_prior=True)
    if fig:
        if res.trend_degree >= 1:
            fig.axes[0].axvline(x=TRUTH[1], ls='--', color='r', alpha=1)
        if res.trend_degree >= 2:
            fig.axes[1].axvline(x=TRUTH[2], ls='--', color='r', alpha=1)
