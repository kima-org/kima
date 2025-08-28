from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from kima import RVData, keplerian

import pytest

@pytest.fixture
def cleanup_after_running():
    from kima.pykima.cli import cli_clean
    yield
    cli_clean(check=False, output=True)


@pytest.fixture(scope="session")
def path_to_test_data():
    return lambda file: (Path(__file__).parent / file).as_posix()


@pytest.fixture
def simulated1(path_to_test_data):
    from kima import RVData
    return RVData(path_to_test_data('simulated1.txt'))


def create_data(plot=False, include=(1, 2, 4, 6), seed=24):

    # calculate time of transit
    def _get_Tc(P, e, w, Tp) -> float: 
        f = np.pi/2 - w
        E = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-e)/(1+e)))
        Tc = Tp + P/(2*np.pi) * (E - e*np.sin(E))
        return Tc

    include = np.array(include) - 1
    np.random.seed(seed)

    t = np.sort(np.random.uniform(0, 100, 98))
    tt = np.linspace(t.min(), t.max(), 2000)
    err = np.random.uniform(0.1, 0.3, t.size)
    M0_epoch = t[0]

    v = np.zeros_like(t)

    # transiting planets
    par1 = dict(P=29, K=2, e=0.6, w=0.1, Tp=10)
    par1['M0'] = 2 * np.pi * (M0_epoch - par1['Tp']) / par1['P']
    par1['Tc'] = _get_Tc(par1['P'], par1['e'], par1['w'], par1['Tp'])

    par2 = dict(P=19, K=1.8, e=0.3, w=0.9, Tp=15)
    par2['M0'] = 2 * np.pi * (M0_epoch - par2['Tp']) / par2['P']
    par2['Tc'] = _get_Tc(par2['P'], par2['e'], par2['w'], par2['Tp'])

    par3 = dict(P=11, K=0.8, e=0.1, w=2.1, Tp=44)
    par3['M0'] = 2 * np.pi * (M0_epoch - par3['Tp']) / par3['P']
    par3['Tc'] = _get_Tc(par3['P'], par3['e'], par3['w'], par3['Tp'])

    # known objects
    par4 = dict(P=43, K=1.3, e=0.1, w=0.1, M0=0.0)
    par5 = dict(P=83, K=1.0, e=0.0, w=0.3, M0=2.0)

    # *normal* planet
    par6 = dict(P=2.1, K=1.0, e=0.0, w=0.3, M0=0.0)

    pars = (par1, par2, par3, par4, par5, par6)

    for par in np.array(pars)[include]:
        v += keplerian(t, par['P'], par['K'], par['e'], par['w'], par['M0'], M0_epoch)

    # add some small Gaussian noise
    v += np.random.normal(loc=0.0, scale=0.05, size=t.size)

    # add a linear trend
    trend = [0.2, -20]
    tmiddle = t.min() + np.ptp(t) / 2
    v += np.polyval(trend, t - tmiddle)

    if plot:
        fig, ax = plt.subplots(constrained_layout=True)
        ax.errorbar(t, v, yerr=err, fmt='o', label='data')

        vv = np.zeros_like(tt)
        for i, par in enumerate(np.array(pars)[include]):
            _v = keplerian(tt, par['P'], par['K'], par['e'], par['w'], par['M0'], M0_epoch)
            vv += _v
            ax.plot(tt, _v + 30, zorder=-1, alpha=0.3, label=f'planet {include[i]+1}')
        ax.plot(tt, vv + 20, label='planets', zorder=-1)
        ax.plot(tt, vv + np.polyval(trend, tt - tmiddle), label='full', color='k', zorder=-1)
        ax.legend()

    return (
        (t, v, err),
        (par1, par2, par3, par4, par5, par6, trend)
    )

