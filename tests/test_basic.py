# ruff: noqa: F401, F811
import pytest
from common import cleanup_after_running, path_to_test_data

import numpy as np

import matplotlib
matplotlib.use('Agg')

import kima


def test_extensions_exist():
    kima.distributions
    kima.RVData, kima.HGPMdata, kima.GAIAdata
    kima.RVmodel, kima.GPmodel
    kima.BINARIESmodel, kima.GAIAmodel, kima.RVGAIAmodel
    kima.RVFWHMmodel
    kima.run


def test_simple_api(path_to_test_data):
    D = kima.RVData(path_to_test_data("simulated1.txt"))
    # print(D.N)
    m = kima.RVmodel(True, 0, D)
    m.trend = True
    m.degree = 2


def test_RVData_reductions():
    pass


def test_RVmodel(path_to_test_data):
    data = kima.RVData(path_to_test_data("simulated1.txt"))
    m = kima.RVmodel(True, 0, data)


def test_GPmodel(path_to_test_data):
    data = kima.RVData(path_to_test_data("simulated1.txt"))
    m = kima.GPmodel(True, 0, data)


def test_RVFWHMmodel(path_to_test_data):
    # should fail because it doesn't read 4th and 5th column
    with pytest.raises(RuntimeError):
        data = kima.RVData(path_to_test_data("simulated1.txt"))
        m = kima.RVFWHMmodel(True, 0, data)

    # this one should work
    data = kima.RVData(path_to_test_data("simulated2.txt"), indicators=["i", "j"])
    m = kima.RVFWHMmodel(True, 0, data)


def test_BINARIESmodel(path_to_test_data):
    data = kima.RVData(path_to_test_data("simulated1.txt"))
    m = kima.BINARIESmodel(True, 0, data)

    # should fail because it doesn't read 4th and 5th column
    with pytest.raises(RuntimeError):
        data = kima.RVData(path_to_test_data("simulated1.txt"), double_lined=True)
        m = kima.BINARIESmodel(True, 0, data)

    # this one should work
    data = kima.RVData(path_to_test_data("simulated2.txt"), double_lined=True)
    m = kima.BINARIESmodel(True, 0, data)


def test_ETVmodel(path_to_test_data):
    D = kima.ETVData(path_to_test_data("simulated3.txt"))
    m = kima.ETVmodel(True, 0, D)


def test_GAIAmodel(path_to_test_data):
    D = kima.GAIAdata(path_to_test_data("simulated4.txt"))
    m = kima.GAIAmodel(True, 0, D)


def test_RVGAIAmodel(path_to_test_data):
    D1 = kima.GAIAdata(path_to_test_data("simulated4.txt"))
    D2 = kima.RVData(path_to_test_data("simulated1.txt"))
    m = kima.RVGAIAmodel(True, 0, D1, D2)


def test_run(cleanup_after_running, path_to_test_data):
    data = kima.RVData(path_to_test_data("simulated1.txt"))

    m = kima.RVmodel(True, 0, data)
    kima.run(m, steps=1)

    m = kima.GPmodel(True, 0, data)
    kima.run(m, steps=1)

    data = kima.RVData(path_to_test_data("simulated2.txt"), indicators=["i", "j"])
    m = kima.RVFWHMmodel(True, 0, data)
    kima.run(m, steps=1)


@pytest.mark.parametrize('include, KO, nKO, TR, nTR, nP',
                         [
                            ((1, 4), True, 1, True, 1, 0),
                            ((1, 2, 3, 4, 5, 6), True, 2, True, 3, 1),
                            ((1, 2, 4, 6), True, 1, True, 2, 1),
                         ])
@pytest.mark.mpl_image_compare
def test_phase_plot(include, KO, nKO, TR, nTR, nP):
    from kima.pykima.display import phase_plot
    from kima.pykima.results import data_holder, KimaResults
    from kima import MODELS
    from common import create_data

    class FakeResult:
        max_components = 2
        model = MODELS.RVmodel
        has_gp = studentt = False
        full_model = KimaResults.full_model
        eval_model = KimaResults.eval_model
        planet_model = KimaResults.planet_model
        stochastic_model = KimaResults.stochastic_model
        residuals = KimaResults.residuals
        n_instruments = 1
        instruments = 'data'
        multi = arbitrary_units = save_plots = False
        return_figs = True
        @property
        def _time_overlaps(self):
            raise ValueError

    data, pars = create_data(plot=False, include=include)
    par1, par2, par3, par4, par5, par6, trend = pars
    d = data_holder()
    d.t, d.y, d.e = data
    d.N = d.t.size
    d.obs = np.ones_like(d.t, dtype=int)
    
    res = FakeResult()
    res.trend, res.degree = True, 1
    res.KO, res.nKO = KO, nKO
    res.TR, res.nTR = TR, nTR
    res.total_parameters = 2 + res.nKO*5 + res.nTR*5 + 1 + 2*5 + 1
    res.posterior_sample = np.empty((1, res.total_parameters + 3))
    res.data = d
    res.M0_epoch = d.t[0]
    res.indices = {
        'jitter': slice(0, 1, None),
        'trend': slice(1, 2, None),
        'KOpars': slice(2, 2 + res.nKO*5, None),
        'TRpars': slice(2 + res.nKO*5, 2 + res.nKO*5 + res.nTR*5, None),
        'np': 2+res.nKO*5+res.nTR*5+2,
        'planets': slice(2+res.nKO*5+res.nTR*5+2+1, 2+res.nKO*5+res.nTR*5+2+1 + 10, None),
        'planets.P': slice(2+res.nKO*5+res.nTR*5+2+1 + 0, 2+res.nKO*5+res.nTR*5+2+1 + 2, None),
        'planets.K': slice(2+res.nKO*5+res.nTR*5+2+1 + 2, 2+res.nKO*5+res.nTR*5+2+1 + 4, None),
        'planets.φ': slice(2+res.nKO*5+res.nTR*5+2+1 + 4, 2+res.nKO*5+res.nTR*5+2+1 + 6, None),
        'planets.e': slice(2+res.nKO*5+res.nTR*5+2+1 + 6, 2+res.nKO*5+res.nTR*5+2+1 + 8, None),
        'planets.w': slice(2+res.nKO*5+res.nTR*5+2+1 + 8, 2+res.nKO*5+res.nTR*5+2+1 + 10, None),
        'vsys': -1
    }

                # jitter, slope
    p = np.array([0.0,    trend[0]])

    if res.nKO == 1:
        p = np.r_[p, [par4['P'], par4['K'], par4['M0'], par4['e'], par4['w']]]
    elif res.nKO == 2:
        # KO:         P1, P2, K1,  K2,  φ1,  φ2,  e1,  e2,  w1,  w2
        p = np.r_[p, [par4['P'], par5['P'], 
                      par4['K'], par5['K'], 
                      par4['M0'], par5['M0'], 
                      par4['e'], par5['e'], 
                      par4['w'], par5['w']]
        ]

    if res.nTR == 1:
        p = np.r_[p, [par1['P'], par1['K'], par1['Tc'], par1['e'], par1['w']]]
    elif res.nTR == 2:
        # TR:         P1, P2, K1, K2,  Tc1,           Tc2,           e1,  e2,  w1,  w2
        p = np.r_[p, [par1['P'], par2['P'],
                      par1['K'], par2['K'],
                      par1['Tc'], par2['Tc'],
                      par1['e'], par2['e'],
                      par1['w'], par2['w']]
        ]
    elif res.nTR == 3:
        p = np.r_[p, [par1['P'], par2['P'], par3['P'],
                      par1['K'], par2['K'], par3['K'],
                      par1['Tc'], par2['Tc'], par3['Tc'],
                      par1['e'], par2['e'], par3['e'],
                      par1['w'], par2['w'], par3['w']]
        ]
    
                # ndim, maxnp, np
    p = np.r_[p, [5,    2,     nP]]

    if nP == 0:
        #              P1,  P2,  K1,  K2,  φ1,  φ2,  e1,  e2,  w1,  w2
        p = np.r_[p, [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]]
    elif nP == 1:
        #            P1,         P2,  K1,        K2,  φ1,         φ2,  e1,        e2,  w1,        w2
        p = np.r_[p, [par6['P'], 0.0, par6['K'], 0.0, par6['M0'], 0.0, par6['e'], 0.0, par6['w'], 0.0]]        

    #             staleness, vsys
    p = np.r_[p, [0,         trend[1]]]

    fig = phase_plot(res, p)
    return fig