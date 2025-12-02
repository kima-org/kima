from copy import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
from scipy.stats import gaussian_kde
from scipy.stats._continuous_distns import reciprocal_gen
from scipy.signal import find_peaks
from astropy.timeseries.periodograms.lombscargle.core import LombScargle

from .. import MODELS
from ..postkepler import Kfroma0
from .analysis import get_bins, np_bayes_factor_threshold, find_outliers
from .analysis import get_planet_mass_and_semimajor_axis
from .utils import (get_prior, hyperprior_samples, percentile68_ranges_latex,
                    wrms, get_instrument_name)
from .utils import mjup2msun, mjup2mearth
from .utils import distribution_rvs, distribution_support

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


def make_plots(res, options, save_plots=False):
    res.save_plots = save_plots
    if options == 'all':  # can be 'all' if called from the interpreter
        options = ('1', '2', '3', '4', '5', '6p', '7', '8')

    allowed_options = {
        # keys are allowed options (strings)
        # values are lists
        #   first item in the list can be a callable, a tuple of callables
        #   or a str
        #       if a callable, it is called
        #       if a tuple of callables, they are all called without arguments
        #       if a str, it is exec'd in globals(), locals()
        #   second item in the list is a dictionary
        #       each entry is an argument with which the callable is called
        '1': [res.plot_posterior_np, {}],
        '2': [res.plot_posterior_period, {'show_prior': True}],
        '3': [res.plot_PKE, {}],
        '4': [res.plot_gp, {}],
        '5': [res.plot_gp_corner, {}],
        '6': [res.plot_random_samples, {'show_vsys': True}],
        '6p': [
            'res.plot_random_samples(show_vsys=True);'\
            'res.phase_plot(res.maximum_likelihood_sample(Np=np_bayes_factor_threshold(res)))',
            {}
        ],
        '7': [
            (res.hist_vsys,
             res.hist_jitter,
             res.hist_trend,
             res.hist_correlations
             ), {}],
        '8': [res.hist_MA, {}],
    }

    for item in allowed_options.items():
        if item[0] in options:
            methods = item[1][0]
            kwargs = item[1][1]
            if isinstance(methods, tuple):
                [m() for m in methods]
            elif isinstance(methods, str):
                exec(methods)
            else:
                methods(**kwargs)


def plot_posterior_np(res, ax=None, errors=False, show_ESS=True,
                      show_detected=True, show_probabilities=False,
                      show_title=True, verbose=True, **kwargs):
    """ Plot the histogram of the posterior for Np

    Args:
        res (kima.KimaResults):
            The `KimaResults` instance
        ax (matplotlib.axes._axes.Axes, optional):
            An existing matplotlib axes where to draw the plot
        errors (bool, optional):
            Whether to estimate and display errors on the Np posterior.
        show_ESS (bool, optional):
            Display the effective sample size on the plot.
        show_detected (bool, optional):
            Highlight the detected Np in the plot (from
            `np_bayes_factor_threshold`).
        show_probabilities (bool, optional):
            Display the probabilities on top of the histogram bars.
        show_title (bool, optional):
            Display the title on the plot
        verbose (bool, optional):
            Print the posterior ratios
        **kwargs:
            Keyword arguments to pass to the `ax.bar` method

    Returns:
        fig (matplotlib.figure.Figure): The matplotlib figure with the plot
    """
    if res.fix:
        print(f'The number of Keplerians is fixed (to {res.npmax}). plot_posterior_np doing nothing')
        return

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    bins = np.arange(res.max_components + 2)
    nplanets = res.posterior_sample[:, res.index_component]
    n, _ = np.histogram(nplanets, bins=bins)
    ax.bar(bins[:-1], n / res.ESS, zorder=2, **kwargs)

    if show_probabilities:
        for i, nn in enumerate(n):
            ax.text(bins[i], 0.08 + (nn / res.ESS), str(round(nn/res.ESS, 2)), 
                    rotation=0, ha='center', va='center', fontsize=8)

    if show_ESS:
        ax.text(0.05, 0.9, f'ESS: {res.ESS}', transform=ax.transAxes)

    if errors:
        # from scipy.stats import multinomial
        prob = n / res.ESS
        # errors = multinomial(res.ESS, prob).rvs(1000).std(axis=0)
        error_multinomial = np.sqrt(res.ESS * prob * (1 - prob)) / res.ESS
        ax.errorbar(bins[:-1],
                    n / res.ESS,
                    error_multinomial,
                    fmt='.',
                    ms=0,
                    capsize=3,
                    color='k')

    if show_detected:
        pt_Np = np_bayes_factor_threshold(res)
        ax.bar(pt_Np, n[pt_Np] / res.ESS, color='C1', zorder=2)

    xlim = (-0.5, res.max_components + 0.5)
    xticks = np.arange(res.max_components + 1)
    ax.set(
        xlabel='Number of Planets',
        ylabel='Number of Posterior Samples / ESS',
        xlim=xlim,
        xticks=xticks,
        title='Posterior distribution for $N_p$' if show_title else ''
    )

    nn = n[np.nonzero(n)]

    if verbose:
        print('Np probability ratios: ', nn.flat[1:] / nn.flat[:-1])

    if errors:
        from scipy.stats import multinomial
        rs = multinomial(res.ESS, prob).rvs(10000)
        print(23*' ', np.divide(rs[:, 1:], rs[:, :-1]).std(axis=0).round(0))

    if res.save_plots:
        filename = 'kima-showresults-fig1.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return ax.figure


def plot_posterior_period(res,
                          nbins=100,
                          bins=None,
                          plims=None,
                          logx=True,
                          kde=False,
                          kde_bw=None,
                          show_peaks=False,
                          show_prior=False,
                          show_year=True,
                          show_timespan=True,
                          show_aliases=False,
                          show_eta3=False,
                          include_known_object=False,
                          include_transiting_planet=False,
                          separate_colors=False,
                          return_bins=False,
                          mark_periods=None,
                          mark_periods_text=False,
                          **kwargs):
    """Plot the histogram (or the kde) of the posterior for the orbital period(s)

    Args:
        res (kima.KimaResults):
            The `KimaResults` instance
        nbins (int, optional):
            Number of bins in the histogram
        bins (array, optional):
            Histogram bins
        plims (tuple, optional):
            Period limits, as (pmin, pmax)
        logx (bool, optional):
            Plot the x axis in lograithmic scale
        kde (bool, optional):
            Show a kernel density estimation (KDE) instead of the histogram
        kde_bw (float, optional):
            Bandwith for the KDE
        show_peaks (bool, optional):
            Try to identify prominent peaks in the posterior
        show_prior (bool, optional):
            Plot the prior together with the posterior
        show_year (bool, optional):
            Show a vertical line at 1 year
        show_timespan (bool, optional):
            Show a vertical line at the timespan of the data
        show_aliases (bool or int, optional):
            Show daily and yearly aliases for top peak(s)
        separate_colors (bool, optional):
            Show different Keplerians as different colors
        return_bins (bool, optional):
            Return the bins used for the histogram
        mark_periods (list, tuple, array, optional):
            Mark specific periods in the plot
        mark_periods_text (bool, optional):
            Write the period values next to the marker

    Returns:
        fig (matplotlib.figure.Figure): The matplotlib figure with the plot
    """
    # if no known_object or not showing known_object periods
    cond = not res.KO or not include_known_object
    # and if no transiting_planet or not showing transiting_planet periods
    cond &= not res.TR or not include_transiting_planet
    # there might be no planet periods to show
    if cond:
        if res.max_components == 0:
            print('Model has no planets! plot_posterior_period() doing nothing...')
            return

        if res.posteriors.P.size == 0:
            print('None of the posterior samples have planets!', end=' ')
            print('plot_posterior_period() doing nothing...')
            return

    # if res.log_period:
    #     T = np.exp(res.T)
    # else:
    #     T = res.T

    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
        fig = ax.figure
    else:
        fig, ax = plt.subplots(1, 1)

    kwline = {'ls': '--', 'lw': 1.5, 'alpha': 0.3, 'zorder': -1}

    if kde:
        T = res.T
        NN = 3000
        kdef = gaussian_kde(T, bw_method=kde_bw)
        if plims is None:
            if logx:
                xx = np.logspace(np.log10(T.min()), np.log10(T.max()), NN)
                y = kdef(xx)
                ax.semilogx(xx, y)
            else:
                xx = np.linspace(T.min(), T.max(), NN)
                y = kdef(xx)
                ax.plot(xx, y)
        else:
            a, b = plims
            if logx:
                xx = np.logspace(np.log10(a), np.log10(b), NN)
                y = kdef(xx)
                ax.semilogx(xx, y)
            else:
                xx = np.linspace(a, b, NN)
                y = kdef(xx)
                ax.plot(xx, y)

        # show the limits of the kde evaluation
        ax.vlines([xx.min(), xx.max()], ymin=0, ymax=0.03, color='k',
                  transform=ax.get_xaxis_transform())

        if show_prior:
            prior = get_prior(res.setup['priors.planets']['Pprior'])
            if isinstance(prior.dist, reciprocal_gen):
                # show pdf per logarithmic interval
                ax.plot(xx, xx * prior.pdf(xx), 'k', label='prior')
            else:
                ax.plot(xx, prior.pdf(xx), 'k', label='prior')

        if show_peaks and find_peaks:
            peaks, _ = find_peaks(y, prominence=0.1)
            for peak in peaks:
                s = r'P$\simeq$%.2f' % xx[peak]
                ax.text(xx[peak], y[peak], s, ha='left')

    else:
        if bins is None:
            if plims is None:
                bins = get_bins(res, nbins=nbins)
            else:
                bins = get_bins(res, *plims, nbins=nbins)

        if 'bottom' in kwargs:
            bottoms = np.full_like(bins, kwargs.pop('bottom'))
        else:
            bottoms =  np.zeros_like(bins)

        for i in range(res.max_components):
            m = res.posterior_sample[:, res.indices['np']] == i + 1
            T = res.posterior_sample[m, res.indices['planets.P']]
            T = T[:, :i + 1].ravel()

            counts, bin_edges = np.histogram(T, bins=bins)

            if separate_colors:
                color = None
            else:
                color = kwargs.get('color', 'C0')

            ax.bar(x=bin_edges[:-1], height=counts / res.ESS, width=np.ediff1d(bin_edges),
                   bottom=bottoms[:-1], align='edge', alpha=0.8, color=color)

            bottoms += np.append(counts / res.ESS, 0)

        if include_known_object and res.KO:
            for i in range(res.nKO):
                counts, bin_edges = np.histogram(res.KOpars[:, i], bins=bins)
                ax.bar(x=bin_edges[:-1], height=counts / res.ESS, width=np.ediff1d(bin_edges),
                       align='edge', alpha=0.8, color='k')
                bottoms += np.append(counts / res.ESS, 0)
        
        if include_transiting_planet and res.TR:
            for i in range(res.nTR):
                counts, bin_edges = np.histogram(res.TRpars[:, i], bins=bins)
                ax.bar(x=bin_edges[:-1], height=counts / res.ESS, width=np.ediff1d(bin_edges),
                       align='edge', alpha=0.8)#, color='k')
                bottoms += np.append(counts / res.ESS, 0)

        # save maximum peak(s)
        peaki = np.argsort(bottoms)[-10:][::-1]
        peakP = np.array([bins[pi:pi+2].mean() for pi in peaki])

        if show_peaks and find_peaks:
            peaks, _ = find_peaks(bottoms, prominence=0.1)
            for peak in peaks:
                s = r'P$\simeq$%.2f' % bin_edges[peak]
                ax.text(bin_edges[peak], bottoms[peak], s, ha='left')

        # ax.hist(T, bins=bins, alpha=0.8, density=density)

        if show_prior:
            kwprior = {"alpha": 0.15, "color": "k", "zorder": -1, "label": "prior"}
            if 'Pprior' in res.priors:
                if res.hyperpriors:
                    P = hyperprior_samples(T.size)
                else:
                    P = distribution_rvs(res.priors['Pprior'], res.ESS)

                counts, bin_edges = np.histogram(P, bins=bins)
                ax.bar(x=bin_edges[:-1], height=res.npmax * counts / res.ESS, 
                       width=np.ediff1d(bin_edges), align="edge", **kwprior)

            if include_transiting_planet and res.TR:
                for i in range(res.nTR):
                    if hasattr(res._priors.TR, f'P{i}'):
                        P = res._priors.TR.get_samples(f'P{i}', res.ESS)
                    else:    
                        P = distribution_rvs(res.priors[f'TR_Pprior_{i}'], res.ESS)

                    counts, bin_edges = np.histogram(P, bins=bins)
                    ax.bar(x=bin_edges[:-1], height=counts / res.ESS, 
                           width=np.ediff1d(bin_edges), align="edge", **kwprior)



    if show_year:  # mark 1 year
        year = 365.25
        ax.axvline(x=year, color='r', label='1 year', **kwline)

    if show_timespan:  # mark the timespan of the data
        try:
            ax.axvline(x=np.ptp(res.data.t), color='k', label='time span', **kwline)
        except AttributeError:
            pass

    if show_aliases is not None:  # mark daily and yearly aliases of top peak
        from .analysis import aliases
        ymax = 1.1 * ax.get_ylim()[1]
        if isinstance(show_aliases, int):
            peakP = peakP[:show_aliases]

        alias_year, alias_solar_day, _ = aliases(peakP)
        ax.plot(peakP, np.full_like(peakP, ymax), 'v', color='orange')
        ax.vlines(alias_year, 0, ymax, color='orange', ls='--', alpha=0.1)
        ax.vlines(alias_solar_day, 0, ymax, color='orange', ls='--', alpha=0.1)

    if show_eta3:
        if res.has_gp:
            counts, bin_edges = np.histogram(res.posteriors.η3, bins=bins)
            ax.bar(x=bin_edges[:-1], height=counts / res.ESS, width=np.ediff1d(bin_edges),
                   bottom=bottoms[:-1], align='edge', alpha=0.8, color='plum',
                   label=r'GP $\eta_3$')
        else:
            print('Model does not have GP! show_eta3=True doing nothing...')


    if kwargs.get('legend', True):
        ax.legend()
    
    ax.set_xscale('log' if logx else 'linear')
    
    if kwargs.get('labels', True):
        ylabel = 'KDE density' if kde else 'Number of posterior samples / ESS'
        ax.set(xlabel=r'Period [days]', ylabel=ylabel)
    
    title = kwargs.get('title', True)
    if title:
        if isinstance(title, str):
            ax.set_title(title)
        else:
            ax.set_title('Posterior distribution for the orbital period(s)')
    
    if plims is not None:
        ax.set_xlim(plims)
    else:
        ax.autoscale()
    
    if mark_periods is not None:
        ax.plot(mark_periods, np.full_like(mark_periods, 1.1), 'rv') #Why is this 
        if mark_periods_text:
            for p in mark_periods:
                ax.text(1.1 * p, 1.1, f'{p:.2f} days')

    if res.save_plots:
        filename = 'kima-showresults-fig2.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        if return_bins:
            return fig, bins
        else:
            return fig


def plot_PKE(res, mask=None, include_known_object=False, include_transiting_planet=False,
             show_prior=False, show_aliases=None, reorder_P=False, sort_by_increasing_P=False,
             points=True, colors_np=True, gridsize=50, **kwargs):
    """
    Plot the 2d histograms of the posteriors for semi-amplitude and orbital
    period and for eccentricity and orbital period. If `points` is True, plot
    each posterior sample, else plot hexbins
    """
    # if no known_object or not showing known_object periods
    cond = not res.KO or not include_known_object
    # and if no transiting_planet or not showing transiting_planet periods
    cond &= not res.TR or not include_transiting_planet
    # there might be no planet periods to show
    if cond:
        if res.max_components == 0:
            print('Model has no planets! plot_PKE() doing nothing...')
            return

        if res.posteriors.P.size == 0:
            print('None of the posterior samples have planets!', end=' ')
            print('plot_PKE() doing nothing...')
            return

    if mask is None:
        if reorder_P:
            from .analysis import reorder_P
            post = reorder_P(res)
            P = post.P.copy()
            K = post.K.copy()
            E = post.e.copy()
        elif sort_by_increasing_P:
            from .analysis import sort_planet_samples
            post = sort_planet_samples(res)
            P = post.P.copy()
            K = post.K.copy()
            E = post.e.copy()
        else:
            P = res.posteriors.P.copy()
            K = res.posteriors.K.copy()
            E = res.posteriors.e.copy()
    else:
        P = res.posteriors.P[mask].copy()
        K = res.posteriors.K[mask].copy()
        E = res.posteriors.e[mask].copy()

    include_known_object = include_known_object and res.KO

    if include_known_object:
        if mask is None:
            KOpars = res.posterior_sample[:, res.indices['KOpars']]
        else:
            KOpars = res.posterior_sample[mask, res.indices['KOpars']]
        P_KO = np.hstack(KOpars[:, 0 * res.nKO:1 * res.nKO])
        K_KO = np.hstack(KOpars[:, 1 * res.nKO:2 * res.nKO])
        E_KO = np.hstack(KOpars[:, 3 * res.nKO:4 * res.nKO])

    include_transiting_planet = include_transiting_planet and res.TR

    if include_transiting_planet:
        if mask is None:
            TRpars = res.posterior_sample[:, res.indices['TRpars']]
        else:
            TRpars = res.posterior_sample[mask, res.indices['TRpars']]
        P_TR = np.hstack(TRpars[:, 0 * res.nTR:1 * res.nTR])
        K_TR = np.hstack(TRpars[:, 1 * res.nTR:2 * res.nTR])
        E_TR = np.hstack(TRpars[:, 3 * res.nTR:4 * res.nTR])

    if res.log_period:
        P = np.exp(P)

    if 'ax1' in kwargs and 'ax2' in kwargs:
        ax1, ax2 = kwargs.pop('ax1'), kwargs.pop('ax2')
        fig = ax1.figure
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)

    # the y scale in loglog looks bad if the semi-amplitude doesn't have
    # high dynamic range; the threshold of 30 is arbitrary
    Khdr_threshold = 30

    if points:
        kw = dict(markersize=2, zorder=2, alpha=0.1, picker=5)
        kw = {**kw, **kwargs}

        # plot known_object first so it always has the same color
        if include_known_object:
            ax1.semilogx(P_KO, K_KO, '.', markersize=2, zorder=2)
            ax2.semilogx(P_KO, E_KO, '.', markersize=2, zorder=2)

        # plot transiting_planet first so it always has the same color
        if include_transiting_planet:
            ax1.semilogx(P_TR, K_TR, '.', markersize=2, zorder=2)
            ax2.semilogx(P_TR, E_TR, '.', markersize=2, zorder=2)

        if not colors_np:
            P = P.ravel()
            K = K.ravel()
            E = E.ravel()

        if K.size > 1 and np.ptp(K) > Khdr_threshold:
            ax1.loglog(P, K, '.', **kw)
        else:
            ax1.semilogx(P, K, '.', **kw)

        ax2.semilogx(P, E, '.', **kw)


    else:
        if K.size > 1 and np.ptp(K) > 30:
            cm = ax1.hexbin(P[P != 0.0], K[P != 0.0], gridsize=gridsize, bins='log', xscale='log',
                            yscale='log', cmap=plt.get_cmap('coolwarm_r'))
        else:
            cm = ax1.hexbin(P[P != 0.0], K[P != 0.0], gridsize=gridsize, bins='log', xscale='log',
                            yscale='linear', cmap=plt.get_cmap('coolwarm_r'))

        ax2.hexbin(P[P != 0.0], E[P != 0.0], gridsize=gridsize, bins='log', xscale='log',
                   cmap=plt.get_cmap('coolwarm_r'))
    
        plt.colorbar(cm, ax=ax1)

    if show_prior:
        kw_prior = dict(ms=2, color='k', alpha=0.05, zorder=-10)
        n = prior_samples or max(10_000, res.ESS)
        
        if include_known_object:
            P_KO_prior, K_KO_prior, E_KO_prior = [], [], []
            for i in range(res.nKO):
                if f'KO_Pprior_{i}' in res.priors:
                    P_KO_prior.append(distribution_rvs(res.priors[f'KO_Pprior_{i}'], n))
                    K_KO_prior.append(distribution_rvs(res.priors[f'KO_Kprior_{i}'], n))
                    E_KO_prior.append(distribution_rvs(res.priors[f'KO_eprior_{i}'], n))
                else:
                    break
            ax1.plot(np.ravel(P_KO_prior), np.ravel(K_KO_prior), '.', **kw_prior)
            ax2.plot(np.ravel(P_KO_prior), np.ravel(E_KO_prior), '.', **kw_prior)

        try:
            P_prior = distribution_rvs(res.priors['Pprior'], n)
            K_prior = distribution_rvs(res.priors['Kprior'], n)
            E_prior = distribution_rvs(res.priors['eprior'], n)
            ax1.plot(P_prior, K_prior, '.', **kw_prior)
            ax2.plot(P_prior, E_prior, '.', **kw_prior)
        except KeyError:
            pass

    if show_aliases is not None:  # mark daily and yearly aliases of top peak
        from .analysis import aliases
        
        point, = ax1.plot([], [], 'v', color='k')

        def mark_alias(event):
            print(event.ind[0])
            print(res.posteriors.P[event.ind[0]])
        
        fig.canvas.callbacks.connect('pick_event', mark_alias)
        # # get maximum peak(s)
        # bins = get_bins(res, nbins=200)
        # counts, _ = np.histogram(P, bins=bins)
        # peaki = np.argsort(counts)[-10:][::-1]
        # peakP = np.array([bins[pi:pi+2].mean() for pi in peaki])
        # ymax = 1.1 * ax1.get_ylim()[1]

        # if isinstance(show_aliases, int):
        #     peakP = peakP[:show_aliases]

        # alias_year, alias_solar_day, alias_sidereal_day = aliases(peakP)
        # for ax in (ax1, ax2):
        #     ax.vlines(alias_year, 0, ymax, color='k', ls='--', alpha=0.1)
        #     ax.vlines(alias_solar_day, 0, ymax, color='k', ls='--', alpha=0.1)
        #     ax.vlines(alias_sidereal_day, 0, ymax, color='k', ls='--', alpha=0.1)
        # ax.set_xlim(np.min(alias_solar_day), None)

    ax1.set(ylabel='Semi-amplitude [m/s]',
            title='Joint posterior semi-amplitude $-$ orbital period')
    ax2.set(ylabel='Eccentricity', xlabel='Period [days]',
            title='Joint posterior eccentricity $-$ orbital period',
            ylim=[0, 1])

    # if show_prior:
    #     try:
    #         minx, maxx = 0, np.inf
    #         maxy = np.inf
    #         if include_known_object:
    #             for i in range(res.nKO):
    #                 _1, _2 = res.priors[f'KO_Pprior_{i}'].support()
    #                 minx = max(minx, _1)
    #                 maxx = min(maxx, _2)
    #                 maxy = min(maxy, res.priors[f'KO_Kprior_{i}'].support()[1])
    #         _1, _2 = res.priors['Pprior'].support()
    #         minx = max(minx, _1)
    #         maxx = min(maxx, _2)
    #         maxy = min(maxy, res.priors['Kprior'].support()[1])
    #         ax1.set(xlim=(minx, maxx), ylim=(None, maxy))
    #     except (AttributeError, KeyError, ValueError):
    #         pass

    if res.save_plots:
        filename = 'kima-showresults-fig3.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig


def plot_gp(res, Np=None, ranges=None, show_prior=False, fig=None, axs=None,
               **hist_kwargs):
    """
    Plot histograms for the GP hyperparameters. If Np is not None, highlight
    the samples with Np Keplerians.
    """
    if not res.has_gp:
        print('Model does not have GP! plot_gp() doing nothing...')
        return

    # dispatch if RVFWHMmodel
    if res.model in (MODELS.RVFWHMmodel, MODELS.RVFWHMRHKmodel):
        return plot_gp_rvfwhm(res, Np, ranges, show_prior, fig, **hist_kwargs)

    n = res.etas.shape[1]
    available_etas = [f'eta_{i}' for i in range(1, n + 1)]
    units = ['m/s', 'days', 'days', None]
    # print(n)
    # labels = [rf'$\eta_{i}$' + f'[{units[i-1]}]' if units[i-1] else '' for i in range(1, n + 1)]

    if ranges is None:
        ranges = n * [None]

    if Np is not None:
        m = res.posterior_sample[:, res.index_component] == Np

    nplots = int(np.ceil(n / 2))
    if axs is None:
        if fig is None:
            fig, axes = plt.subplots(2, nplots, constrained_layout=True)
        else:
            axes = np.array(fig.axes)
    else:
        axes = axs
        fig = axes.ravel()[0].figure
    assert len(axes.ravel()) >= 2 * nplots, 'figure has too few axes!'


    hist_kwargs.setdefault('density', True)

    for i, eta in enumerate(available_etas):
        ax = np.ravel(axes)[i]


        try:
            val = getattr(res.posteriors, f'η{i+1}')
        except AttributeError:
            val = res.etas[:, i]

        estimate = percentile68_ranges_latex(val)
        ax.hist(val, bins='doane', range=ranges[i], label=estimate, **hist_kwargs)

        if show_prior:
            previous_xlim = ax.get_xlim()
            prior = res.priors[f'eta{i+1}_prior']
            ax.hist(distribution_rvs(prior, res.ESS), bins='doane', 
                    color='k', alpha=0.2, density=True)
            ax.set_xlim(previous_xlim)

        ax.legend()
        label = available_etas[i].replace('eta', r'\eta')
        label = f'${label}$'
        if units[i] is not None:
            label += f' [{units[i]}]'
        ax.set(xlabel=label, ylabel='posterior')

    for j in range(i + 1, 2 * nplots):
        np.ravel(axes)[j].axis('off')

    if res.save_plots:
        filename = 'kima-showresults-fig4.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig


def plot_gp_rvfwhm(res, Np=None, ranges=None, show_prior=False, fig=None,
                   **hist_kwargs):
    """
    Plot histograms for the GP hyperparameters. If Np is not None, highlight
    the samples with Np Keplerians. 
    """
    if not res.has_gp:
        print('Model does not have GP! plot_gp() doing nothing...')
        return

    FWHMmodel = res.model is MODELS.RVFWHMmodel
    FWHMRHKmodel = res.model is MODELS.RVFWHMRHKmodel

    # n = res.etas.shape[1]
    labels = ('η1', 'η2', 'η3', 'η4')

    if ranges is None:
        ranges = len(labels) * [None]

    if Np is not None:
        m = res.posterior_sample[:, res.index_component] == Np

    if FWHMmodel:
        fig, axs = plt.subplot_mosaic(
            [
                ['η1RV', 'η2'], 
                ['η1RV', 'η2'], 
                ['η1RV', 'η3'],
                ['η1FW', 'η3'],
                ['η1FW', 'η4'],
                ['η1FW', 'η4'],
            ],
        figsize=(8, 6), constrained_layout=True)

    elif FWHMRHKmodel:
        fig, axs = plt.subplot_mosaic(
            [
                ['η1RV', 'η1FW', 'η1RHK'], 
                ['η2', 'η3', 'η4']
            ],
        figsize=(8, 6), constrained_layout=True)

    histkw = dict(density=True, bins='doane')
    histkw2 = {**histkw, **{'histtype':'step'}}
    allkw = dict(yticks=[])

    estimate = percentile68_ranges_latex(res.etas[:, 0]) + ' m/s'
    axs['η1RV'].hist(res.etas[:, 0], **histkw)
    axs['η1RV'].set_title(estimate, loc='right', fontsize=10)
    axs['η1RV'].set(xlabel=r'$\eta_1$ RV [m/s]', ylabel='posterior', **allkw)
    axs['η1RV'].set_xlim((0, None))
    j = 1

    estimate = percentile68_ranges_latex(res.etas[:, 1]) + ' m/s'
    axs['η1FW'].hist(res.etas[:, 1], label=estimate, **histkw)
    axs['η1FW'].set_title(estimate, loc='right', fontsize=10)
    axs['η1FW'].set(xlabel=r'$\eta_1$ FWHM [m/s]', ylabel='posterior', **allkw)
    axs['η1FW'].set_xlim((0, None))
    j = 2

    if FWHMRHKmodel:
        estimate = percentile68_ranges_latex(res.etas[:, 2])
        axs['η1RHK'].hist(res.etas[:, 2], **histkw)
        axs['η1RHK'].set_title(estimate, loc='right', fontsize=10)
        axs['η1RHK'].set(xlabel=r"$\eta_1$ R'$_{HK}$", ylabel='posterior', **allkw)
        axs['η1RHK'].set_xlim((0, None))
        j = 3

    if show_prior:
        kw = dict(color='k', alpha=0.2, density=True, zorder=-1, bins='doane')
        prior = res.priors[f'eta1_prior']
        axs['η1RV'].hist(distribution_rvs(prior, res.ESS), **kw)
        prior = res.priors[f'eta1_fwhm_prior']
        axs['η1FW'].hist(distribution_rvs(prior, res.ESS), **kw)
        if FWHMRHKmodel:
            prior = res.priors[f'eta1_rhk_prior']
            axs['η1RHK'].hist(distribution_rvs(prior, res.ESS), **kw)


    from itertools import islice
    def grouper(n, iterable):
        iterable = iter(iterable)
        return iter(lambda: list(islice(iterable, n)), [])

    units = [' [days]', ' [days]', '']
    group = 3 if FWHMRHKmodel else 2

    for i, n in zip(grouper(group, res._GP_par_indices[j:]), (2, 3, 4)):
        ax = axs[f'η{n}']

        locs = ('left', 'center', 'right')
        for k, j in enumerate(np.unique(i)):
            estimate = percentile68_ranges_latex(res.etas[:, j])
            ax.set_title(estimate, loc=locs[k], fontsize=10, color=f'C{k}')
            ax.hist(res.etas[:, j], **histkw)
        
        if not getattr(res, f'share_eta{n}'):
            ax.text(0.96, 0.96, f'share_eta{n} = False', transform=ax.transAxes, 
                    fontsize=10, ha='right', va='top')

        ax.set(xlabel=fr'$\eta_{n}$' + units[n - 2], ylabel='posterior', **allkw)

        # TODO: implement show_prior ? 

    return fig


def plot_gp_corner(res, include_jitters=False, ranges=None):
    """ Corner plot for the GP hyperparameters """

    if not res.has_gp:
        print('Model does not have GP! plot_gp_corner() doing nothing...')
        return

    data = []
    labels = []

    if include_jitters:
        if res.multi:
            instruments = res.instruments
            if res.model == 'RVFWHMmodel':
                labels += [rf'$s_{{\rm {i}}}^{{\rm RV}}$' for i in instruments]
                labels += [rf'$s_{{\rm {i}}}^{{\rm FWHM}}$' for i in instruments]
            elif res.model == 'RVFWHMRHKmodel':
                labels += [rf'$s_{{\rm {i}}}^{{\rm RV}}$' for i in instruments]
                labels += [rf'$s_{{\rm {i}}}^{{\rm FWHM}}$' for i in instruments]
                labels += [rf'$s_{{\rm {i}}}^{{\rm RHK}}$' for i in instruments]
            else:
                labels += [rf'$s_{{\rm {i}}}$' for i in instruments]

        data.append(res.jitter)

    if res.model == 'RVFWHMmodel':
        labels += [r'$\eta_1^{RV}$', r'$\eta_1^{FWHM}$']
        for i in range(2, res.n_hyperparameters):
            labels += [rf'$\eta_{i}$']
    elif res.model == 'RVFWHMRHKmodel':
        labels += [r'$\eta_1^{RV}$', r'$\eta_1^{FWHM}$', r'$\eta_1^{RHK}$']
        for i in range(2, res.n_hyperparameters):
            labels += [rf'$\eta_{i}$']
    else:
        labels += [rf'$\eta_{i+1}$' for i in range(res.n_hyperparameters)]

    data.append(res.etas)

    data = np.hstack(data)

    if data.shape[0] < data.shape[1]:
        print('Not enough samples to make the corner plot')
        return

    rc = {
        'font.size': 6,
    }
    with plt.rc_context(rc):
        fig = corner(data, show_titles=True, labels=labels, titles=labels,
                     plot_datapoints=True, plot_contours=False,
                     plot_density=False)

    fig.subplots_adjust(top=0.95, bottom=0.1, wspace=0, hspace=0)


def corner_all(res):
    post = np.c_[res.posteriors.jitter]

    if res.multi:
        post = np.c_[post, res.posteriors.offset]
    
    if res.KO:
        post = np.c_[post, res.KOpars]
    if hasattr(res, 'TR') and res.TR:
        post = np.c_[post, res.TRpars]

    post = np.c_[
        post, res.posteriors.P, res.posteriors.K, res.posteriors.φ, res.posteriors.e, res.posteriors.w
    ]

    if res.studentt:
        post = np.c_[post, res.nu]

    vsys = res.posteriors.vsys.copy()
    mean_vsys = int(res.posteriors.vsys.mean())
    subtracted_mean = False
    if abs(mean_vsys) > 100:
        vsys -= mean_vsys
        subtracted_mean = True
    post = np.c_[post, vsys]

    names = copy(res.parameters)

    # remove trailing zeroes if there's just one planet
    if res.npmax == 1:
        for _p in ('P', 'K', 'phi', 'ecc', 'w'):
            names[names.index(f'{_p}0')] = _p


    if res.multi:
        for i, inst in enumerate(res.instruments, start=1):
            names[names.index(f'jitter{i}')] = f'jitter {inst}'

    if subtracted_mean:
        names[names.index('vsys')] = f'vsys - {mean_vsys}'

    # remove parameters with no range (fixed)
    ind = np.ptp(post, axis=0) == 0
    post = post[:, ~ind]
    names = np.array(names)[~ind]

    hkw = dict(density=True)
    fig = plt.figure(figsize=(10, 10))#, constrained_layout=True)
    corner(post, fig=fig, color='k', hist_kwargs=hkw, labels=names, show_titles=True,
           plot_density=False, plot_contours=False, plot_datapoints=True)

    axs = np.array(fig.axes)
    for name, ax in zip(names, axs[::post.shape[1] + 1]): # diagonal axes
        ax.set_title(ax.get_title().replace(name + ' = ', ''))

    return fig

    n = max(10000, res.ESS)

    values = res.posterior_sample[:, res.indices['jitter']]
    prior_rvs = []
    for p in res.parameter_priors[res.indices['jitter']]:
        prior_rvs.append(p.rvs(n))

    for par in ('planets', 'vsys'):
        values = np.c_[values, res.posterior_sample[:, res.indices[par]]]
        prior = res.parameter_priors[res.indices[par]]
        if isinstance(prior, list):
            for p in prior:
                prior_rvs.append(p.rvs(n))
        else:
            prior_rvs.append(prior.rvs(n))

    hkw = dict(density=True)
    fig = corner(values, color='C0', hist_kwargs=hkw,
                 plot_density=False, plot_contours=False, plot_datapoints=True)
    xlims = [ax.get_xlim() for ax in fig.axes]

    hkw = dict(density=True, alpha=0.5)
    fig = corner(np.array(prior_rvs).T, fig=fig, color='k', hist_kwargs=hkw,
                 plot_density=False, plot_contours=False, plot_datapoints=False)

    for xlim, ax in zip(xlims, fig.axes):
        ax.set_xlim(xlim)


def corner_orbital(samples, labels=None, units=None, ranges=None, priors=None,
                   truths=None, Tc=False, degrees=False, wrap_M0=False, wrap_w=False, angles=True, 
                   fig=None, axs=None, color='k', cmap='Greys', force_ecc0_limit=False,
                   show_titles=True, show_lines=True, joint='hist2d',
                   prior_kwargs={}, true_value_label='', true_value_kwargs={},
                   add_smooth=False, upper=False):
    from .utils import percentile68_ranges, percentile68_ranges_latex
    from corner import hist2d
    from scipy.stats import gaussian_kde
    ns, n = samples.shape

    def _check(lst, name, default=''):
        if lst is None:
            lst = n * [default]
        else:
            assert len(lst) == n, \
                f'Number of {name} {len(lst)} must match number of parameters {n}'
        return lst

    labels = _check(labels, 'labels')
    units = _check(units, 'units')
    ranges = _check(ranges, 'ranges', None)
    priors = _check(priors, 'priors', None)
    truths = _check(truths, 'truths', None)

    _axs_provided = axs is not None

    style = 'seaborn-v0_8-deep'
    with plt.style.context(style):
        _axis_off_others = True

        if fig is None:
            if axs is None:
                fig, axs = plt.subplots(n, n, constrained_layout=not upper,
                                        gridspec_kw={'wspace': 0.1},
                                        figsize=(8, 6))
            else:
                axs = np.array(axs).reshape(n, n)
                fig = axs[0, 0].figure
                _axis_off_others = False
        else:
            if axs is None:
                axs = np.array(fig.axes).reshape(n, n)
            else:
                axs = np.array(axs).reshape(n, n)
                _axis_off_others = False

        if _axis_off_others:
            if upper:
                outside_diag_axs = np.tril(axs, -1).flatten()
            else:
                outside_diag_axs = np.triu(axs, 1).flatten()
            for ax in outside_diag_axs:
                if ax:
                    ax.axis('off')

        vars = samples.T

        # set titles
        if show_titles:
            if upper:
                title_axs = axs[0, :]
                for ax, var, label in zip(title_axs, vars, labels):
                    if np.ptp(var) == 0.0:
                        title = f'{label} = {var[0]}'
                    else:
                        title = ' = '.join([label, percentile68_ranges_latex(var, collapse=False)])
                    ax.set_title(title, fontsize=10)#, y=-0.4 if upper else 1)
            else:
                title_axs = np.diag(axs)
                for ax, var, label in zip(title_axs, vars, labels):
                    if np.ptp(var) == 0.0:
                        title = f'{label} = {var[0]}'
                    else:
                        title = ' = '.join([label, percentile68_ranges_latex(var, collapse=False)])
                    ax.set_title(title, fontsize=10)#, y=-0.4 if upper else 1)

        # set x labels
        if upper:
            xlabel_axs = np.diag(axs)
            for i, (ax, lab, un) in enumerate(zip(xlabel_axs, labels, units)):
                if un != '':
                    ax.set_xlabel(f'{lab} [{un}]')
                else:
                    ax.set_xlabel(lab)
        else:
            xlabel_axs = axs[-1, :]
            for i, (ax, lab, un) in enumerate(zip(xlabel_axs, labels, units)):
                if un != '':
                    ax.set_xlabel(f'{lab} [{un}]')
                else:
                    ax.set_xlabel(lab)

        # set y labels
        if upper:
            ylabel_axs = axs[:, -1]
            for i, (ax, lab, un) in enumerate(zip(ylabel_axs, labels[:-1], units[:-1])):
                ax.yaxis.set_label_position('right')
                if un != '':
                    ax.set_ylabel(f'{lab} [{un}]')
                else:
                    ax.set_ylabel(lab)
        else:
            ylabel_axs = axs[1:, 0]
            for i, (ax, lab, un) in enumerate(zip(ylabel_axs, labels[1:], units[1:])):
                if un != '':
                    ax.set_ylabel(f'{lab} [{un}]')
                else:
                    ax.set_ylabel(lab)

        # remove x ticks
        if upper:
            no_xtick_axs = np.triu(axs, 1)
            for ax in no_xtick_axs.flatten():
                if ax:
                    ax.set_xticklabels([])
        else:
            no_xtick_axs = np.tril(axs)[:-1, :]
            for ax in no_xtick_axs.flatten():
                if ax:
                    ax.set_xticklabels([])

        # remove y ticks
        if upper:
            ytick_right = np.triu(axs, 1)
            for ax in ytick_right.flatten():
                if ax:
                    ax.yaxis.tick_right()
            for ax in axs[:, :-1].flatten():
                ax.set_yticklabels([])
            # no_ytick_axs = np.triu(axs, 1)
            # for ax in no_ytick_axs.flatten():
            #     if ax:
            #         ax.set_yticklabels([])
        else:
            no_ytick_axs = axs[1:, 1:]
            for ax in no_ytick_axs.flatten():
                ax.set_yticklabels([])

        diag_axs = np.diag(axs)
        for i, (var, label, ax) in enumerate(zip(samples.T, labels, diag_axs)):
            ax.hist(var, density=True, histtype='step', bins='doane',
                    color=color, range=ranges[i], label='posterior')

            if add_smooth:
                # 'scott', 'silverman',
                kde = gaussian_kde(var, bw_method=0.5)
                xlim = ranges[i] or ax.get_xlim()
                x = np.linspace(*xlim, 100)
                ax.plot(x, kde(x), lw=2, zorder=-1)

            # if show_titles:
            #     title = ' = '.join([label, percentile68_ranges_latex(var)])
            #     ax.set_title(title, fontsize=10, y=-0.4 if upper else 1)
            ax.set_yticks([])

            # if upper or ax != diag_axs[-1]:
            #     ax.set_xticklabels([])

            ax.margins(x=0)

            if priors[i] is not None:
                xlim = ax.get_xlim()
                prior_kwargs.setdefault('color', 'C0')
                prior_kwargs.setdefault('alpha', 0.2)
                prior_kwargs.setdefault('label', 'prior')
                ax.hist(priors[i], density=True, bins='doane', 
                        # restricting the priors to the range doesn't look good...
                        # range=ranges[i], 
                        **prior_kwargs)
                ax.set_xlim(xlim)
            
            # print(truths[i])
            if truths[i] is not None:
                true_value_kwargs.setdefault('color', 'g')
                true_value_kwargs.setdefault('ls', '-')
                true_value_kwargs.setdefault('lw', 2)
                label = true_value_label if true_value_label != '' else None
                ax.axvline(truths[i], label=label, **true_value_kwargs)

        if show_lines:
            line_kws = dict(color='C1', linestyle='--')
            for i, var in enumerate(samples.T):
                me, pl, mi = percentile68_ranges(var)
                if upper:
                    v_axs = axs[:i+1, i].flatten()
                    h_axs = axs[i, i+1:].flatten()
                else:
                    v_axs = axs[i:, i].flatten()
                    h_axs = axs[i, :i].flatten()

                for ax in v_axs:
                    ax.axvline(me, **line_kws, alpha=0.4)
                    ax.axvline(me + pl, **line_kws, alpha=0.15)
                    ax.axvline(me - mi, **line_kws, alpha=0.15)

                if upper or i > 0:
                    for ax in h_axs:
                        ax.axhline(me, **line_kws, alpha=0.4)
                        ax.axhline(me + pl, **line_kws, alpha=0.15)
                        ax.axhline(me - mi, **line_kws, alpha=0.15)

        for ax in axs.flatten():
            ax.minorticks_on()

        for i in range(1, n):
            for j in range(i):
                ax = axs[j, i] if upper else axs[i, j]
                x = samples[:, i] if upper else samples[:, j]
                y = samples[:, j] if upper else samples[:, i]
                _, binsx = np.histogram(x, bins='doane', range=ranges[j])
                _, binsy = np.histogram(y, bins='doane')
                if joint == 'hist2d':
                    ax.hist2d(x, y, bins=(binsx, binsy), density=True, cmap=cmap)
                elif joint == 'contour':
                    hist2d(x, y, #levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-4.5)), 
                           color=color,
                           bins=(binsx.size, binsy.size), contour_kwargs={'cmap': cmap, 'colors': None}, 
                           ax=ax, plot_datapoints=False, plot_density=False, smooth=True)
                    # counts, xbins, ybins = np.histogram2d(x, y, bins=(binsx, binsy), density=True)
                    # print(_calc_levels(counts, (1 - np.exp(-0.5), )))
                    # ax.contour(counts.T, xbins, ybins,
                    #            levels=_calc_levels(counts, (1 - np.exp(-0.5), )),
                    #         #    levels=(0.1, 2, 20), 
                    #         #    levels=2,
                    #            cmap=cmap, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()])

        # if upper:
        #     outside_diag_axs = np.triu(axs, 1).flatten()
        #     for ax in outside_diag_axs:
        #         if ax:
        #             ax.set_xticklabels([])
        # else:
        #     for ax in axs[:-1].flatten():
        #         ax.set_xticklabels([])


        # if angles:
        #     pi_axs = []
        #     pi_axs_vert = []
        #     if Tc:
        #         two_pi_axs = axs[-1, 4:].flatten()
        #         two_pi_axs_vert = axs[4:, 0].flatten()
        #     else:
        #         two_pi_axs = axs[-1, 3:].flatten()
        #         two_pi_axs_vert = axs[3:, 0].flatten()

        #     if wrap_M0 or wrap_w:
        #         if wrap_M0 and wrap_w:
        #             i1, i2 = slice(3, 5), slice(5, 7)
        #         elif wrap_M0:
        #             i1, i2 = slice(3, 4), slice(4, 7)
        #         elif wrap_w:
        #             i1, i2 = slice(4, 5), slice(6, 6)
        #         pi_axs = axs[-1, i1].flatten()
        #         pi_axs_vert = axs[i1, 0].flatten()
        #         two_pi_axs = axs[-1, i2].flatten()
        #         two_pi_axs_vert = axs[i2, 0].flatten()

        #     two_pi_labels = ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
        #     for ax in two_pi_axs:
        #         ax.set(xticks=np.linspace(0, 2*np.pi, 5), xticklabels=two_pi_labels)
        #     for ax in two_pi_axs_vert:
        #         ax.set(yticks=np.linspace(0, 2*np.pi, 3), yticklabels=two_pi_labels[::2])
        #     pi_labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$']
        #     for ax in pi_axs:
        #         ax.set(xticks=np.linspace(-np.pi, np.pi, 5), xticklabels=pi_labels)
        #     for ax in pi_axs_vert:
        #         ax.set(yticks=np.linspace(-np.pi, np.pi, 3), yticklabels=pi_labels[::2])

        if force_ecc0_limit:
            if upper:
                ecc_axs = np.triu(axs)[:, 2].flatten()
                for ax in ecc_axs:
                    if ax:
                        ax.set_xlim(0, None)
            else:
                ecc_axs = np.tril(axs)[:, 2].flatten()
                for ax in ecc_axs:
                    if ax:
                        ax.set_xlim(0, None)

        # share axis limits
        if upper:
            share_x_axs = np.triu(axs)
            for i, ax_col in enumerate(share_x_axs.T):
                for ax in ax_col[:i]:
                    ax.set_xlim(ax_col[i].get_xlim())
            for i, ax_row in enumerate(share_x_axs):
                for ax in ax_row[i+1:]:
                    ax.set_ylim(ax_row[i].get_xlim())
        else:
            share_x_axs = np.tril(axs)
            for i, ax_col in enumerate(share_x_axs.T):
                for ax in ax_col[i:]:
                    ax.set_xlim(ax_col[i].get_xlim())
            for i, ax_row in enumerate(share_x_axs):
                for ax in ax_row[:i]:
                    ax.set_ylim(ax_row[i].get_xlim())


        # number of ticks
        if upper:
            tick_axs = np.triu(axs, 1)
            # y axis
            for ax in tick_axs.flatten():
                if ax:
                    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins='auto', min_n_ticks=3))
            # x axis
            for ax in np.diag(axs):
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbins='auto', min_n_ticks=3))
                ax.xaxis.set_tick_params(rotation=45)
        else:
            tick_axs = np.tril(axs, -1)
            # y axis
            for ax in  tick_axs.flatten():
                if ax:
                    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins='auto', min_n_ticks=3))
            # x axis
            for ax in np.r_[tick_axs.flatten(), axs[-1, -1]]:
                if ax:
                    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins='auto', min_n_ticks=3))
                    ax.xaxis.set_tick_params(rotation=45)


        # if upper:
        #     row_axs = axs[0, :]
        #     # for ax in row_axs:
        #     #     # ax.xaxis.tick_top()
        #     #     # ax.xaxis.set_label_position('top') 
        # else:
        #     row_axs = axs[-1, :]

        # for i, (label, unit, ax) in enumerate(zip(labels, units, row_axs)):
        #     if ranges[i] is not None:
        #         ax.set_xlim(ranges[i])
        #     elif i == 2 and force_ecc0_limit:
        #         ax.set_xlim(0, None)
        #         # ax.xaxis.set_major_locator(plt.MaxNLocator(3))


        #     if not upper and not _axs_provided:
        #         # same column
        #         for _ax in axs[:-1, i].flatten():
        #             _ax.set_xlim(ax.get_xlim())

        #     if not _axs_provided:
        #         # corresponding row
        #         for _ax in axs[i, :i].flatten():
        #             _ax.set_ylim(ax.get_xlim())

        if Tc:
            axs[-2, 0].get_yaxis().get_major_formatter().set_useOffset(False)
            axs[-1, -2].get_xaxis().get_major_formatter().set_useOffset(False)

    if np.any(priors):
        leg = axs[0, 0].get_legend_handles_labels()
        axs[0, -1].legend(*leg)

    # if upper and not _axs_provided:
    #     fig.subplots_adjust(bottom=0.1, top=0.95, left=0.01, right=0.95)

    # fig.tight_layout()
    return fig, axs

# TODO: this function is clearly too complicated
def corner_planet_parameters(res, fig=None, Np=None, true_values=None, period_ranges=None,
                             include_known_object=False, include_transiting_planet=False,
                             KO_Np=None, TR_Np=None, show_prior=False, mass_units='mearth',
                             degrees=False, wrap_M0=False, wrap_w=False, replace_angles_with_mass=False,
                             star_mass=1.0, a_factor=1.0, show_stellar_mass=True,
                             force_ecc0_limit=False, full_output=False, 
                             true_value_label='', true_value_kwargs={}, **kwargs):
    """ Corner plots of the posterior samples for the planet parameters """

    if res.model is MODELS.GAIAmodel:
        labels = ['$P$',  r'$\phi$', 'e', 'a',  r'$\omega$', r'$\cos i$', 'W']
        units  = ['days', 'rad',     '',  'AU', 'rad',       '',          'rad']
    elif res.model is MODELS.RVHGPMmodel:
        if replace_angles_with_mass:
            labels = ['$P$',  '$K$', '$e$', '$M_p$',          '$a$']
            if mass_units == 'mjup':
                units = ['days', 'm/s', '', r'M$_{\rm Jup}$', 'AU']
            elif mass_units == 'mearth':
                units = ['days', 'm/s', '', r'M$_\oplus$', 'AU']
        else:
            labels = ['$P$',  '$K$', '$e$', '$M_0$', r'$\omega$', '$i$', r'$\Omega$']
            units  = ['days', 'm/s', '',    'rad',   'rad',       'rad', 'rad']
    else:
        if replace_angles_with_mass:
            labels = [r'$P$', r'$K$', '$e$', r'$M_p \sin i$', '$a$']
            if mass_units == 'mjup':
                units = ['days', 'm/s', '', r'M$_{\rm Jup}$', 'AU']
            elif mass_units == 'mearth':
                units = ['days', 'm/s', '', r'M$_\oplus$', 'AU']
            if a_factor != 1.0:
                labels[-1] = f'${a_factor:.0e}'.replace('e+0', '0^') + r'\times a$'
        else:
            labels = [r'$P$', r'$K$', '$e$', '$M_0$', r'$\omega$']
            if degrees:
                units = ['days', 'm/s', '', 'deg', 'deg']
            else:
                units = ['days', 'm/s', '', 'rad', 'rad']

    nk = res.max_components

    if Np is None:
        Np = list(range(1, nk + 1))
    else:
        if isinstance(Np, int):
            Np = [Np]

    if fig is None:
        previous_fig = [None] * nk
    else:
        assert len(fig) == len(Np), f'{len(fig)=} should be equal to {len(Np)=}'
        previous_fig = fig

    if true_values is None:
        n = nk
        n += res.nKO if include_known_object else 0
        n += res.nTR if include_transiting_planet else 0
        true_values = [None] * n
    else:
        if len(true_values) != res.n_dimensions:
            assert len(true_values) == len(Np), \
                f'len(true_values) should be {len(Np)}, got {len(true_values)}'
        else:
            tr = [None] * nk
            for _Np in Np:
                tr[_Np - 1] = true_values
            true_values = tr
            for i, tr in enumerate(true_values):
                if tr:
                    assert len(tr) == res.n_dimensions, \
                        f'len(true_values[i]) should be {res.n_dimensions}, got {i}: {len(tr)}'

    if period_ranges is None:
        period_ranges = [None] * nk
    else:
        assert len(period_ranges) == len(Np), \
            f'{len(period_ranges)=} should be equal to {len(Np)=}'
        pr = [None] * nk
        for _Np in Np:
            pr[_Np - 1] = period_ranges[0]
        period_ranges = pr

    figs, axss = [], []

    kwargs = {
        **kwargs,
        **{
            'wrap_M0': wrap_M0,
            'wrap_w': wrap_w,
            'angles': not replace_angles_with_mass,
            'degrees': degrees,
            'force_ecc0_limit': force_ecc0_limit
    }}

    for i in range(nk):
        if i + 1 not in Np:
            continue
        if replace_angles_with_mass:
            if period_ranges[i] is not None:
                mask = (period_ranges[i][0] < res.posteriors.P[:, i]) & (res.posteriors.P[:, i] < period_ranges[i][1])
            else:
                mask = np.full(res.ESS, True)

            m, a = get_planet_mass_and_semimajor_axis(
                res.posteriors.P[mask, i], 
                res.posteriors.K[mask, i], 
                res.posteriors.e[mask, i],
                star_mass=star_mass, full_output=True
            )
            m = m[2]
            if mass_units == 'mearth':
                m *= mjup2mearth
            a = a[2] * a_factor

            if res.model is MODELS.RVHGPMmodel:
                m /= np.sin(res.posteriors.i[mask, i])

            samples = np.c_[
                res.posteriors.P[mask, i].copy(),
                res.posteriors.K[mask, i].copy(),
                res.posteriors.e[mask, i].copy(),
                m,
                a,
            ]
        else:
            samples = np.c_[
                res.posteriors.P[:, i].copy(),
                res.posteriors.K[:, i].copy(),
                res.posteriors.e[:, i].copy(),
                res.posteriors.φ[:, i].copy(),
                res.posteriors.w[:, i].copy(),
            ]

            if res.model is MODELS.RVHGPMmodel:
                samples = np.c_[
                    samples,
                    res.posteriors.i[:, i].copy(),
                    res.posteriors.Ω[:, i].copy()
                ]


        if wrap_M0 and not replace_angles_with_mass:
            samples[:, 3] = np.arctan2(np.sin(samples[:, 3]), np.cos(samples[:, 3]))
        if wrap_w and not replace_angles_with_mass:
            samples[:, 4] = np.arctan2(np.sin(samples[:, 4]), np.cos(samples[:, 4]))

        if degrees and not wrap_M0 and not replace_angles_with_mass:
            samples[:, 3] = np.rad2deg(samples[:, 3])
        if degrees and not wrap_w and not replace_angles_with_mass:
            samples[:, 4] = np.rad2deg(samples[:, 4])

        ranges = res.n_dimensions * [None]

        if replace_angles_with_mass:
            ranges = 5 * [None]

        if period_ranges is not None:
            ranges[0] = period_ranges[i]

        priors = None
        if show_prior:
            priors = [res.priors[k] for k in ['Pprior', 'Kprior', 'eprior', 'phiprior', 'wprior']]
            if res.model is MODELS.RVHGPMmodel:
                priors += [res.priors[k] for k in ['iprior', 'Omegaprior']]
            priors = [distribution_rvs(p, res.ESS) if p else None for p in priors]

            if replace_angles_with_mass:
                (*_, m), (*_, a) = get_planet_mass_and_semimajor_axis(
                    priors[0], priors[1], priors[2], 
                    star_mass=star_mass, full_output=True
                )
                if mass_units == 'mearth':
                    m *= mjup2mearth
                a *= a_factor
                priors[3] = m
                priors[4] = a

        fig, axs = corner_orbital(samples, labels=labels, units=units, 
                                  priors=priors, truths=true_values[i], ranges=ranges,
                                  true_value_label=true_value_label, true_value_kwargs=true_value_kwargs,
                                  fig=previous_fig[i], **kwargs)

        figs.append(fig)
        axss.append(axs)
    
    if res.KO and include_known_object:
        if KO_Np is None:
            KO_Np = list(range(1, res.nKO + 1))
        else:
            if isinstance(KO_Np, int):
                KO_Np = [KO_Np]

        for i in range(res.nKO):
            if i + 1 not in KO_Np:
                continue
            samples = res.KOpars[:, i::res.nKO].copy()
            # swtich M0 and ecc, just for convenience
            samples[:, [3, 2]] = samples[:, [2, 3]]

            if replace_angles_with_mass:
                (*_, m), (*_, a) = get_planet_mass_and_semimajor_axis(
                    samples[:, 0], samples[:, 1], samples[:, 2],
                    star_mass=star_mass, full_output=True
                )
                if mass_units == 'mearth':
                    m *= mjup2mearth
                a *= a_factor

                samples = np.c_[samples[:, :3], m, a]

            if wrap_M0 and not replace_angles_with_mass:
                samples[:, 3] = np.arctan2(np.sin(samples[:, 3]), np.cos(samples[:, 3]))
            if wrap_w and not replace_angles_with_mass:
                samples[:, 4] = np.arctan2(np.sin(samples[:, 4]), np.cos(samples[:, 4]))

            priors = None
            if show_prior:
                priors = [p for k, p in res.priors.items()
                          if 'KO_' in k and f'_{i}' in k]
                priors = [distribution_rvs(p, res.ESS) if p else None for p in priors]
                if replace_angles_with_mass:
                    (*_, m), (*_, a) = get_planet_mass_and_semimajor_axis(
                        priors[0], priors[1], priors[2], 
                        star_mass=star_mass, full_output=True
                    )
                    if mass_units == 'mearth':
                        m *= mjup2mearth
                    a *= a_factor
                    priors[3] = m
                    priors[4] = a

            fig, axs = corner_orbital(samples, labels=labels, units=units,
                                      priors=priors, truths=true_values[i], **kwargs)
            figs.append(fig)
            axss.append(axs)

    if res.TR and include_transiting_planet:
        if TR_Np is None:
            TR_Np = list(range(1, res.nTR + 1))
        else:
            if isinstance(TR_Np, int):
                TR_Np = [TR_Np]

        for i in range(res.nTR):
            if i + 1 not in TR_Np:
                continue
            samples = res.TRpars[:, i::res.nTR].copy()
            # swtich M0 and ecc, just for convenience
            samples[:, [3, 2]] = samples[:, [2, 3]]

            if replace_angles_with_mass:
                pass
            else:
                labels[3] = '$T_c$'
                units[3] = 'days'

            if wrap_M0 and not replace_angles_with_mass:
                samples[:, 3] = np.arctan2(np.sin(samples[:, 3]), np.cos(samples[:, 3]))
            if wrap_w and not replace_angles_with_mass:
                samples[:, 4] = np.arctan2(np.sin(samples[:, 4]), np.cos(samples[:, 4]))

            if replace_angles_with_mass:
                (*_, m), (*_, a) = get_planet_mass_and_semimajor_axis(
                    samples[:, 0], samples[:, 1], samples[:, 2],
                    star_mass=star_mass, full_output=True
                )
                if mass_units == 'mearth':
                    m *= mjup2mearth
                a *= a_factor

                samples = np.c_[samples[:, :3], m, a]

            fig, axs = corner_orbital(samples, labels=labels, units=units, Tc=True,
                                      priors=None, truths=true_values[i], **kwargs)
            figs.append(fig)
            axss.append(axs)

    if 'axs' in kwargs:
        show_stellar_mass = False

    if show_stellar_mass and replace_angles_with_mass:
        for axs, fig in zip(axss, figs):
            text = 'stellar mass: '
            if isinstance(star_mass, float):
                text += rf'${star_mass} \, M_{{\odot}}$'
            elif isinstance(star_mass, tuple):
                text += rf'${star_mass[0]} \pm {star_mass[1]} \, M_{{\odot}}$'
            leg = axs[0, -1].get_legend_handles_labels()
            if kwargs.get('upper', False):
                fig.legend(*leg, title=text, frameon=False, loc='lower left')
            else:
                fig.legend(*leg, title=text, frameon=False)

    if full_output:
        return figs, axss, samples
    else:
        return figs, axss


def hist_vsys(res, ax=None, show_offsets=True, show_other=False, 
              specific=None, show_prior=False, **kwargs):
    """
    Plot the histogram of the posterior for the systemic velocity and for the
    between-instrument offsets. (if the model has multiple instruments).

    Args:
        res (kima.KimaResults):
            The `KimaResults` instance
        show_offsets (bool, optional):
            Whether to plot the histograms for the between-instrument offsets
        show_other (bool, optional):
            Whether to plot the histogram for the constant offsets corresponding
            to the activity indicators in the corresponding models
        specific (tuple, optional):
            If not None, it should be a tuple with two instrument names
            (matching `res.instrument`). In that case, this function works out
            (and plots) the RV offset between those two instruments.
        show_prior (bool, optional):
            Whether to plot the histogram of the prior distribution
        **kwargs:
            Keyword arguments passed to `matplotlib.pyplot.hist`
    """
    figures = []
    units = ' (arbitrary)' if res.arbitrary_units else ' (m/s)'

    if ax is None:
        if res.model is MODELS.BINARIESmodel and res.double_lined:
            fig, axs = plt.subplots(1, 2, constrained_layout=True,figsize=[10.4, 4.8])
            ax,ax2 = axs[0],axs[1]
        else:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
    else:
        fig = ax.figure
        if res.model is MODELS.BINARIESmodel and res.double_lined:
            ax2 = ax
    figures.append(fig)

    bins = kwargs.get('bins', 'doane')
    density = kwargs.get('density', True)
    hist_kw = dict(bins=bins, density=density)
    hist_prior_kw = dict(**hist_kw, alpha=0.15, color='k', zorder=-1)

    vsys = res.posteriors.vsys.copy()
    estimate = percentile68_ranges_latex(vsys) + units
    ax.hist(vsys, label=estimate, **hist_kw)

    title = 'Posterior distribution for $v_{\\rm sys}$'
    ylabel = 'posterior' if density else 'posterior samples'
    ax.set(xlabel='vsys' + units, ylabel=ylabel, title=title)

    if res.model is MODELS.BINARIESmodel and res.double_lined:
        vsys_sec = res.posteriors.vsys_sec.copy()
        estimate = percentile68_ranges_latex(vsys_sec) + units
        ax2.hist(vsys_sec, label=estimate, **hist_kw)

        title2 = 'Posterior distribution for $v_{\\rm sys,sec}$'
        ax2.set(xlabel='vsys_sec' + units, title=title2)

    if show_prior:
        try:
            prior = res.priors['Cprior']
        except KeyError:
            prior = res.priors['Vprior']
        ax.hist(distribution_rvs(prior, size=res.ESS), label='prior', **hist_prior_kw)
        if res.model is MODELS.BINARIESmodel and res.double_lined:
            ax2.hist(distribution_rvs(prior, size=res.ESS), label='prior', **hist_prior_kw)

    ax.legend()
    if res.model is MODELS.BINARIESmodel and res.double_lined:
        ax2.legend()

    if res.save_plots:
        filename = 'kima-showresults-fig7.2.png'
        print('saving in', filename)
        fig.savefig(filename)

    if show_offsets and res.multi:
        n_inst_offsets = res.inst_offsets.shape[1]
        squeeze=True
        if res.model is MODELS.RVFWHMmodel:
            nrows = 2 
            squeeze = False
        elif res.model is MODELS.BINARIESmodel and res.double_lined:
            nrows = 2 
            squeeze = False
        else:
            nrows = 1
        fig, axs = plt.subplots(nrows, n_inst_offsets // nrows, sharey=True,
                                figsize=(2 + 3 * n_inst_offsets//nrows, 2 + 3 * nrows), squeeze=squeeze,
                                constrained_layout=True)
        figures.append(fig)
        if n_inst_offsets == 1:
            axs = [axs,]
        prior = res.priors['offsets_prior']

        if res.model is MODELS.RVFWHMmodel:
            k = 0
            wrt = res.instruments[-1]
            for j in range(2):
                for i in range(n_inst_offsets // 2):
                    this = res.instruments[i]
                    a = res.inst_offsets[:, k]
                    axs[j, i].hist(a, **hist_kw)
                    label = 'offset\n%s rel. to %s' % (this, wrt)
                    estimate = percentile68_ranges_latex(a) + units
                    axs[j, i].set(xlabel=label, title=estimate)
                    k += 1

                    if show_prior and j == 0:
                        axs[j, i].hist(distribution_rvs(prior, size=res.ESS), **hist_prior_kw)
                        axs[j, i].legend(['posterior', 'prior'])
        elif res.model is MODELS.BINARIESmodel and res.double_lined:
            k = 0
            wrt = res.instruments[-1]
            for j in range(2):
                extra = '_pri'
                if j==1:
                    extra = '_sec'
                for i in range(n_inst_offsets // 2):
                    this = res.instruments[i] + extra
                    a = res.inst_offsets[:, k]
                    axs[j, i].hist(a, **hist_kw)
                    label = 'offset\n%s rel. to %s' % (this, wrt+extra)
                    estimate = percentile68_ranges_latex(a) + units
                    axs[j, i].set(xlabel=label, title=estimate)
                    k += 1

                    if show_prior and j == 0:
                        axs[j, i].hist(distribution_rvs(prior, size=res.ESS), **hist_prior_kw)
                        axs[j, i].legend(['posterior', 'prior'])
        else:
            for i in range(n_inst_offsets):
                # wrt = get_instrument_name(res.data_file[-1])
                # this = get_instrument_name(res.data_file[i])
                wrt = res.instruments[-1]
                this = res.instruments[i]
                label = 'offset\n%s rel. to %s' % (this, wrt)
                a = res.inst_offsets[:, i]
                estimate = percentile68_ranges_latex(a) + units
                axs[i].hist(a, **hist_kw)

                if show_prior:
                    try:
                        prior = res.priors[f'individual_offset_prior[{i}]']
                    except KeyError:
                        pass
                    axs[i].hist(distribution_rvs(prior, size=res.ESS), **hist_prior_kw)
                    axs[i].legend(['posterior', 'prior'])

                axs[i].set(xlabel=label, title=estimate, ylabel=ylabel)

        title = 'Posterior distribution(s) for instrument offset(s)'
        fig.suptitle(title)

        if res.save_plots:
            filename = 'kima-showresults-fig7.2.1.png'
            print('saving in', filename)
            fig.savefig(filename)

        if specific is not None:
            assert isinstance(specific, tuple), '`specific` should be a tuple'
            assert len(specific) == 2, '`specific` should have size 2'

            first = specific[0]
            found_first = any([first in f for f in res.data_file]) or first in res.instruments
            assert found_first, 'first element is not in res.data_file or res.instruments'

            second = specific[1]
            found_second = any([second in f for f in res.data_file]) or second in res.instruments
            assert found_second, 'second element is not in res.data_file or res.instruments'

            # all RV offsets are with respect to the last data file
            # so if the second element is that last one, we don't have to do much
            if second in res.instruments[-1] or second in res.data_file[-1]:
                # find the first
                if first in res.instruments:
                    i = res.instruments.index(first)
                else:
                    i = [_i for _i, f in enumerate(res.instruments) if first in f][0]
                wrt, this = res.instruments[-1], res.instruments[i]
                label = 'offset\n%s rel. to %s' % (this, wrt)
                offset = res.inst_offsets[:, i]
                estimate = percentile68_ranges_latex(offset) + units
                fig, ax = plt.subplots(1, 1, constrained_layout=True)
                ax.hist(offset)
                ax.set(xlabel=label, title=estimate, ylabel='posterior samples')
            # if, instead, the first element is the last instrument,
            # we just need a sign change
            elif first in res.data_file[-1] or first in res.instruments[-1]:
                # find the second
                if second in res.instruments:
                    i = res.instruments.index(second)
                else:
                    i = [_i for _i, f in enumerate(res.instruments) if second in f][0]
                wrt, this = res.instruments[i], res.instruments[-1]
                label = 'offset\n%s rel. to %s' % (this, wrt)
                offset = res.inst_offsets[:, i] * -1
                estimate = percentile68_ranges_latex(offset) + units
                fig, ax = plt.subplots(1, 1, constrained_layout=True)
                ax.hist(offset)
                ax.set(xlabel=label, title=estimate, ylabel='posterior samples')
            # otherwise, we have to do a little more work
            else:
                if second in res.instruments:
                    i = res.instruments.index(second)
                else:
                    i = [_i for _i, f in enumerate(res.instruments) if second in f][0]
                wrt = res.instruments[i]

                if first in res.instruments:
                    j = res.instruments.index(first)
                else:
                    j = [_i for _i, f in enumerate(res.instruments) if first in f][0]
                this = res.instruments[j]

                label = 'offset\n%s rel. to %s' % (this, wrt)
                of1 = res.inst_offsets[:, j]
                of2 = res.inst_offsets[:, i]
                estimate = percentile68_ranges_latex(of1 - of2) + units
                fig, ax = plt.subplots(1, 1, constrained_layout=True)
                ax.hist(of1 - of2)
                ax.set(xlabel=label, title=estimate, ylabel='posterior samples')

    else:
        figures.append(None)

    if show_other:
        if res.model is MODELS.RVFWHMmodel:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            val = res.posterior_sample[:, res.indices['cfwhm']]
            estimate = percentile68_ranges_latex(val)
            ax.hist(val, label=estimate, **hist_kw)

            if show_prior:
                prior = res.priors['Cfwhm_prior']
                ax.hist(distribution_rvs(prior, size=res.ESS), label='prior',
                        **hist_prior_kw)

            ax.legend()
            ax.set(xlabel='C fwhm', ylabel=ylabel)

        if res.model is MODELS.SPLEAFmodel and res.nseries > 1:
            fig, axs = plt.subplots(1, res.nseries - 1, constrained_layout=True)
            axs = np.atleast_1d(axs)

            zp = res.posterior_sample[:, res.indices['zero_points']]
            for i, label in zip(range(res.nseries - 1), res._extra_data_names[::2]):
                estimate = percentile68_ranges_latex(zp[:, i])
                axs[i].hist(zp[:, i], label=estimate, **hist_kw)
                axs[i].set(xlabel=f'zero point {label}', ylabel=ylabel)

                if show_prior:
                    prior = res.priors[f'zero_points_prior_{i+1}']
                    axs[i].hist(distribution_rvs(prior, size=res.ESS), label='prior',
                                **hist_prior_kw)

                axs[i].legend()

    if res.return_figs:
        return figures


def hist_jitter(res, show_prior=False, show_stats=False, show_title=True,
                show_stellar_jitter=True, ax=None, **kwargs):
    """
    Plot the histogram of the posterior for the additional white noise
    """
    # if res.arbitrary_units:
    #     units = ' (arbitrary)'
    # else:
    #     units = ' (m/s)'  # if res.units == 'ms' else ' (km/s)'

    RVFWHM = res.model is MODELS.RVFWHMmodel
    RVFWHMRHK = res.model is MODELS.RVFWHMRHKmodel
    SPLEAF = res.model is MODELS.SPLEAFmodel
    _models_with_stellar_jitter = (
        MODELS.RVmodel, MODELS.RVHGPMmodel
    )
    SB2 = res.model is MODELS.BINARIESmodel and res.double_lined
    GAIA = res.model is MODELS.GAIAmodel
    RVGAIA = res.model is MODELS.RVGAIAmodel

    all_in_one_plot = False

    n_jitters = res.n_jitters

    if not show_stellar_jitter:
        if res.model in _models_with_stellar_jitter and res.multi:
            n_jitters -= 1

    if ax is None:
        kw = dict(constrained_layout=True, sharey=False)
        if RVFWHM:
            fig, axs = plt.subplots(2, n_jitters // 2, 
                                    figsize=(min(10, 5 + n_jitters * 2), 4), **kw)
        elif RVFWHMRHK:
            fig, axs = plt.subplots(3, n_jitters // 3, 
                                    figsize=(min(10, 5 + n_jitters * 2), 6), **kw)
        elif SPLEAF:
            fig, axs = plt.subplots(res.n_instruments, res.nseries, 
                                    # figsize=(min(10, 5 + n_jitters * 2), 6), 
                                    **kw
                                    )
        elif SB2:
            fig, axs = plt.subplots(2, n_jitters // 2, 
                                    figsize=(min(10, 5 + n_jitters), 6), **kw)
        else:
            # nrows = {1:1, 2:1, 3:1, 4:1, 5:2, 6:2, 7:2, 8:2}[n_jitters]
            nrows = int(np.floor((n_jitters - 1) / 4) + 1)
            fig, axs = plt.subplots(nrows, int(np.ceil(n_jitters / nrows)),
                                    figsize=(min(10, 5 + n_jitters * 2), 4), **kw)
        axs = np.atleast_1d(axs)
    else:
        axs = np.atleast_1d(ax)
        if axs.size == 1 and n_jitters > 1:
            axs = np.tile(axs, n_jitters)
            all_in_one_plot = True
        
        # assert len(axs) == n_jitters, \
        #     f'length of ax should be same as number of jitters ({n_jitters})'
        fig = axs[0].figure

    if show_title:
        if len(axs) > 1:
            fig.suptitle('Posterior distribution for extra white noise')
        else:
            axs[0].set_title('Posterior distribution for extra white noise')

    if isinstance(axs, np.ndarray) and res.multi:
        if RVFWHM or RVFWHMRHK:
            for ax in axs[1]:
                ax.sharex(axs[0, 0])

    kwargs.setdefault('density', True)
    kwargs.setdefault('bins', 'doane')
    axs = np.ravel(axs)

    # trick the loop with a None axis
    if not show_stellar_jitter:
        if res.model in _models_with_stellar_jitter and res.multi:
            axs = np.r_[None, axs]

    for i, ax in enumerate(axs):

        units = ' m/s'
        if GAIA:
            units = ' mas'
        if RVGAIA and i==0:
            units = ' mas'

        if ax is None:
            continue

        if i >= res.n_jitters:
            ax.axis('off')
            continue

        j = i // res.n_instruments
        estimate = percentile68_ranges_latex(res.jitter[:, i]) + units

        # remove "m/s" from jitter slope
        if res.jitter_propto_indicator and i == res.n_jitters - 1:
            estimate = estimate[:-3]

        ax.hist(res.jitter[:, i], label=estimate, **kwargs)

        leg = ax.legend()
        leg._legend_box.sep = 0

        if show_prior:
            if RVFWHM:
                prior_name = 'Jprior' if j==0 else f'J{j+1}prior'
                prior = distribution_rvs(res.priors[prior_name], size=res.ESS)
            elif RVFWHMRHK:
                prior_name = 'Jprior' if j==0 else f'J{j+1}prior'
                prior = distribution_rvs(res.priors[prior_name], size=res.ESS)
            else:
                if res.model in _models_with_stellar_jitter and res.multi and i==0:
                    prior = distribution_rvs(res.priors['stellar_jitter_prior'], size=res.ESS)
                else:
                    prior = distribution_rvs(res.priors['Jprior'], size=res.ESS)

            ax.hist(prior, density=True, color='k', alpha=0.15, zorder=-1)

        if show_stats:
            from matplotlib import transforms
            transform = transforms.blended_transform_factory(ax.transData,
                                                             ax.transAxes)
            kw = dict(fontsize=8, transform=transform)

            if RVFWHM and i >= res.n_instruments:
                j = i - res.n_instruments
                m = res.data.e2[res.data.obs == j + 1].mean()
                ax.axvline(m, 0, 0.2, color='r')
                ax.text(m, 0.1, r'$\overline{\sigma}_{FWHM}$', color='r', **kw)
                s = res.data.y2[res.data.obs == j + 1].std()
                ax.axvline(s, 0, 0.2, color='g')
                ax.text(s, 0.2, r'SD FWHM', color='g', **kw)

            else:
                m = res.data.e[res.data.obs == i + 1].mean()
                ax.axvline(m, 0, 0.2, color='r')
                ax.text(m, 0.1, r'$\overline{\sigma}_{RV}$', color='r', **kw)
                s = res.data.y[res.data.obs == i + 1].std()
                ax.axvline(s, 0, 0.2, color='g')
                ax.text(s, 0.2, r'SD RV', color='g', **kw)

    if not show_stellar_jitter:
        if res.model in _models_with_stellar_jitter and res.multi:
            axs = axs[1:]

    for ax in axs:
        ax.set(yticks=[], ylabel='posterior')

    instruments = res.instruments
    if isinstance(instruments, str):
        instruments = [instruments]
    insts = [get_instrument_name(i) for i in instruments]

    if RVFWHM:
        labels = [f'RV jitter {i} [m/s]' for i in insts]
        labels += [f'FWHM jitter {i} [m/s]' for i in insts]
    elif RVFWHMRHK:
        labels = [f'RV jitter {i} [m/s]' for i in insts]
        labels += [f'FWHM jitter {i} [m/s]' for i in insts]
        labels += [f'RHK jitter {i}' for i in insts]
    elif SPLEAF:
        labels = [f'RV jitter {i} [m/s]' for i in insts]
        labels += [f'{s} jitter {i} [m/s]' for i in insts 
                   for s in res._extra_data_names[::2]]
    elif SB2:
        labels = [f'RV jitter {i}_pri [m/s]' for i in insts]
        labels += [f'RV jitter {i}_sec [m/s]' for i in insts]
    elif GAIA:
        labels = ['Astrometric jitter [mas]']
    elif RVGAIA:
        labels = ['Astrometric jitter [mas]']+[f'jitter {i} [m/s]' for i in insts]
    else:
        labels = [f'jitter {i} [m/s]' for i in insts]
        if show_stellar_jitter:
            if res.model is MODELS.RVmodel:
                if res.multi:
                    labels.insert(0, 'stellar jitter')
                if res.jitter_propto_indicator:
                    labels.append('jitter slope')

    if all_in_one_plot:
        ax.set_xlabel('jitter [m/s]')
    else:
        for ax, label in zip(axs, labels):
            ax.set_xlabel(label, fontsize=10)

    if res.save_plots:
        filename = 'kima-showresults-fig7.3.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig


def hist_correlations(res, show_prior=False):
    """ Plot the histogram of the posterior for the activity correlations """
    if not res.indicator_correlations:
        msg = 'Model has no activity correlations! hist_correlations() doing nothing...'
        print(msg)
        return

    # units = ' (m/s)' if res.units=='ms' else ' (km/s)'
    # estimate = percentile68_ranges_latex(res.offset) + units

    n = len(res.activity_indicators)
    fig, axs = plt.subplots(n, 1, constrained_layout=True)

    for i, ax in enumerate(np.ravel(axs)):
        estimate = percentile68_ranges_latex(res.betas[:, i])
        estimate = '$c_{%s}$ = %s' % (res.activity_indicators[i], estimate)
        ax.hist(res.betas[:, i], label=estimate)
        ax.set(ylabel='posterior samples',
               xlabel='$c_{%s}$' % res.activity_indicators[i])

        if show_prior:
            prior = distribution_rvs(res.priors['beta_prior'], res.ESS)
            ax.hist(prior, alpha=0.15, color='k', zorder=-1, label='prior')

        leg = ax.legend(frameon=False)
        #leg.legendHandles[0].set_visible(False)

    title = 'Posterior distribution for activity correlations'
    fig.suptitle(title)

    if res.save_plots:
        filename = 'kima-showresults-fig7.4.png'
        print('saving in', filename)
        fig.savefig(filename)


def hist_trend(res, per_year=True, ax=None,
               show_prior=False, show_title=True):
    """
    Plot the histogram of the posterior for the coefficients of the trend
    """
    if not res.trend:
        print('Model has no trend! hist_trend() doing nothing...')
        return

    deg = res.trend_degree
    names = ['slope', 'quadr', 'cubic']
    if res.arbitrary_units:
        units = ['/yr', '/yr²', '/yr³']
    else:
        units = ['m/s/yr', 'm/s/yr²', 'm/s/yr³']

    trend = res.trendpars.copy()

    if per_year:  # transfrom from /day to /yr
        trend *= 365.25**np.arange(1, res.trend_degree + 1)

    if ax is not None:
        ax = np.atleast_1d(ax)
        assert len(ax) == deg, f'wrong length, need {deg} axes'
        fig = ax[0].figure
    else:
        fig, ax = plt.subplots(deg, 1, constrained_layout=True, squeeze=True)
        ax = np.atleast_1d(ax)

    if show_title:
        fig.suptitle('Posterior distribution for trend coefficients')

    for i in range(deg):
        estimate = percentile68_ranges_latex(trend[:, i]) + ' ' + units[i]

        ax[i].hist(trend[:, i].ravel(), label='posterior')
        if show_prior:
            prior = res.priors[names[i] + '_prior']
            f = 365.25**(i + 1) if per_year else 1.0
            ax[i].hist(distribution_rvs(prior, res.ESS) * f,
                       alpha=0.15, color='k', zorder=-1, label='prior')

        ax[i].set(xlabel=f'{names[i]} ({units[i]})')

        if show_prior:
            ax[i].legend(title=estimate)
        else:
            leg = ax[i].legend([], [], title=estimate)
            leg._legend_box.sep = 0


    # fig.set_constrained_layout_pads(w_pad=0.3)
    # fig.text(0.01, 0.5, 'posterior samples', rotation=90, va='center')

    if res.save_plots:
        filename = 'kima-showresults-fig7.5.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig

def hist_MA(res):
    """ Plot the histogram of the posterior for the MA parameters """
    if not res.MAmodel:
        print('Model has no MA! hist_MA() doing nothing...')
        return

    # units = ' (m/s/day)' # if res.units=='ms' else ' (km/s)'
    # estimate = percentile68_ranges_latex(res.trendpars) + units

    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
    ax1.hist(res.MA[:, 0])
    ax2.hist(res.MA[:, 1])
    title = 'Posterior distribution for MA parameters'
    fig.suptitle(title)
    ax1.set(xlabel=r'$\sigma$ MA [m/s]', ylabel='posterior samples')
    ax2.set(xlabel=r'$\tau$ MA [days]', ylabel='posterior samples')

    if res.save_plots:
        filename = 'kima-showresults-fig7.6.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig


def hist_nu(res, show_prior=False, **kwargs):
    """
    Plot the histogram of the posterior for the Student-t degrees of freedom
    """
    if not res.studentt:
        print('Model has Gaussian likelihood! hist_nu() doing nothing...')
        return


    if res.model is MODELS.RVGAIAmodel:
        estimate_Gaia = percentile68_ranges_latex(res.nu_GAIA)
        estimate_RV = percentile68_ranges_latex(res.nu_RV)

        fig, ax = plt.subplots(1, 2, figsize=[12,6])
        ax[0].hist(res.nu_GAIA, **kwargs)
        ax[1].hist(res.nu_RV, **kwargs)
        title1 = 'Posterior distribution for degrees of freedom $\\nu$ for Gaia data \n '\
                '%s' % estimate_Gaia
        title2 = 'Posterior distribution for degrees of freedom $\\nu$ for RV data \n '\
                '%s' % estimate_RV
        if kwargs.get('density', False):
            ylabel = 'posterior'
        else:
            ylabel = 'posterior samples'

        ax[0].set(xlabel='$\\nu$', ylabel=ylabel, title=title1)
        ax[1].set(xlabel='$\\nu$', ylabel=ylabel, title=title2)

        if show_prior:
            try:
                # low, upp = res.priors['Jprior'].interval(1)
                d = kwargs.get('density', False)
                ax[0].hist(res.priors['J_GAIA_prior'].rvs(res.ESS), density=d, alpha=0.15,
                        color='k', zorder=-1)
                ax[0].legend(['posterior', 'prior'])
                ax[1].hist(res.priors['J_RV_prior'].rvs(res.ESS), density=d, alpha=0.15,
                        color='k', zorder=-1)
                ax[1].legend(['posterior', 'prior'])

            except Exception as e:
                print(str(e))

    else:
        estimate = percentile68_ranges_latex(res.nu)

        fig, ax = plt.subplots(1, 1, figsize=[8,6])
        ax.hist(res.nu, **kwargs)
        title = 'Posterior distribution for degrees of freedom $\\nu$ \n '\
                '%s' % estimate
        if kwargs.get('density', False):
            ylabel = 'posterior'
        else:
            ylabel = 'posterior samples'
        ax.set(xlabel='$\\nu$', ylabel=ylabel, title=title)

        if show_prior:
            try:
                # low, upp = res.priors['Jprior'].interval(1)
                d = kwargs.get('density', False)
                ax.hist(res.priors['Jprior'].rvs(res.ESS), density=d, alpha=0.15,
                        color='k', zorder=-1)
                ax.legend(['posterior', 'prior'])

            except Exception as e:
                print(str(e))

def plot_RVData(data, **kwargs):
    """ Simple plot of RV data. **kwargs are passed to plt.errorbar() """
    t = np.array(data.t).copy()
    y = np.array(data.y).copy()
    e = np.array(data.sig).copy()
    obs = np.array(data.obsi).copy()
    sb2 = data.double_lined
    if sb2:
        y2 = np.array(data.y2).copy()
        e2 = np.array(data.sig2).copy()


    time_offset = False
    if t[0] > 24e5:
        time_offset = True
        t -= 24e5

    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    kw = dict(fmt='o', ms=3)
    kw.update(**kwargs)

    if data.multi:
        uobs = np.unique(obs)
        for i in uobs:
            mask = obs == i
            ax.errorbar(t[mask], y[mask], e[mask], **kw)  
            if sb2:
                ax.errorbar(t[mask], y2[mask], e2[mask], mfc='none', **kw)  
    else:
        ax.errorbar(t, y, e,  **kw)
        if sb2:
            ax.errorbar(t, y2, e2, mfc='none', **kw)

    if time_offset:
        ax.set(xlabel='BJD - 2400000 [days]', ylabel='RV [m/s]')
    else:
        ax.set(xlabel='Time [days]', ylabel='RV [m/s]')
    return fig, ax

def plot_HGPMdata(data, pm_ra_bary=None, pm_dec_bary=None, 
                  show_legend=True, **kwargs):
    fig, axs = plt.subplots(1, 4, width_ratios=[4, 0.6, 4, 0.6], #height_ratios=[4, 2], 
                            constrained_layout=True, figsize=(7, 3))

    if pm_ra_bary is None and pm_dec_bary is None:
        f1, f2 = 1.0, 1.0
    else:
        f1, f2 = pm_ra_bary, pm_dec_bary

    kwH = dict(fmt='o', ms=4, color='C0')
    kwG = dict(fmt='o', ms=4, color='C1')
    kw = dict(fmt='o', ms=4, color='k')

    axs[0].errorbar(data.epoch_ra_hip - 5e4, data.pm_ra_hip, data.sig_hip_ra, **kwH)
    axs[0].errorbar(data.epoch_ra_gaia - 5e4, data.pm_ra_gaia, data.sig_gaia_ra, **kwG)
    axs[1].errorbar(0.5, data.pm_ra_hg, data.sig_hg_ra, **kw, mfc="w")
    axs[1].axhline(data.pm_ra_hg, color="k", zorder=-1)
    axs[1].set(yticks=[], xticks=[], xlim=(0, 1))
    axs[1].sharey(axs[0])
    axs[1].tick_params(left=False, labelleft=False)

    axs[2].errorbar(data.epoch_dec_hip - 5e4, data.pm_dec_hip, data.sig_hip_dec, **kwH)
    axs[2].errorbar(data.epoch_dec_gaia - 5e4, data.pm_dec_gaia, data.sig_gaia_dec, **kwG)
    axs[3].errorbar(0.5, data.pm_dec_hg, data.sig_hg_dec, **kw, mfc='w')
    axs[3].axhline(data.pm_dec_hg, color='k', zorder=-1)
    axs[3].set(yticks=[], xticks=[])
    axs[3].sharey(axs[2])
    axs[3].tick_params(left=False, labelleft=False)

    axs[0].set(xlabel='Epoch [BJD - 2450000]', ylabel=r'$\mu$ RA [mas/yr]')
    axs[2].set(xlabel='Epoch [BJD - 2450000]', ylabel=r'$\mu$ Dec [mas/yr]')
    # mi, ma = min(axs[0].get_ylim()[0], axs[1].get_ylim()[0]), max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
    # axs[0].set(ylim=(mi, ma))
    # axs[1].set(ylim=(mi, ma))
    # axs[1, 1].axis('off')
    # axs[1, 3].axis('off')

    if show_legend:
        axs[0].legend(['Hipparcos', 'Gaia'], ncols=2,
                      bbox_to_anchor=(0, 1.11), loc='upper left')
    return fig, axs


def plot_data(res, ax=None, axf=None, axr=None, y=None, e=None, y2=None, y3=None, extract_offset=True,
              ignore_y2=False, ignore_y3=False, time_offset=0.0, highlight=None,
              legend=True, show_rms=False, outliers=None, offsets=None, secondary_star=False, **kwargs):

    fwhm_model = res.model is MODELS.RVFWHMmodel and not ignore_y2
    rhk_model = res.model is MODELS.RVFWHMRHKmodel and not (ignore_y3 or ignore_y2)
    sb2 = False
    if res.model is MODELS.BINARIESmodel:
        if res.double_lined:
            sb2 = True

    if ax is None:
        if fwhm_model:
            fig, (ax, axf) = plt.subplots(2, 1, sharex=True)
        elif rhk_model:
            fig, (ax, axf, axr) = plt.subplots(3, 1, sharex=True)
        else:
            fig, ax = plt.subplots(1, 1)

    t = res.data.t.copy()

    if y is None:
        if sb2 and secondary_star:
            y = res.data.y2.copy()
            e = res.data.e2.copy()
        else:
            y = res.data.y.copy()
            e = res.data.e.copy()
    else:
        if y.ndim > 1:
            y = y[0]
        assert e is not None, 'must provide `e` if `y` is not None'

    if fwhm_model or rhk_model:
        y2 = res.data.y2.copy()
        e2 = res.data.e2.copy()
    
    if rhk_model:
        y3 = res.data.y3.copy()
        e3 = res.data.e3.copy()

    assert y.size == res.data.N, 'wrong dimensions!'

    if offsets is not None:
        extract_offset = False

    if extract_offset:
        if isinstance(extract_offset, float):
            y_offset = extract_offset
        else:
            y_offset = round(y.mean(), 0) if abs(y.mean()) > 100 else 0
        if fwhm_model or rhk_model:
            y2_offset = round(y2.mean(), 0) if abs(y2.mean()) > 100 else 0
        if rhk_model:
            y3_offset = round(y3.mean(), 0) if abs(y3.mean()) > 100 else 0
    else:
        y_offset = 0
        if fwhm_model or rhk_model:
            y2_offset = 0
        if rhk_model:
            y3_offset = 0

    kw = dict(fmt='o', ms=3)
    kw.update(**kwargs)

    if res.multi:
        if offsets is not None:
            msg = f'expected {res.n_instruments} offsets, got {offsets.size}'
            assert offsets.size == res.n_instruments, msg

        for j in range(res.n_instruments):
            inst = res.instruments[j]
            m = res.data.obs == j + 1
            kw.update(label=inst)
            if sb2:
                if secondary_star:
                    kw.update(label=inst+'_sec')
                else:
                    kw.update(label=inst+'_pri')

            if highlight is not None:
                if highlight in inst:
                    kw.update(alpha=1)
                else:
                    kw.update(alpha=0.1)

            if offsets is not None:
                y[m] -= offsets[j]

            if outliers is None:
                ax.errorbar(t[m] - time_offset, y[m] - y_offset, e[m], **kw)
            else:
                ax.errorbar(t[m & ~outliers] - time_offset,
                            y[m & ~outliers] - y_offset, e[m & ~outliers],
                            **kw)
            if fwhm_model or rhk_model:
                axf.errorbar(t[m] - time_offset, y2[m] - y2_offset, e2[m], **kw)
            if rhk_model:
                axr.errorbar(t[m] - time_offset, y3[m] - y3_offset, e3[m], **kw)
    else:
        try:
            kw.update(label=res.data.instrument)
            if sb2:
                if secondary_star:
                    kw.update(label=res.data.instrument+'_sec')
                else:
                    kw.update(label=res.data.instrument+'_pri')
            if kw['label'] == '':
                raise AttributeError
        except AttributeError:
            kw.update(label=res.instruments)

        if offsets is not None:
            y -= offsets

        if outliers is None:
            ax.errorbar(t - time_offset, y - y_offset, e, **kw)
        else:
            ax.errorbar(t[~outliers] - time_offset, y[~outliers] - y_offset, e[~outliers], **kw)
        if fwhm_model or rhk_model:
            axf.errorbar(t - time_offset, y2 - y2_offset, e2, **kw)
        if rhk_model:
            axr.errorbar(t - time_offset, y3 - y3_offset, e3, **kw)

    if legend:
        ax.legend(loc='best')

    if res.arbitrary_units:
        lab = dict(xlabel='Time [days]', ylabel='Q [arbitrary]')
    else:
        lab = dict(xlabel='Time [days]', ylabel='RV [m/s]')

    ax.set(**lab)
    if fwhm_model or rhk_model:
        axf.set(xlabel='Time [days]', ylabel='FWHM [m/s]')
    if rhk_model:
        axr.set(xlabel='Time [days]', ylabel=r"$\log$ R'$_{HK}$")

    if show_rms:
        rms = wrms(y, 1 / e**2)
        if outliers is None or not np.any(outliers):
            title = f'rms: {rms:.2f} [m/s]'
        else:
            rms_out = wrms(y[~outliers], 1 / e[~outliers]**2)
            title = f'rms: {rms:.2f} ({rms_out:.2f} w/o outliers) [m/s]'

        ax.set_title(title, loc='right', fontsize=10)

    if y_offset != 0:
        sign_symbol = {1.0: '+', -1.0: '-'}
        offset = sign_symbol[np.sign(y_offset)] + str(int(abs(y_offset)))
        fs = ax.xaxis.get_label().get_fontsize()
        ax.set_title(offset, loc='left', fontsize=fs)

    if (fwhm_model or rhk_model) and y2_offset != 0:
        sign_symbol = {1.0: '+', -1.0: '-'}
        offset = sign_symbol[np.sign(y2_offset)] + str(int(abs(y2_offset)))
        fs = axf.xaxis.get_label().get_fontsize()
        axf.set_title(offset, loc='left', fontsize=fs)
    
    if rhk_model and y3_offset != 0:
        sign_symbol = {1.0: '+', -1.0: '-'}
        offset = sign_symbol[np.sign(y3_offset)] + str(int(abs(y3_offset)))
        fs = axr.xaxis.get_label().get_fontsize()
        axr.set_title(offset, loc='left', fontsize=fs)

    if fwhm_model:
        return ax, axf, y_offset, y2_offset
    elif rhk_model:
        return ax, axf, axr, y_offset, y2_offset, y3_offset
    else:
        return ax, y_offset


def plot_data_jitters(res, sample, ax=None, axf=None, axr=None,
                      extract_offset=True, ignore_y2=False, ignore_y3=False,
                      time_offset=0.0, highlight=None, legend=True,
                      show_rms=False, outliers=None, **kwargs):

    fwhm_model = res.model is MODELS.RVFWHMmodel and not ignore_y2
    rhk_model = res.model is MODELS.RVFWHMRHKmodel and not ignore_y3

    if ax is None:
        if fwhm_model:
            fig, (ax, axf) = plt.subplots(2, 1, sharex=True)
        elif rhk_model:
            fig, (ax, axf, axr) = plt.subplots(3, 1, sharex=True)
        else:
            fig, ax = plt.subplots(1, 1)

    t = res.data.t.copy()
    e = res.data.e.copy()
    y = res.data.y.copy()

    jRV = sample[res.indices['jitter']][:res.n_instruments]
    jRV = jRV[res.data.obs.astype(int) - 1]

    if fwhm_model or rhk_model:
        y2 = res.data.y2.copy()
        e2 = res.data.e2.copy()
    
        jFW = sample[res.indices['jitter']][res.n_instruments : 2*res.n_instruments]
        jFW = jFW[res.data.obs.astype(int) - 1]

    if rhk_model:
        y3 = res.data.y3.copy()
        e3 = res.data.e3.copy()

        jRHK = sample[res.indices['jitter']][2*res.n_instruments:]
        jRHK = jRHK[res.data.obs.astype(int) - 1]

    assert y.size == res.data.N, 'wrong dimensions!'

    if extract_offset:
        y_offset = round(y.mean(), 0) if abs(y.mean()) > 100 else 0
        if fwhm_model or rhk_model:
            y2_offset = round(y2.mean(), 0) if abs(y2.mean()) > 100 else 0
        if rhk_model:
            y3_offset = round(y3.mean(), 0) if abs(y3.mean()) > 100 else 0
    else:
        y_offset = 0
        if fwhm_model or rhk_model:
            y2_offset = 0
        if rhk_model:
            y3_offset = 0

    kw = dict(fmt='o', ms=0, ecolor='k', alpha=0.1)
    kw.update(**kwargs)

    if res.multi:
        for j in range(res.n_instruments):
            inst = res.instruments[j]
            m = res.data.obs == j + 1
            kw.update(label=inst)

            if highlight is not None:
                if highlight in inst:
                    kw.update(alpha=1)
                else:
                    kw.update(alpha=0.1)

            if outliers is None:
                ax.errorbar(t[m] - time_offset, y[m] - y_offset, np.hypot(e[m], jRV[m]), **kw)
            else:
                ax.errorbar(t[m & ~outliers] - time_offset,
                            y[m & ~outliers] - y_offset, np.hypot(e[m & ~outliers], jRV[m & ~outliers]),
                            **kw)
            if fwhm_model or rhk_model:
                axf.errorbar(t[m] - time_offset, y2[m] - y2_offset, np.hypot(e2[m], jFW[m]), **kw)
            if rhk_model:
                axr.errorbar(t[m] - time_offset, y3[m] - y3_offset, np.hypot(e3[m], jRHK[m]), **kw)
    else:
        kw.update(label=res.data.instrument)

        if outliers is None:
            ax.errorbar(t - time_offset, y - y_offset, np.hypot(e, jRV), **kw)
        else:
            ax.errorbar(t[~outliers] - time_offset, y[~outliers] - y_offset, np.hypot(e[~outliers], jRV[~outliers]), **kw)
        if fwhm_model or rhk_model:
            axf.errorbar(t - time_offset, y2 - y2_offset, np.hypot(e2, jFW), **kw)
        if rhk_model:
            axr.errorbar(t - time_offset, y3 - y3_offset, np.hypot(e3, jRHK), **kw)

    if legend:
        ax.legend(loc='best')

    if res.arbitrary_units:
        lab = dict(xlabel='Time [days]', ylabel='Q [arbitrary]')
    else:
        lab = dict(xlabel='Time [days]', ylabel='RV [m/s]')

    ax.set(**lab)
    if fwhm_model or rhk_model:
        axf.set(xlabel='Time [days]', ylabel='FWHM [m/s]')
    if rhk_model:
        axr.set(xlabel='Time [days]', ylabel=r"$\log$ R'$_{HK}$")

    if show_rms:
        rms = wrms(y, 1 / e**2)
        if outliers is None or not np.any(outliers):
            title = f'rms: {rms:.2f} [m/s]'
        else:
            rms_out = wrms(y[~outliers], 1 / e[~outliers]**2)
            title = f'rms: {rms:.2f} ({rms_out:.2f} w/o outliers) [m/s]'

        ax.set_title(title, loc='right', fontsize=10)

    if y_offset != 0:
        sign_symbol = {1.0: '+', -1.0: '-'}
        offset = sign_symbol[np.sign(y_offset)] + str(int(abs(y_offset)))
        fs = ax.xaxis.get_label().get_fontsize()
        ax.set_title(offset, loc='left', fontsize=fs)

    if (fwhm_model or rhk_model) and y2_offset != 0:
        sign_symbol = {1.0: '+', -1.0: '-'}
        offset = sign_symbol[np.sign(y2_offset)] + str(int(abs(y2_offset)))
        fs = axf.xaxis.get_label().get_fontsize()
        axf.set_title(offset, loc='left', fontsize=fs)
    
    if rhk_model and y3_offset != 0:
        sign_symbol = {1.0: '+', -1.0: '-'}
        offset = sign_symbol[np.sign(y3_offset)] + str(int(abs(y3_offset)))
        fs = axr.xaxis.get_label().get_fontsize()
        axr.set_title(offset, loc='left', fontsize=fs)

    if fwhm_model:
        return ax, axf, y_offset, y2_offset
    elif rhk_model:
        return ax, axf, axr, y_offset, y2_offset, y3_offset
    else:
        return ax, y_offset


def gls_data(res, sample=None, ax=None):
    from gatspy.periodic import LombScargle, LombScargleMultiband
    from astropy.timeseries import LombScargle as GLS
    fwhm_model = res.model is MODELS.RVFWHMmodel #and not ignore_y2

    if ax is None:
        kw = dict(sharex=True, constrained_layout=True)
        if fwhm_model:
            fig, (axw, ax, axf) = plt.subplots(3, 1, **kw)
        else:
            fig, (axw, ax) = plt.subplots(2, 1, **kw)

    window_function = GLS(res.data.t, np.ones_like(res.data.t), res.data.e,
                          fit_mean=False, center_data=False)
    freq, power = window_function.autopower()
    axw.semilogx(1 / freq, power)

    if res.multi:
        model = LombScargleMultiband(Nterms_base=1, Nterms_band=0)
        model.fit(res.data.t, res.data.y, res.data.e, filts=res.data.obs)
        # power = model.periodogram(period)
    else:
        model = LombScargle()
        model.fit(res.data.t, res.data.y, res.data.e)

    period, power = model.periodogram_auto(oversampling=30)
    ax.semilogx(period, power)

    if fwhm_model:
        if res.multi:
            model = LombScargleMultiband(Nterms_base=1, Nterms_band=0)
            model.fit(res.data.t, res.data.y2, res.data.e2, filts=res.data.obs)
            # power = model.periodogram(period)
        else:
            model = LombScargle()
            model.fit(res.data.t, res.data.y2, res.data.e2)

        period, power = model.periodogram_auto(oversampling=30)
        axf.semilogx(period, power)


def plot_transit_data(res, ax=None, y=None, extract_offset=False,
                      time_offset=0.0, legend=True, show_rms=False, **kwargs):

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if y is None:
        y = res.data.y.copy()

    assert y.size == res.data.t.size, 'wrong dimensions!'

    if extract_offset:
        y_offset = round(y.mean(), 0) if abs(y.mean()) > 100 else 0
    else:
        y_offset = 0

    kw = dict(fmt='o')
    kw.update(**kwargs)

    if res.multi:
        for j in range(res.n_instruments):
            inst = res.instruments[j]
            m = res.data.obs == j + 1
            kw.update(label=inst)
            ax.errorbar(res.data.t[m] - time_offset, y[m] - y_offset,
                        res.data.e[m], **kw)
    else:
        kw.update(label=res.instruments)

        ax.errorbar(res.data.t - time_offset, y - y_offset, res.data.e, **kw)

    if legend:
        ax.legend(loc='upper left')

    if res.multi:
        kw = dict(color='b', lw=2, alpha=0.1, zorder=-2)
        for ot in res._offset_times:
            ax.axvline(ot - time_offset, **kw)

    lab = dict(xlabel='Time [days]', ylabel='Flux')
    ax.set(**lab)

    if show_rms:
        # if res.studentt:
        #     outliers = find_outliers(res)
        #     rms1 = wrms(y, 1 / res.e**2)
        #     rms2 = wrms(y[~outliers], 1 / res.e[~outliers]**2)
        #     ax.set_title(f'rms: {rms2:.2f} ({rms1:.2f}) m/s', loc='right')
        # else:
        rms = wrms(y, 1 / res.data.e**2)
        ax.set_title(f'rms: {rms:.2f} m/s', loc='right', fontsize=10)

    if y_offset != 0:
        sign_symbol = {1.0: '+', -1.0: '-'}
        offset = sign_symbol[np.sign(y_offset)] + str(int(abs(y_offset)))
        fs = ax.xaxis.get_label().get_fontsize()
        ax.set_title(offset, loc='left', fontsize=fs)

    return ax, y_offset




def corner_known_object(res, star_mass=1.0, adda=False, **kwargs):
    if not res.KO:
        print('Model has no known object! '
              'corner_known_object() doing nothing...')
        return

    import pygtc
    labels = [r'$P$', r'$K$', r'$e$', r'$M_0$', r'$\omega$']

    for i in range(res.nKO):
        data = res.KOpars[:, i::res.nKO].copy()
        # swtich M0 and ecc, just for convenience
        data[:, [3, 2]] = data[:, [2, 3]]

        fig = pygtc.plotGTC(
            chains=data,
            smoothingKernel=0,
            paramNames=labels,
            plotDensity=False,
            labelRotation=(True, False),
            filledPlots=False,
            colorsOrder=['blues_old'],
            figureSize='AandA_page',  # AandA_column
        )
        axs = fig.axes
        for ax in axs:
            ax.yaxis.set_label_coords(-0.3, 0.5, transform=None)
            ax.xaxis.set_label_coords(0.5, -0.4, transform=None)

        fs = axs[-1].xaxis.label.get_fontsize()
        start = len(labels)
        for i in range(-start, 0):
            val = percentile68_ranges_latex(data.T[i])
            # print(f'{names[i]} = {val}', fs)
            axs[i].set_title(f'{labels[i]} = {val}', fontsize=fs - 1)

        fig.tight_layout()

    # labels = [f'{n}{u}' for n, u in zip(names, units)]
    # print(data.shape, labels)


    # fs = axs[-1].xaxis.label.get_fontsize()
    # start = -3 if fixed_ecc else -4
    # start = start - 1 if adda else start


    # # fig = corner(np.c_[P, K, E, M], labels=labels, show_titles=True, **kwargs)
    fig.subplots_adjust(wspace=0.15, hspace=0)
    # return fig, inds, (M, A)


def phase_plot_logic(res, sample, sort_by_decreasing_K=False, sort_by_increasing_P=False):
    from string import ascii_lowercase
    letters = ascii_lowercase[1:]

    nplanets = int(sample[res.indices['np']])

    if res.model is MODELS.RVGAIAmodel:
        da,dd,mua,mud,plx = sample[res.indices['astrometric_solution']]

    params = {letters[i]: {} for i in range(nplanets)}
    for i, k in enumerate(params.keys()):
        params[k]['P'] = P = sample[res.indices['planets.P']][i]
        params[k]['e'] = e = sample[res.indices['planets.e']][i]
        params[k]['φ'] = φ = sample[res.indices['planets.φ']][i]
        params[k]['w'] = w = sample[res.indices['planets.w']][i]
        if res.model is MODELS.RVGAIAmodel:
            params[k]['a0'] = a0 = sample[res.indices['planets.a0']][i]
            params[k]['cosi'] = cosi = sample[res.indices['planets.cosi']][i]
            params[k]['K'] = K = Kfroma0(P,a0,e,cosi,plx)
        else:
            params[k]['K'] = K = sample[res.indices['planets.K']][i]
        params[k]['Tp'] = res.M0_epoch - (P * φ) / (2*np.pi)
        params[k]['type'] = 'planet'
        params[k]['index'] = i + 1

    pj = 0
    if res.KO:
        ko = {letters[i]: {} for i in range(nplanets, nplanets + res.nKO)}
        nplanets += res.nKO
        for i, k in enumerate(ko.keys()):
            ko[k]['P'] = P = sample[res.indices['KOpars']][i]
            if res.model is MODELS.BINARIESmodel:
                ko[k]['K'] = K = sample[res.indices['KOpars']][i + res.nKO]
                if res.double_lined:
                    ko[k]['q'] = q = sample[res.indices['KOpars']][i + 2 * res.nKO]
                    ko[k]['φ'] = φ = sample[res.indices['KOpars']][i + 3 * res.nKO]
                    ko[k]['e'] = e = sample[res.indices['KOpars']][i + 4 * res.nKO]
                    ko[k]['w'] = w = sample[res.indices['KOpars']][i + 5 * res.nKO]
                    ko[k]['wdot'] = wdot = sample[res.indices['KOpars']][i + 6 * res.nKO]
                    ko[k]['cosi'] = cosi = sample[res.indices['KOpars']][i + 7 * res.nKO]
                else:
                    ko[k]['φ'] = φ = sample[res.indices['KOpars']][i + 2 * res.nKO]
                    ko[k]['e'] = e = sample[res.indices['KOpars']][i + 3 * res.nKO]
                    ko[k]['w'] = w = sample[res.indices['KOpars']][i + 4 * res.nKO]
                    ko[k]['wdot'] = wdot = sample[res.indices['KOpars']][i + 5 * res.nKO]
                    ko[k]['cosi'] = cosi = sample[res.indices['KOpars']][i + 6 * res.nKO]
            else:
                ko[k]['φ'] = φ = sample[res.indices['KOpars']][i + 2 * res.nKO]
                ko[k]['e'] = e = sample[res.indices['KOpars']][i + 3 * res.nKO]
                ko[k]['w'] = w = sample[res.indices['KOpars']][i + 4 * res.nKO]
                if res.model is MODELS.RVGAIAmodel:
                    ko[k]['a0'] = a0 = sample[res.indices['KOpars']][i + res.nKO]
                    ko[k]['cosi'] = cosi = sample[res.indices['KOpars']][i + 5 * res.nKO]
                    ko[k]['K'] = K = Kfroma0(P,a0,e,cosi,plx)
                else:
                    ko[k]['K'] = K = sample[res.indices['KOpars']][i + res.nKO]
            ko[k]['Tp'] = res.M0_epoch - (P * φ) / (2*np.pi)
            ko[k]['type'] = 'KO'
            ko[k]['index'] = -pj - 1
            pj += 1
        params.update(ko)

    if res.TR:
        tr = {letters[i]: {} for i in range(nplanets, nplanets + res.nTR)}
        nplanets += res.nTR
        for i, k in enumerate(tr.keys()):
            tr[k]['P'] = P = sample[res.indices['TRpars']][i]
            tr[k]['K'] = K = sample[res.indices['TRpars']][i + res.nTR]
            tr[k]['Tc'] = Tc = sample[res.indices['TRpars']][i + 2 * res.nTR]
            tr[k]['e'] = e = sample[res.indices['TRpars']][i + 3 * res.nTR]
            tr[k]['w'] = w = sample[res.indices['TRpars']][i + 4 * res.nTR]
            f = np.pi/2 - w
            E = 2.0 * np.arctan(np.tan(f/2.0) * np.sqrt((1.0 - e) / (1.0 + e)))
            tr[k]['φ'] = φ = E - e * np.sin(E)
            # tr[k]['Tp'] = res.M0_epoch - (P * φ) / (2*np.pi)
            tr[k]['type'] = 'TR'
            tr[k]['index'] = -pj - 1
            pj += 1
        params.update(tr)

    keys = list(params.keys())

    if sort_by_decreasing_K:
        keys = sorted(params, key=lambda i: params[i]['K'], reverse=True)

    if sort_by_increasing_P:
        keys = sorted(params, key=lambda i: params[i]['P'])
    
    # print(nplanets)
    # print(params)
    # print(keys)
    return nplanets, params, keys


def phase_plot(res, sample, phase_axs=None, xaxis='mean anomaly',
               sort_by_increasing_P=False, sort_by_decreasing_K=True,
               highlight=None, highlight_points=None, only=None,
               show_titles=True, sharey=False, show_gls_residuals=False,
               show_outliers=False, fancy_ticks=False, dates='BJD', date_sub=None, 
               colormap='plasma', include_jitter=False,  **kwargs):
    """
    Plot the planet phase curves, GP, and residuals, for a given `sample`.
    
    Args:
        res (kima.KimaResults):
            The `KimaResults` instance
        sample (array):
            Array with one posterior sample
        phase_axs (list[matplotlib.axes.Axes]):
            One or more axes for the phase plot(s)
        xaxis (str):
            Plot the phase curve against 'mean anomaly' or 'mean longitude'
        sort_by_increasing_P (bool):
            Sort the planets by increasing period
        sort_by_decreasing_K (bool):
            Sort the planets by decreasing semi-amplitude
        highlight (list):
            Highlight all data points from a specific instrument
        highlight_points (list):
            Highlight specific data points by index
        only (list):
            Only show data from specific instrument(s)
        show_titles (bool):
            Add titles to each phase plot
        sharey (bool):
            Share the y-axis of the phase plots
        show_gls_residuals (bool):
            Add a panel with the Lomb-Scargle periodogram of the residuals
        show_outliers (bool):
            Show points identified as outliers
        fancy_ticks (bool):
            Use fancy ticks for angles
        **kwargs (dict):
            Keyword arguments passed to `plt.errorbar`
    
    Warning:
        This is probably the most complicated function in the whole package! For
        one, the layout of the axes in the figure may not always be optimal.
    """

    if res.max_components == 0 and not res.KO and not res.TR:
        print('Model has no planets! phase_plot() doing nothing...')
        return

    if res.model is MODELS.GAIAmodel:
        astrometry_phase_plot(res, sample, dates=dates, date_sub=date_sub, colormap=colormap,include_jitter=include_jitter)
        return
    elif res.model is MODELS.RVGAIAmodel:
        astrometry_phase_plot(res, sample, dates=dates, date_sub=date_sub, colormap=colormap,include_jitter=include_jitter)


    if xaxis not in ('mean anomaly', 'mean longitude'):
        raise ValueError(f'`xaxis` must be "mean anomaly" or "mean longitude", got {xaxis}')

    tau = 2 * np.pi

    # make copies to not change attributes
    t, y, e = res.data.t.copy(), res.data.y.copy(), res.data.e.copy()
    if res.model is MODELS.BINARIESmodel:
        if res.double_lined:
            y2, e2 = res.data.y2.copy(), res.data.e2.copy()
    obs = res.data.obs.copy()
    
    jitters = sample[res.indices['jitter']]
    if res.model is MODELS.RVGAIAmodel:
        jitters = jitters[1:]
    jitter_array = jitters[obs.astype(int) - 1]
    if res.model is MODELS.BINARIESmodel:
        if res.double_lined:
            jitter2_array = jitters[obs.astype(int) - 1 + res.n_instruments]

    if t[0] > 24e5:
        time_offset = 24e5
        time_label = 'Time ['+dates+' - 2400000]'
    elif t[0] > 5e4:
        time_offset = 5e4
        if dates in ['mjd','MJD']:
            time_label = 'Time ['+dates+' - 50000]'
        else:
            time_label = 'Time ['+dates+' - 2450000]'
    else:
        time_offset = 0
        time_label = 'Time ['+dates+']'

    if highlight_points is not None:
        hlkw = dict(fmt='*', ms=6, color='y', zorder=2)
        hl = highlight_points
        highlight_points = True

    nplanets, params, keys = phase_plot_logic(res, sample, 
                                              sort_by_decreasing_K, sort_by_increasing_P)

    if nplanets == 0:
        print('Sample has no planets! phase_plot() doing nothing...')
        return

    # subtract stochastic model and vsys / offsets from data
    v = res.full_model(sample, t=None, include_planets=False)

    if res.model in (MODELS.RVFWHMmodel, MODELS.RVFWHMRHKmodel, MODELS.SPLEAFmodel):
        v = v[0]
    if res.model is MODELS.BINARIESmodel:
        if res.double_lined:
            v2 = v[1]
            v = v[0]
            y2 = y2 - v2

    y = y - v

    # errorbar plot arguments
    colors = kwargs.pop('colors', None)
    ekwargs = kwargs
    ekwargs.setdefault('ms', 4)
    ekwargs.setdefault('capsize', 0)
    ekwargs.setdefault('elinewidth', 0.8)

    #sb2 change marker
    e2kwargs = ekwargs.copy()
    e2kwargs.setdefault('fmt', 'o')
    e2kwargs.setdefault('mfc', 'none')

    #now set default marker
    ekwargs.setdefault('fmt', 'o')
    ekwargs.setdefault('mec', 'none')

    # very complicated logic just to make the figure the right size
    fs = [
        max(7, 7 + 1.5 * (nplanets - 2)),
        max(6, 6 + 1 * (nplanets - 3))
    ]
    if res.has_gp:
        fs[1] += 3
    if show_gls_residuals:
        fs[0] += 2

    # axes layout:
    # # up to 3 planets: 1 row, up to 3 cols
    # # 4 planets: 2 rows, 2 cols
    # # up to 6 planets: 2 rows, 3 cols
    # GP panel ?
    # residuals panel


    fig = plt.figure(constrained_layout=True, figsize=fs)
    nrows = {
        1: 1, 2: 1, 3: 1,
        4: 2, 5: 2, 6: 2,
        7: 4, 8: 4, 9: 4,
    }[nplanets]

    # at least the residuals plot
    nrows += 1

    if res.has_gp:
        nrows += 1

    ncols = {
        1: 1, 2: 2, 3: 3,
        4: 2,
        5: 3, 6: 3, 7: 3, 8: 3, 9: 3
    }[nplanets]
    hr = [2] * (nrows - 1) + [1]
    wr = None

    if show_gls_residuals:
        wr = ncols * [2] + [1]
        ncols += 1
        # fig.set_size_inches(fs[0], fs[1])

    gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                           height_ratios=hr, width_ratios=wr)
    # gs_indices = {i: (i // 3, i % 3) for i in range(50)}

    if phase_axs is None:
        if nplanets == 1:
            axs = [fig.add_subplot(gs[0, 0])]
        elif nplanets == 2:
            axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        elif nplanets == 3:
            axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
        elif nplanets == 4:
            axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                   fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
        elif nplanets == 5:
            axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), 
                   fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
        elif nplanets == 6:
            axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
                   fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
        elif nplanets == 7:
            axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
                   fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),
                   fig.add_subplot(gs[2, 0])]
        elif nplanets == 8:
            axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
                   fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),
                   fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
        elif nplanets == 9:
            axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
                   fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),
                   fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2])]
        else:
            raise NotImplementedError
    else:
        axs = phase_axs

    for ax in axs:
        ax.minorticks_on()

    # for each planet in this sample
    for i, letter in enumerate(keys):
        ax = axs[i]

        # ax.axvline(0.5, ls='--', color='k', alpha=0.2, zorder=-5)
        # ax.axhline(0.0, ls='--', color='k', alpha=0.2, zorder=-5)
        ax.axvline(np.pi, ls='--', color='k', alpha=0.2, zorder=-5)
        ax.axhline(0.0, ls='--', color='k', alpha=0.2, zorder=-5)

        P = params[letter]['P']
        w = params[letter]['w']
        M0 = params[letter]['φ']
        # Tp = params[letter]['Tp']

        # plot the keplerian curve in phase (3 times)
        phase = np.linspace(0, tau, 200)
        if xaxis == 'mean anomaly':
            tt = (phase - M0) * P / tau + res.M0_epoch
        elif xaxis == 'mean longitude':
            tt = (phase - M0 - w) * P / tau + res.M0_epoch

        # Msmooth = np.linspace(0, 360, 200)
        # M0 = 180 / np.pi * (λ0 - w)
        # # M = (M0 + 360 / P * t) % 360
        # tt = (Msmooth - M0) * P / 360
        # # mod = kep.rv((Msmooth - M0) * P / 360)


        # keplerian for this planet
        planet_index = params[letter]['index']
        vv = res.eval_model(sample, tt, single_planet=planet_index)
        # the background model at these times
        offset_model = res.eval_model(sample, tt, include_planets=False)

        if res.model in (MODELS.RVFWHMmodel, MODELS.RVFWHMRHKmodel, MODELS.SPLEAFmodel):
            vv = vv[0]
            offset_model = offset_model[0]
        if res.model is MODELS.BINARIESmodel:
            if res.double_lined:
                vv2 = vv[1]
                offset_model2 = offset_model[1]
                vv = vv[0]
                offset_model = offset_model[0]


        for j in (-1, 0, 1):
            alpha = 0.2 if j in (-1, 1) else 1
            ax.plot(np.sort(phase) + j * tau, vv[np.argsort(phase)] - offset_model,
                    color='k', alpha=alpha, zorder=100)
        if res.model is MODELS.BINARIESmodel:
            if res.double_lined:
                for j in (-1, 0, 1):
                    alpha = 0.2 if j in (-1, 1) else 1
                    ax.plot(np.sort(phase) + j * tau, vv2[np.argsort(phase)] - offset_model2,
                            color='k', alpha=alpha, zorder=100)

        # subtract the other planets from the data and plot it (the data)
        vv = res.planet_model(sample, except_planet=planet_index)

        if res.model in (MODELS.RVFWHMmodel, MODELS.RVFWHMRHKmodel):
            vv = vv[0]
        if res.model is MODELS.BINARIESmodel:
            if res.double_lined:
                vv2 = vv[1]
                vv = vv[0]


        if res.studentt and show_outliers:
            outliers = find_outliers(res, sample)

        from .utils import mean_anomaly_from_epoch


        if res.multi:
            for k in range(1, res.n_instruments + 1):
                m = obs == k
                # phase = ((t[m] - res.M0_epoch) / P) % 1.0
                if xaxis == 'mean anomaly':
                    phase = mean_anomaly_from_epoch(t[m], P, M0, res.M0_epoch) % tau
                elif xaxis == 'mean longitude':
                    phase = (mean_anomaly_from_epoch(t[m], P, M0, res.M0_epoch) + w) % tau

                yy = (y - vv)[m]
                ee = e[m].copy()

                if include_jitter:
                    jit = jitters[k-1]
                    ee = np.hypot(ee,jit)

                if res.model is MODELS.BINARIESmodel:
                    if res.double_lined:
                        yy2 = (y2 - vv2)[m]
                        ee2 = e2[m].copy()
                        if include_jitter:
                            jit2 = jitters[k-1+res.n_instruments]
                            ee2 = np.hypot(ee2,jit2)

                if colors is None:
                    # one color for each instrument
                    color = f'C{k-1}'
                elif isinstance(colors, dict):
                    color = colors[res.instruments[k - 1]]
                else:
                    color = colors[k - 1]

                for j in (-1, 0, 1):
                    label = res.instruments[k - 1] if j == 0 else ''
                    alpha = 0.2 if j in (-1, 1) else 1
                    if highlight:
                        if highlight not in res.data_file[k - 1]:
                            alpha = 0.5
                    elif only:
                        if only not in res.data_file[k - 1]:
                            alpha = 0

                    _phi = np.sort(phase) + j * tau
                    _y = yy[np.argsort(phase)]
                    _e = ee[np.argsort(phase)]
                    ax.errorbar(_phi, _y, _e,
                                label=label, color=color, alpha=alpha, **ekwargs)
                    
                    if res.model is MODELS.BINARIESmodel:
                        if res.double_lined:
                            _phi = np.sort(phase) + j * tau
                            _y = yy2[np.argsort(phase)]
                            _e = ee2[np.argsort(phase)]
                            ax.errorbar(_phi, _y, _e,
                                        label=label, color=color, alpha=alpha, **e2kwargs)
                
                    if res.studentt and show_outliers:
                        _phi = np.sort(phase[outliers[m]]) + j * tau
                        _y = yy[outliers[m]][np.argsort(phase[outliers[m]])]
                        _e = ee[outliers[m]][np.argsort(phase[outliers[m]])]
                        ax.errorbar(_phi, _y, _e, fmt='xr', alpha=alpha, zorder=-10)
                        if res.model is MODELS.BINARIESmodel:
                            if res.double_lined:
                                _phi = np.sort(phase[outliers[m]]) + j * tau
                                _y = yy2[outliers[m]][np.argsort(phase[outliers[m]])]
                                _e = ee2[outliers[m]][np.argsort(phase[outliers[m]])]
                                ax.errorbar(_phi, _y, _e, fmt='xr', alpha=alpha, zorder=-10)

                    if highlight_points:
                        hlm = (m & hl)[m]
                        ax.errorbar(np.sort(phase[hlm]) + j,
                                    yy[np.argsort(phase[hlm])],
                                    ee[np.argsort(phase[hlm])],
                                    alpha=alpha, **hlkw)
                        if res.model is MODELS.BINARIESmodel:
                            if res.double_lined:
                                ax.errorbar(np.sort(phase[hlm]) + j,
                                    yy2[np.argsort(phase[hlm])],
                                    ee2[np.argsort(phase[hlm])],
                                    alpha=alpha, **hlkw)

        else:
            # phase = ((t - res.M0_epoch) / P) % 1.0
            if xaxis == 'mean anomaly':
                phase = mean_anomaly_from_epoch(t, P, M0, res.M0_epoch) % tau
            elif xaxis == 'mean longitude':
                phase = (mean_anomaly_from_epoch(t, P, M0, res.M0_epoch) + w) % tau

            yy = y - vv
            ee = e.copy()
            if include_jitter:
                jit = jitters[0]
                ee = np.hypot(ee,jit)


            for j in (-1, 0, 1):
                alpha = 0.3 if j in (-1, 1) else 1
                ax.errorbar(np.sort(phase) + j * tau, yy[np.argsort(phase)], ee[np.argsort(phase)],
                            color='C0', alpha=alpha, **ekwargs)
                
            if res.model is MODELS.BINARIESmodel:
                if res.double_lined:
                    yy2 = y2 - vv2
                    ee2 = e2.copy()
                    if include_jitter:
                        jit2 = jitters[1]
                        ee2 = np.hypot(ee2,jit2)

                    for j in (-1, 0, 1):
                        alpha = 0.3 if j in (-1, 1) else 1
                        ax.errorbar(np.sort(phase) + j * tau, yy2[np.argsort(phase)], ee2[np.argsort(phase)],
                                    color='C0', alpha=alpha, **e2kwargs)

        ax.set(xlabel=xaxis, ylabel="RV [m/s]")
        # ax.set_xlim(-0.1, 1.1)
        ax.set_xlim(-0.3, 2*np.pi+0.3)

        if fancy_ticks:
            ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
            ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

        if show_titles:
            # ax.set_title('%s' % letter, loc='left', **title_kwargs)
            K = params[letter]['K']
            ecc = params[letter]['e']
            title = f'{P=:.2f} days\n {K=:.2f} m/s  {ecc=:.2f}'
            title_kwargs = dict(fontsize=12)
            ax.set_title(title, loc='right', **title_kwargs)

    if sharey:
        ymin, ymax = axs[0].get_ylim()
        for ax in axs:
            _ymin, _ymax = ax.get_ylim()
            ymin = min(ymin, _ymin)
            ymax = max(ymax, _ymax)
            ax.sharey(axs[0])
        axs[0].set_ylim(ymin, ymax)

    end = -1 if show_gls_residuals else None

    try:
        overlap = res._time_overlaps[0]
    except ValueError:
        overlap = False

    ## GP panel
    ###########
    if res.has_gp:
        # grab the first row after the one that is attributed to the planets
        axGP = fig.add_subplot(gs[-2, :end])

        y = res.data.y.copy()
        y = y - res.eval_model(sample)

        _, y_offset = plot_data(res, y=y, e=res.data.e, ax=axGP, ignore_y2=True, legend=False,
                                highlight=highlight, only=only, time_offset=time_offset,
                                **ekwargs)
        axGP.set(xlabel=time_label, ylabel="GP [m/s]")

        tt = np.linspace(t[0], t[-1], 3000)
        no_planets_model = res.eval_model(sample, tt, include_planets=False)
        no_planets_model = res.burst_model(sample, tt, no_planets_model)

        if res.model in (MODELS.RVFWHMmodel, MODELS.RVFWHMRHKmodel, MODELS.SPLEAFmodel):
            (pred, *_), (std, *_) = res.stochastic_model(sample, tt, return_std=True)
            if overlap:
                no_planets_model = no_planets_model[::2]
            else:
                no_planets_model = no_planets_model[0]
        
        else:
            pred, std = res.stochastic_model(sample, tt, return_std=True)

        # pred = pred #+ no_planets_model - y_offset
        pred = np.atleast_2d(pred)
        for p in pred:
            axGP.plot(tt - time_offset, p, 'k')
            axGP.fill_between(tt - time_offset, p - 2 * std, p + 2 * std,
                              color='m', alpha=0.2)

    ## residuals
    ############
    ax = fig.add_subplot(gs[-1, :end])
    residuals = res.residuals(sample, full=True)

    if include_jitter:
        errors = np.hypot(res.data.e.copy(),jitter_array)
    else:
        errors = res.data.e.copy()

    if res.model is MODELS.BINARIESmodel:
        if res.double_lined:
            if include_jitter:
                errors_2 = np.hypot(res.data.e2.copy(),jitter2_array)
            else:
                errors_2 = res.data.e2.copy()


    if res.model in (MODELS.RVFWHMmodel, MODELS.RVFWHMRHKmodel):
        residuals = residuals[0]
    if res.model is MODELS.BINARIESmodel:
        if res.double_lined:
            residuals2 = residuals[1]
            residuals = residuals[0]

    if res.studentt and show_outliers:
        outliers = find_outliers(res, sample)
        ax.errorbar(res.data.t[outliers] - time_offset, residuals[outliers],
                    errors[outliers], fmt='xr', ms=7, lw=3, zorder=-10)
        if res.model is MODELS.BINARIESmodel:
            if res.double_lined:
                # outliers = find_outliers(res, sample) #Maybe necessary to edit the function for sb2s and include again
                ax.errorbar(res.data.t[outliers] - time_offset, residuals2[outliers],
                    errors_2[outliers], fmt='xr', ms=7, lw=3, zorder=-10)
    else:
        outliers = None


    plot_data(res, ax=ax, y=residuals, e=errors, ignore_y2=True, legend=True,
              show_rms=True, time_offset=time_offset, outliers=outliers,
              highlight=highlight, **ekwargs)

    if res.model is MODELS.BINARIESmodel:
            if res.double_lined:
                plt.gca().set_prop_cycle(None)
                plot_data(res, ax=ax, y=residuals2, e=errors_2, ignore_y2=True, legend=True,
              show_rms=True, time_offset=time_offset, outliers=outliers,
              highlight=highlight, secondary_star=True, **e2kwargs)

    # legend in the residual plot?
    hand, lab = ax.get_legend_handles_labels()
    ncol = 2 if res.n_instruments % 2 == 0 else 3
    leg = ax.legend(hand, lab, loc='upper left', ncol=ncol, borderaxespad=0.,
                    borderpad=0.3, bbox_to_anchor=(0.0, 1.3), handletextpad=0,
                    columnspacing=0.1)

    if highlight_points:
        ax.errorbar(t[hl], residuals[hl], e[hl], **hlkw)


    ax.axhline(y=0, ls='--', alpha=0.5, color='k')
    ax.set_ylim(np.tile(np.abs(ax.get_ylim()).max(), 2) * [-1, 1])
    ax.set(xlabel=time_label, ylabel='r [m/s]')
    title_kwargs = dict(loc='right', fontsize=12)


    if show_gls_residuals:
        axp = fig.add_subplot(gs[:, -1])
        from astropy.timeseries import LombScargle
        gls = LombScargle(res.data.t, residuals, res.data.e)
        freq, power = gls.autopower(maximum_frequency=1.0, minimum_frequency=1/np.ptp(res.data.t))
        axp.semilogy(power, 1 / freq, 'k', alpha=0.6)

        kwl = dict(color='k', alpha=0.2, ls='--')
        kwt = dict(color='k', alpha=0.3, rotation=90, ha='left', va='top', fontsize=9)
        fap001 = gls.false_alarm_level(0.01)
        axp.axvline(fap001, **kwl)
        axp.text(0.98 * fap001, 1 / freq.min(), '1%', **kwt)

        fap01 = gls.false_alarm_level(0.1)
        axp.axvline(fap01, **kwl)
        axp.text(0.98 * fap01, 1 / freq.min(), '10%', **kwt)

        axp.set(xlabel='residual power', ylabel='Period [days]')
        axp.invert_xaxis()
        axp.yaxis.tick_right()
        axp.yaxis.set_label_position('right')


    if res.save_plots:
        filename = 'kima-showresults-fig6.1.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig

    return residuals

def astrometry_phase_plot_logic(res, sample, sort_by_decreasing_a=False, sort_by_increasing_P=False):
    from string import ascii_lowercase
    letters = ascii_lowercase[1:]

    nplanets = int(sample[res.indices['np']])

    

    pj = 0
    nKOs = 0
    if res.KO:
        nKOs = res.nKO
        # ko = {letters[i]: {} for i in range(nplanets, nplanets + res.nKO)}
        params = {letters[i]: {} for i in range(nKOs)}
        nplanets += res.nKO
        for i, k in enumerate(params.keys()):
            params[k]['P'] = P = sample[res.indices['KOpars']][i]
            params[k]['a0'] = a0 = sample[res.indices['KOpars']][i + 1 * res.nKO]
            params[k]['φ'] = φ = sample[res.indices['KOpars']][i + 2 * res.nKO]
            params[k]['e'] = e = sample[res.indices['KOpars']][i + 3 * res.nKO]
            params[k]['w'] = w = sample[res.indices['KOpars']][i + 4 * res.nKO]
            params[k]['cosi'] = cosi = sample[res.indices['KOpars']][i + 5 * res.nKO]
            params[k]['W'] = W = sample[res.indices['KOpars']][i + 6 * res.nKO]
            params[k]['Tp'] = res.M0_epoch - (P * φ) / (2*np.pi)
            params[k]['type'] = 'KO'
            params[k]['index'] = -pj - 1
            pj += 1
    else:
        params = {}
        

    planets = {letters[i]: {} for i in range(nKOs, nplanets)}
    for i, k in enumerate(planets.keys()):
        planets[k]['P'] = P = sample[res.indices['planets.P']][i]
        planets[k]['φ'] = φ = sample[res.indices['planets.φ']][i]
        planets[k]['e'] = e = sample[res.indices['planets.e']][i]
        if res.thiele_innes:
            planets[k]['A'] = A = sample[res.indices['planets.A']][i]
            planets[k]['B'] = B = sample[res.indices['planets.B']][i]
            planets[k]['F'] = F = sample[res.indices['planets.F']][i]
            planets[k]['G'] = G = sample[res.indices['planets.G']][i]
        else:
            planets[k]['a0'] = a0 = sample[res.indices['planets.a0']][i]
            planets[k]['w'] = w = sample[res.indices['planets.w']][i]
            planets[k]['cosi'] = cosi = sample[res.indices['planets.cosi']][i]
            planets[k]['W'] = W = sample[res.indices['planets.W']][i]
        planets[k]['Tp'] = res.M0_epoch - (P * φ) / (2*np.pi)
        planets[k]['type'] = 'planet'
        planets[k]['index'] = i + 1
    params.update(planets)

    keys = list(params.keys())

    if sort_by_decreasing_a:
        keys = sorted(params, key=lambda i: params[i]['a0'], reverse=True)

    if sort_by_increasing_P:
        keys = sorted(params, key=lambda i: params[i]['P'])

    return nplanets, params, keys


def astrometry_phase_plot(res, sample, dates='jd', date_sub=None, colormap='plasma',include_jitter=False):
    try:
        from pystrometry.pystrometry import get_parallax_factors
    except ImportError:
        raise ImportError('pystrometry is required for astrometry phase plots')

    twopi = 2 * np.pi
    from ..kepler import brandt_solver

    def ellip_rectang(t, P, e, Tper):
        M = twopi * (t - Tper) / P
        E = brandt_solver(M, e)
        X = np.cos(E) - e
        Y = np.sqrt(1-e**2) * np.sin(E)
        return X, Y

    def Thiele_Innes(a0, w, W, cosi):
        A = a0*(np.cos(w)*np.cos(W) - np.sin(w)*np.sin(W)*cosi)
        B = a0*(np.cos(w)*np.sin(W) + np.sin(w)*np.cos(W)*cosi)
        F = -a0*(np.sin(w)*np.cos(W) + np.cos(w)*np.sin(W)*cosi)
        G = -a0*(np.sin(w)*np.sin(W) - np.cos(w)*np.cos(W)*cosi)
        return A, B, F, G

    def ra_dec_orb(P,Tper,e,cosi,W,w,a0,t):
        A, B, F, G = Thiele_Innes(a0, w, W, cosi)
        X, Y = ellip_rectang(t, P, e, Tper)
        return B*X + G*Y, A*X + F*Y
    
    def ra_dec_orb_TI(P,Tper,e,A,B,F,G,t):
        X, Y = ellip_rectang(t, P, e, Tper)
        return B*X + G*Y, A*X + F*Y

    def wss(da,dd,par,mua,mud,t,psi,pf,tref):
        T = t - tref
        return (da + mua*T)*np.sin(psi) + (dd +mud*T)*np.cos(psi) +par*pf

    def wk_orb(P,Tper,e,cosi,W,w,a0,t,psi):
        A, B, F, G = Thiele_Innes(a0, w, W, cosi)
        X, Y = ellip_rectang(t, P, e, Tper)
        return (B*X + G*Y)*np.sin(psi) + (A*X + F*Y)*np.cos(psi)
    
    def wk_orb_TI(P,Tper,e,A,B,F,G,t,psi):
        X, Y = ellip_rectang(t, P, e, Tper)
        return (B*X + G*Y)*np.sin(psi) + (A*X + F*Y)*np.cos(psi)
    
    def wss_dep(da,dd,par,mua,mud,t,pfra,pfdec,tref):
        #Obtain RA and DEC values for parallax and PM plot curve
        T = t - tref
        return (da + mua*T) + pfra*par,(dd +mud*T)+pfdec*par
    
    def wss_dep_errs(alpha_res,dec_res,alpha_err,dec_err,da,dd,par,mua,mud,t,pf_ra,pf_dec,tref):
        #Obtain RA and DEC values and errors for parallax and PM plot points
        T = t - tref
        ra = (da + mua*T) + pf_ra*par + alpha_res
        raplus = (da + mua*T) + pf_ra*par + alpha_res + alpha_err
        raminus = (da + mua*T) + pf_ra*par + alpha_res - alpha_err
        dec = (dd +mud*T) + pf_dec*par + dec_res
        decplus = (dd +mud*T) + pf_dec*par + dec_res + dec_err
        decminus = (dd +mud*T) + pf_dec*par + dec_res - dec_err
        return ra,raplus,raminus,dec,decplus,decminus    

    t = np.array(res.GAIAdata.t)
    t2 = t.copy()
    wobs = np.array(res.GAIAdata.w)
    ws_err = np.array(res.GAIAdata.wsig)
    psi = np.array(res.GAIAdata.psi)
    pf = np.array(res.GAIAdata.pf)

    if include_jitter:
        ws_err = np.hypot(ws_err,sample[res.indices['jitter']][0])

    alpha_errs = ws_err*np.sin(psi)
    dec_errs = ws_err*np.cos(psi)

    nplanets, params, keys = astrometry_phase_plot_logic(res,sample)

    da, dd, mua, mud, par = sample[res.indices['astrometric_solution']]
    # P, phi, e, a0, w, cosi, W = sample[res.indices['planets']]

    nrows = {
        0: 3, 1: 3, 2: 3, 3: 4,
        4: 5, 5: 5, 6: 6
    }[nplanets]

    # at least the residuals plot

    ncols = {
        0: 1, 1: 2, 2: 3, 3: 3,
        4: 3, 5: 3, 6: 3
    }[nplanets]

    fs = [ncols*4,nrows*2]

    fig = plt.figure(tight_layout=True, figsize=fs)

    gs = gridspec.GridSpec(nrows, ncols, figure=fig)
    # gs_indices = {i: (i // 3, i % 3) for i in range(50)}

    if nplanets == 0:
        axs = [fig.add_subplot(gs[:2, 0])]
        resax = fig.add_subplot(gs[2, 0])
    elif nplanets == 1:
        axs = [fig.add_subplot(gs[:2, 0]),fig.add_subplot(gs[:2, 1])]
        resax = fig.add_subplot(gs[2, :])
    elif nplanets == 2:
        axs = [fig.add_subplot(gs[:2, 0]),fig.add_subplot(gs[:2, 1]),fig.add_subplot(gs[:2, 2])]
        resax = fig.add_subplot(gs[2, :])
    elif nplanets == 3:
        axs = [fig.add_subplot(gs[:2, 0]), fig.add_subplot(gs[:2, 1]), fig.add_subplot(gs[:2, 2]), 
               fig.add_subplot(gs[2:, 0])]
        resax = fig.add_subplot(gs[2:, 1:])
    elif nplanets == 4:
        axs = [fig.add_subplot(gs[:2, 0]), fig.add_subplot(gs[:2, 1]), fig.add_subplot(gs[:2, 2]),
                fig.add_subplot(gs[2:4, 0]), fig.add_subplot(gs[2:4, 1])]
        resax = fig.add_subplot(gs[4, :])
    elif nplanets == 5:
        axs = [fig.add_subplot(gs[:2, 0]), fig.add_subplot(gs[:2, 1]), fig.add_subplot(gs[:2, 2]),
                fig.add_subplot(gs[2:4, 0]), fig.add_subplot(gs[2:4, 1]), fig.add_subplot(gs[2:4, 2])]
        resax = fig.add_subplot(gs[4, :])
    elif nplanets == 6:
        axs = [fig.add_subplot(gs[:2, 0]), fig.add_subplot(gs[:2, 1]), fig.add_subplot(gs[:2, 2]),
               fig.add_subplot(gs[2:4, 0]), fig.add_subplot(gs[2:4, 1]), fig.add_subplot(gs[2:4, 2]),
               fig.add_subplot(gs[4:, 0])]
        resax = fig.add_subplot(gs[4:, 1:])
    else:
        raise NotImplementedError
    
    wmodel = wss(da, dd, par, mua, mud, t, psi, pf, res.M0_epoch)
    
    #Get full model
    for letter in keys:
        # P, phi, e, a0, w, cosi, W, Tper, type, index = params[letter]
        P = params[letter]['P']
        phi = params[letter]['φ']
        e = params[letter]['e']
        if res.thiele_innes:
            A = params[letter]['A']
            B = params[letter]['B']
            F = params[letter]['F']
            G = params[letter]['G']
        else:
            a0 = params[letter]['a0']
            w = params[letter]['w']
            cosi = params[letter]['cosi']
            W = params[letter]['W']
            A,B,F,G = Thiele_Innes(a0,w,W,cosi)
        Tper = params[letter]['Tp']

        wmodel += wk_orb_TI(P, Tper, e, A, B, F, G, t, psi)
    #get residuals
    wws = wobs - wmodel
    alpha_res, dec_res = wws * np.sin(psi), wws * np.cos(psi)

    #make parallax-proper-motion panel
    if t[0] < 2400000 and dates != 'mjd' and date_sub==None:
        raise Exception('times are not labelled in mjd but values are < 2400000, please either specify dates=\'mjd\' or give a value to date_sub such as date_sub = 2400000 such that date is in jd (or jd with correction)')
    elif t[0] < 2400000 and dates != 'mjd':
        t2 += date_sub
        reft = res.M0_epoch + date_sub
        time_label = str(dates) + ' - ' + str(date_sub)
    elif t[0] < 2400000 and dates == 'mjd':
        t2 += 2400000.5
        reft = res.M0_epoch + 2400000.5
        time_label = 'mjd'
    else:
        reft = res.M0_epoch
        time_label = str(dates)


    time_array = np.arange(np.min(t),np.max(t),1.0)
    time_array2 = np.arange(np.min(t2),np.max(t2),1.0)
    ax = axs[0]

    if res.RA == 0.0 or res.DEC ==0.0:
        print('RA and/or DEC value is 0.0 in the kima model, parallax plot will not be generated')
    else:
        parfra,parfdec = get_parallax_factors(res.RA,res.DEC, time_array2,verbose=False,overwrite=False)
        parfra_vals,parfdec_vals = get_parallax_factors(res.RA,res.DEC, t2,verbose=False,overwrite=False)
        a,ap,am,b,bp,bm = wss_dep_errs(alpha_res,dec_res,alpha_errs,dec_errs,da,dd,par,mua,mud,np.array(t2),parfra_vals,parfdec_vals,reft)
        ax.scatter(a,b,c=np.array(t2),cmap=colormap)
        cmap = matplotlib.colormaps[colormap]
        for i in range(len(t2)):
            colour = cmap((t2[i]-t2[0])/(t2[len(t2)-1]-t2[0]))
            ax.plot([am[i],ap[i]],[bm[i],bp[i]],c=colour,alpha=0.6)
        a,b = wss_dep(da,dd,par,mua,mud,time_array2,parfra,parfdec,reft)
        ax.plot(a,b,c='black',alpha=0.8)

    ax.xaxis.set_inverted(True)
    ax.set_box_aspect(1)
    ax.set(xlabel=r'$\Delta \alpha\,\cos\delta$ [mas]', ylabel=r'$\Delta \delta$ [mas]',title='Parallax and Proper-Motion')


    #make individual orbit plots "phased"
    
    for j,letter in enumerate(keys):
        P = params[letter]['P']
        phi = params[letter]['φ']
        e = params[letter]['e']
        if res.thiele_innes:
            A = params[letter]['A']
            B = params[letter]['B']
            F = params[letter]['F']
            G = params[letter]['G']
        else:
            a0 = params[letter]['a0']
            w = params[letter]['w']
            cosi = params[letter]['cosi']
            W = params[letter]['W']
            A,B,F,G = Thiele_Innes(a0,w,W,cosi) 
        Tper = params[letter]['Tp']  
        

        ra, dec = ra_dec_orb_TI(P, Tper, e, A, B, F, G, t)
        ra2, dec2 = ra_dec_orb_TI(P, Tper, e, A, B, F, G, time_array)

        ax = axs[j+1]
        # ax.scatter(ra, dec, marker='o', c=t, cmap='plasma')
        ax.plot(ra2, dec2, color='k', lw=2, zorder=-1)
        ax.scatter(ra + alpha_res, dec + dec_res, c=t, cmap=colormap, alpha=1)
        cmap = matplotlib.colormaps[colormap]
        for i in range(len(t)):
            colour = cmap((t[i]-t[0])/(t[len(t)-1]-t[0]))
            ax.plot([ra[i]+alpha_res[i]-alpha_errs[i],ra[i]+alpha_res[i]+alpha_errs[i]],[dec[i]+dec_res[i]-dec_errs[i],dec[i]+dec_res[i]+dec_errs[i]],c=colour,alpha=0.6)

        #Add line connecting COM to pericentre
        ra_per, dec_per = ra_dec_orb_TI(P, Tper, e, A, B, F, G, Tper)
        ax.scatter(0,0,marker='x',c='grey')
        ax.plot([0,ra_per],[0,dec_per],c='grey',ls=':')

        #Make plot square to get good visual on e and inc
        lowx,highx = ax.get_xlim()
        lowy,highy = ax.get_ylim()
        xwidth = highx - lowx
        ywidth = highy - lowy
        if xwidth < ywidth:
            delta = ywidth - xwidth
            ax.set(xlim = [lowx - delta/2,highx + delta/2])
        else:
            delta = xwidth - ywidth
            ax.set(ylim = [lowy - delta/2,highy + delta/2])
        ax.xaxis.set_inverted(True)

        ax.set_box_aspect(1)
        ax.set(xlabel=r'$\Delta \alpha\,\cos\delta$ [mas]', ylabel=r'$\Delta \delta$ [mas]',title='Photocentre Orbit '+str(j+1))

    #along-scan residuals plot
    resax.errorbar(t,wws,yerr=ws_err,fmt='.',c='grey',alpha=0.8,zorder=2)
    resax.scatter(t,wws,c=t,cmap=colormap,zorder=3)
    resax.axhline(0,c='grey',zorder=1)
    resax.set(xlabel='Time ('+time_label+')',ylabel='Along-scan residuals (mas)')


def corner_astrometric_solution(res, star_mass=1.0, adda=False, **kwargs):
    if res.model != MODELS.GAIAmodel:
        print('Model is not GAIAmodel! '
              'corner_astrometric_solution() doing nothing...')
        return

    import pygtc

    labels = ['$d_a$', '$d_d$', r'$\mu_a$', r'$\mu_d$', r'$\pi$']
    data = res.posterior_sample[:, res.indices['astrometric_solution']]

    fig = pygtc.plotGTC(
        chains=data,
        # smoothingKernel=0,
        paramNames=labels,
        plotDensity=False,
        labelRotation=(True, True),
        filledPlots=False,
        colorsOrder=['blues_old'],
        figureSize='AandA_page',  # AandA_column
    )
    axs = fig.axes
    for ax in axs:
        ax.yaxis.set_label_coords(-0.3, 0.5, transform=None)
        ax.xaxis.set_label_coords(0.5, -0.4, transform=None)

    fs = axs[-1].xaxis.label.get_fontsize()
    start = len(labels)
    for i in range(-start, 0):
        val = percentile68_ranges_latex(data.T[i])
        axs[i].set_title(f'{labels[i]} = {val}', fontsize=fs - 1)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25, hspace=0.15)
    return fig


def plot_hgpm(res, pm_data, ncurves=50, normalize=False,
              include_planets=None, **kwargs):
    if res.model != MODELS.RVHGPMmodel:
        print('Model is not RVHGPMmodel! plot_hgpm() doing nothing...')
        return
    from ..kepler import keplerian_rvpm

    t_ra = np.linspace(pm_data.epoch_ra_hip-1000, pm_data.epoch_ra_gaia+1000, 500)
    t_dec = np.linspace(pm_data.epoch_dec_hip-1000, pm_data.epoch_dec_gaia+1000, 500)
    t_pm = np.empty((t_ra.size + t_dec.size,), dtype=t_ra.dtype)
    t_pm[0::2] = t_ra
    t_pm[1::2] = t_dec

    if normalize:
        p = res.maximum_likelihood_sample(printit=False)
        pm_ra_bary = p[res.indices['pm_ra_bary']]
        pm_dec_bary = p[res.indices['pm_dec_bary']]
        fig, axs = pm_data.plot(pm_ra_bary=pm_ra_bary, pm_dec_bary=pm_dec_bary)
    else:
        fig, axs = pm_data.plot()

    fig.set_size_inches(10, 4)

    if include_planets is None:
        include_planets = np.arange(res.max_components)

    ncurves = min(ncurves, res.ESS)

    # handles, labels = axs[0].get_legend_handles_labels()
    # print(handles)

    for i in np.random.choice(np.arange(res.ESS), size=ncurves, replace=False):
    # for i in range(res.ESS):
        p = res.posterior_sample[i]

        model_ra = np.zeros_like(t_ra)
        model_dec = np.zeros_like(t_dec)

        for j in range(int(p[res.indices['np']])):
            if j not in include_planets:
                continue
            model = keplerian_rvpm(
                res.data.t, t_pm, 
                p[res.indices['parallax']], 
                p[res.indices['planets.P']][j], 
                p[res.indices['planets.K']][j],
                p[res.indices['planets.e']][j],
                p[res.indices['planets.w']][j],
                p[res.indices['planets.φ']][j],
                res.M0_epoch, 
                p[res.indices['planets.i']][j],
                p[res.indices['planets.W']][j]
            )
            model_ra += model[1]
            model_dec += model[2]

        pm_ra_bary = p[res.indices['pm_ra_bary']]
        pm_dec_bary = p[res.indices['pm_dec_bary']]


        kw = dict(color='k', alpha=0.1 if ncurves > 10 else 1.0, zorder=-1, lw=0.5)
        kw_line = dict(ls='--', color='tomato', alpha=0.1)

        if normalize:
            axs[0].plot(t_ra - 5e4, model_ra + pm_ra_bary, **kw)
            axs[2].plot(t_dec - 5e4, model_dec + pm_dec_bary, **kw)
        else:
            axs[0].axhline(pm_ra_bary, **kw_line)
            axs[2].axhline(pm_dec_bary, **kw_line)
            axs[0].plot(t_ra - 5e4, pm_ra_bary + model_ra, **kw)
            axs[2].plot(t_dec - 5e4, pm_dec_bary + model_dec, **kw)
    
    for ax in axs[::2]:
        ax.set_xlim(-4000, 10_000)
        # ax.set_ylim(-40, 40)
    
    return fig


def hist_bary(res, show_prior=False):
    if res.model != MODELS.RVHGPMmodel:
        print('Model is not RVHGPMmodel! hist_pm_bary() doing nothing...')
        return

    units = 2 * [" [mas/yr]"] + [" [mas]"]
    hist_kw = dict(bins='doane', density=True)
    hist_prior_kw = dict(**hist_kw, alpha=0.15, color='k', zorder=-1)

    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(10, 4))

    estimate = percentile68_ranges_latex(res.posteriors.pm_ra_bary)
    axs[0].hist(res.posteriors.pm_ra_bary, label=estimate + units[0], **hist_kw)

    estimate = percentile68_ranges_latex(res.posteriors.pm_dec_bary)
    axs[1].hist(res.posteriors.pm_dec_bary, label=estimate + units[0], **hist_kw)

    estimate = percentile68_ranges_latex(res.posteriors.plx)
    axs[2].hist(res.posteriors.plx, label=estimate + units[1], **hist_kw)

    if show_prior:
        prior = res.priors['pm_ra_bary_prior']
        axs[0].hist(distribution_rvs(prior, size=res.ESS), label='prior', **hist_prior_kw)
        prior = res.priors['pm_dec_bary_prior']
        axs[1].hist(distribution_rvs(prior, size=res.ESS), label='prior', **hist_prior_kw)

    for ax in axs:
        ax.legend()
        ax.set_ylabel('posterior')
    axs[0].set(xlabel=r'$\mu_{\mathrm{RA}}$' + units[0])
    axs[1].set(xlabel=r'$\mu_{\mathrm{Dec}}$' + units[0])
    axs[2].set(xlabel=r'$\pi$' + units[1])

    return fig, axs


def plot_random_samples(res, ncurves=50, samples=None, tt=None, over=0.1, ntt=5000,
                        subtract_offsets=False, clip_curves_to_data=False,
                        show_vsys=False, show_gp=True, isolate_known_object=True, isolate_transiting_planet=True,
                        isolate_apodized_keplerians=True, include_jitters_in_points=False, 
                        include_jitters_in_predict=True, full_plot=False, show_outliers=False, **kwargs):
    """
    Display the RV data together with curves from the posterior predictive. 

    Args:
        ncurves (int, optional):
            Number of posterior predictive curves to show.
        samples (array, optional):
            Specific posterior sample(s) to plot.
        tt (array, optional):
            Time grid for the plots. By default uses `res._get_tt` or
            `res._get_ttGP` depending on the model.
        over (float, optional):
            Curves are calculated covering 100*(1 + `over`)% of the timespan of
            the data.
        ntt (int, optional):
            Number of points for the time grid.
        subtract_offsets (bool, optional):
            Subtract the RV offsets from the data. Only used when `ncurves = 1`.
        clip_curves_to_data (bool, optional):
            Clip the curves to the time span of the data.
        show_vsys (bool, optional):
            Show the systemic velocity for each sample.
        show_gp (bool, optional):
            Show the GP prediction for each sample.
        isolate_known_object (bool, optional):
            Show the Keplerian curves for the known object(s), if present in the
            model.
        isolate_transiting_planet (bool, optional):
            Show the Keplerian curves for the transiting planet(s), if present
            in the model.
        include_jitters_in_points (bool, optional):
            Include an extra error bar for each point which takes into account
            the jitter(s). This only works when ncurves=1.
        include_jitters_in_predict (bool, optional):
            Include the jitters in the GP prediction (if model has a GP). This
            should almost always be True!
        full_plot (bool, optional):
            If True (and ncurves=1), adds panels with the residuals from the
            given sample and their GLS periodogram.
        show_outliers (bool, optional):
            Highlight the outliers (if likelihood is Student-t).

    Returns:
        fig (matplotlib.figure.Figure):
            The figure with the plot
    """

    SB2 = res.model is MODELS.BINARIESmodel and res.double_lined

    if samples is None:
        samples = res.posterior_sample.copy()
        samples_provided = False
    else:
        samples = np.atleast_2d(samples)
        samples_provided = True

    mask = np.ones(samples.shape[0], dtype=bool)

    t = res.data.t.copy()
    M0_epoch = copy(res.M0_epoch)

    if t[0] > 24e5:
        t -= 24e5
        M0_epoch -= 24e5

    if tt is None:
        tt = res._get_tt(ntt, over)
        if res.has_gp:
            tt = res._get_ttGP(ntt, over)

    if t.size > 100:
        ncurves = min(10, ncurves)

    ncurves = min(ncurves, samples.shape[0])

    if full_plot and ncurves > 1:
        print('full_plot can only be used when ncurves=1')
        full_plot = False
    
    if subtract_offsets and ncurves > 1:
        print('subtract_offsets can only be used when ncurves=1')
        subtract_offsets = False

    if samples.shape[0] == 1:
        ii = np.zeros(1, dtype=int)
    elif ncurves == samples.shape[0] or samples_provided:
        # ii = np.arange(ncurves)
        ii = np.random.choice(np.arange(samples.shape[0]), size=ncurves, replace=False)
    else:
        try:
            # select `ncurves` indices from the 70% highest likelihood samples
            lnlike = res.posterior_lnlike[:, 1]
            sorted_lnlike = np.sort(lnlike)[::-1]
            mask_lnlike = lnlike > np.percentile(sorted_lnlike, 70)
            ii = np.random.choice(np.where(mask & mask_lnlike)[0], size=ncurves,
                                  replace=False)
        except ValueError:
            ii = np.random.choice(np.arange(samples.shape[0]), size=ncurves, replace=False)

    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
        fig = ax.figure
    else:
        if full_plot:
            fig, axs = plt.subplot_mosaic('aac\naac\nbbc', constrained_layout=True,
                                           figsize=(8, 5))
            ax = axs['a']
            axs['b'].sharex(ax)
        else:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)

    cc = kwargs.pop('curve_color', 'k')
    gpc = kwargs.pop('gp_color', 'plum')
    actc = kwargs.pop('act_color', 'tomato')

    if res.multi and subtract_offsets:
        # provide offsets and 0 to plot_data: the data will appear around vsys
        kwargs['offsets'] = np.r_[
            samples[0][res.indices['inst_offsets']],
            0.0
        ]

    _, y_offset = plot_data(res, ax, **kwargs)

    if SB2:
        kwargs2 = kwargs.copy()
        kwargs2['mfc'] = 'none'
        plt.gca().set_prop_cycle(None)
        _, y_offset_sec = plot_data(res, ax, secondary_star=True, **kwargs2)

    if include_jitters_in_points:
        if ncurves == 1:
            _ = plot_data_jitters(res, samples[0], ax=ax, ms=3, legend=False)
        else:
            import warnings
            warnings.warn('include_jitters_in_points can only be used when ncurves=1')

    if show_outliers:
        if res.studentt:
            outliers = find_outliers(res, res.maximum_likelihood_sample(printit=False))
            if outliers.any():
                mi = res.data.y[~outliers].min() - res.data.e.max()
                ma = res.data.y[~outliers].max() + res.data.e.max()
                yclip = np.clip(res.data.y, mi, ma)
                ax.plot(res.data.t[outliers], yclip[outliers], 'rs')
                ax.set_ylim(mi, ma)
        else:
            print('cannot identify outliers, likelihood is not Student t')


    # plot the Keplerian curves
    alpha = kwargs.pop('alpha', 0.2 if ncurves > 1 else 0.8)

    if clip_curves_to_data:
        tt_plot = np.tile(tt, (res.n_instruments, 1)).T
        from .utils import Interval
        for i in range(res.n_instruments):
            time_mask = res.data.t[res.data.obs == i + 1]
            m = Interval.from_array(time_mask).mask(tt_plot[:, i])
            tt_plot[~m, i] = np.nan
    else:
        tt_plot = tt

    for icurve, i in enumerate(ii):
        sample = samples[i]

        if ncurves == 1:
            stoc_model, stoc_std = res.stochastic_model(sample, tt, return_std=True,
                                                        include_jitters=include_jitters_in_predict)
        else:
            stoc_model = res.stochastic_model(sample, tt, include_jitters=include_jitters_in_predict)
        stoc_model = np.atleast_2d(stoc_model)
        model = np.atleast_2d(res.eval_model(sample, tt))
        offset_model = res.eval_model(sample, tt, include_planets=False)

        if res.multi and not subtract_offsets:
            model = res.burst_model(sample, tt, model)
            offset_model = res.burst_model(sample, tt, offset_model)

        ax.set_prop_cycle(None)
        # if model.shape[0] == 1:
        #     color = cc
        # else:
        #     color = None

        ax.plot(tt_plot, (stoc_model + model).T - y_offset,
                color=cc, alpha=alpha, zorder=-1)

        if res.has_gp and show_gp:
            m = (stoc_model + offset_model).T - y_offset
            ax.plot(tt, m, color=gpc, alpha=alpha)

            # plot the predictive std if there's only one curve
            if ncurves == 1:
                m = m.squeeze()
                ax.fill_between(tt, m - stoc_std, m + stoc_std, 
                                color=gpc, alpha=0.2*alpha)

        if hasattr(res, 'indicator_correlations') and res.indicator_correlations:
            model_wo_ind = res.eval_model(sample, tt,
                                          include_indicator_correlations=False)
            curve = (model - model_wo_ind + offset_model).T - y_offset
            ax.plot(tt, curve, color=actc, alpha=alpha)

        if show_vsys:
            kw = dict(alpha=alpha, color='r', ls='--')
            if res.multi:
                for j in range(res.n_instruments):
                    instrument_mask = res.data.obs == j + 1
                    start = t[instrument_mask].min()
                    end = t[instrument_mask].max()
                    m = np.where((tt > start) & (tt < end))
                    ax.plot(tt[m], offset_model[j][m] - y_offset, **kw)
            else:
                ax.plot(tt, offset_model - y_offset, **kw)

        pj = 0
        if res.KO and isolate_known_object:
            for _ in range(1, res.nKO + 1):
                if SB2:
                    kepKO,kepKO_sec = res.eval_model(sample, tt, single_planet=-(pj+1))
                    ax.plot(tt, kepKO - y_offset, alpha=alpha,
                            label='known object' if icurve == 0 else None)
                    ax.plot(tt, kepKO_sec - y_offset, alpha=alpha,
                            label='known object' if icurve == 0 else None)
                else:
                    kepKO = res.eval_model(sample, tt, single_planet=-(pj+1))
                    ax.plot(tt, kepKO - y_offset, alpha=alpha,
                            label='known object' if icurve == 0 else None)
                pj += 1

        if hasattr(res, 'TR') and res.TR and isolate_transiting_planet:
            for _ in range(1, res.nTR + 1):
                kepTR = res.eval_model(sample, tt, single_planet=-(pj+1))
                ax.plot(tt, kepTR - y_offset, alpha=alpha,
                        label='transiting planet' if icurve == 0 else None)
                pj += 1

    if kwargs.get('legend', True):
        ax.legend()

    if full_plot:
        r = res.residuals(sample, full=True)
        plot_data(res, ax=axs['b'], y=r, e=res.data.e, legend=False, show_rms=True)
        axs['b'].axhline(y=0, ls='--', color='k', alpha=0.5)
        gls = LombScargle(res.data.t, r, res.data.e)
        f, p = gls.autopower(samples_per_peak=15)
        axs['c'].semilogy(p, 1 / f, color='k', alpha=0.8)
        axs['c'].invert_xaxis()
        fap001 = gls.false_alarm_level(0.01)
        axs['c'].axvline(fap001, ls='--', alpha=0.2)

    if res.save_plots:
        filename = 'kima-showresults-fig6.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig


def plot_random_samples_multiseries(res, ncurves=50, samples=None, over=0.1, ntt=10000, 
                                    show_vsys=False, isolate_known_object=True,
                                    include_jitters_in_points=False, include_jitters_in_predict=True, 
                                    just_rvs=False, full_plot=False, highest_likelihood=False, **kwargs):
    """
    Display the RV data together with curves from the posterior predictive.
    A total of `ncurves` random samples are chosen, and the Keplerian 
    curves are calculated covering 100 + `over`% of the data timespan.
    If the model has a GP component, the prediction is calculated using the
    GP hyperparameters for each of the random samples.
    """
    full_plot = kwargs.pop('full_plot', False)
    rhk = res.model is MODELS.RVFWHMRHKmodel


    if samples is None:
        samples = res.posterior_sample
    else:
        samples = np.atleast_2d(samples)

    t = res.data.t.copy()
    M0_epoch = res.M0_epoch
    if t[0] > 24e5:
        t -= 24e5
        M0_epoch -= 24e5

    tt = np.linspace(t.min() - over * np.ptp(t), t.max() + over * np.ptp(t), ntt + int(100 * over))

    if res.has_gp:
        # let's be more reasonable for the number of GP prediction points
        #! OLD: linearly spaced points (lots of useless points within gaps)
        #! ttGP = np.linspace(t[0], t[-1], 1000 + t.size*3)
        #! NEW: have more points near where there is data
        kde = gaussian_kde(t)
        ttGP = kde.resample(25000 + t.size * 3).reshape(-1)
        # constrain ttGP within observed times, to not waste
        ttGP = (ttGP + t[0]) % np.ptp(t) + t[0]
        ttGP = np.r_[ttGP, t]
        ttGP.sort()  # in-place

        # if t.size > 100:
        #     ncurves = min(10, ncurves)

    y = res.data.y.copy()
    yerr = res.data.e.copy()

    y2 = res.data.y2.copy()
    y2err = res.data.e2.copy()

    if rhk:
        y3 = res.data.y3.copy()
        y3err = res.data.e3.copy()

    ncurves = min(ncurves, samples.shape[0])

    if samples.shape[0] == 1:
        ii = np.zeros(1, dtype=int)
    elif ncurves == samples.shape[0]:
        ii = np.arange(ncurves)
    else:
        if highest_likelihood:
            i = np.argsort(res.posterior_lnlike[:, 1])
            ii = i[-ncurves:]
        else:
            ii = np.random.choice(np.arange(samples.shape[0]), ncurves)

        # select `ncurves` indices from the 70% highest likelihood samples
        lnlike = res.posterior_lnlike[:, 1]
        sorted_lnlike = np.sort(lnlike)[::-1]
        mask_lnlike = lnlike > np.percentile(sorted_lnlike, 70)
        # ii = np.random.choice(np.where(mask)[0], ncurves)

    ax1, ax2, ax3 = None, None, None
    if just_rvs:
        if 'ax' in kwargs:
            ax1 = kwargs.pop('ax')
            fig = ax1.figure
    else:
        if 'ax1' in kwargs and 'ax2' in kwargs:
            ax1, ax2 = kwargs.pop('ax1'), kwargs.pop('ax2')
            fig = ax1.figure
        if rhk:
            if 'ax3' in kwargs:
                ax3 = kwargs.pop('ax3')

    if ax1 is None and ax2 is None and ax3 is None:
        fig = plt.figure(constrained_layout=True, figsize=(10, 8))
        if full_plot:
            width_ratios = [3, 1, 1]
            gs = plt.GridSpec(2, 3, width_ratios=width_ratios, figure=fig)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
            ax1p = fig.add_subplot(gs[0, 1])
            ax1r = fig.add_subplot(gs[0, 2])
            ax2p = fig.add_subplot(gs[1, 1])
            ax2r = fig.add_subplot(gs[1, 2])
            # ax1r, ax2r = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[3, 0])
        else:
            if just_rvs:
                ax1 = fig.add_subplot(1, 1, 1)
            else:
                if rhk:
                    gs = plt.GridSpec(3, 1, figure=fig)
                    ax1 = fig.add_subplot(gs[0])
                    ax2 = fig.add_subplot(gs[1], sharex=ax1)
                    ax3 = fig.add_subplot(gs[2], sharex=ax1)
                else:
                    gs = plt.GridSpec(2, 1, figure=fig)
                    ax1 = fig.add_subplot(gs[0])
                    ax2 = fig.add_subplot(gs[1], sharex=ax1)
            # ax1r, ax2r = fig.add_subplot(gs[1]), fig.add_subplot(gs[3])

    if just_rvs:
        _, y_offset = plot_data(res, ax=ax1, ms=3, legend=False, ignore_y2=True)
    else:
        if rhk:
            _, _, _, y_offset, y2_offset, y3_offset = plot_data(res, ax=ax1, axf=ax2, axr=ax3, ms=3, legend=False)
        else:
            _, _, y_offset, y2_offset = plot_data(res, ax=ax1, axf=ax2, ms=3, legend=False)


    if include_jitters_in_points:
        if ncurves == 1:
            if just_rvs:
                _ = plot_data_jitters(res, samples[0], ax=ax1, ms=3, legend=False, ignore_y2=True)
            else:
                if rhk:
                    _ = plot_data_jitters(res, samples[0], ax=ax1, axf=ax2, axr=ax3, ms=3, legend=False)
                else:
                    _ = plot_data_jitters(res, samples[0], ax=ax1, axf=ax2, ms=3, legend=False)
        else:
            import warnings
            warnings.warn('include_jitters_in_points in the data points can only be used when ncurves=1')

    ## plot the Keplerian curves
    alpha = 0.2 if ncurves > 1 else 1

    try:
        overlap = res._time_overlaps[0]
    except ValueError:
        overlap = False

    for icurve, i in enumerate(ii):
        # just the GP, centered around 0
        # for models without GP, stoc_model will be full of zeros
        stoc_model = res.stochastic_model(samples[i], tt, 
                                          include_jitters=include_jitters_in_predict)
        # the model, including planets, systemic RV/FWHM, and offsets
        model = res.eval_model(samples[i], tt)
        # burst the model if there are multiple instruments
        model = res.burst_model(samples[i], tt, model)

        kw = dict(color='k', alpha=alpha, zorder=-2)
        if overlap:
            v = stoc_model[0] + model[::2] - y_offset
            ax1.plot(tt, v.T, **kw)
        else:
            ax1.plot(tt, stoc_model[0] + model[0] - y_offset, **kw)
            # ax2.plot(tt, stoc_model[1] + model[1], 'k', alpha=alpha)

        # the model without planets, just systemic RV/FWHM and offsets
        offset_model = res.eval_model(samples[i], tt, include_planets=False)
        # burst the model if there are multiple instruments
        offset_model = res.burst_model(samples[i], tt, offset_model)

        if res.KO and isolate_known_object:
            for iko in range(res.nKO):
                KOpl = res.eval_model(samples[i], tt,
                                      single_planet=-iko - 1)[0]
                ax1.plot(tt, KOpl - y_offset + (iko + 1) * np.ptp(res.data.y),
                         color='g', alpha=alpha)

        if res.has_gp:
            kw = dict(color='plum', alpha=alpha, zorder=1)#, ls='--')
            if overlap:
                v = stoc_model[0] + offset_model[::2] - y_offset
                ax1.plot(tt, v.T, **kw)
                if not just_rvs:
                    f = stoc_model[1] + offset_model[1::2] - y2_offset
                    ax2.plot(tt, f.T, **kw)
                if rhk:
                    pass
            else:
                ax1.plot(tt, stoc_model[0] + offset_model[0] - y_offset, **kw)
                if not just_rvs:
                    ax2.plot(tt, stoc_model[1] + offset_model[1] - y2_offset, **kw)
                    if rhk:
                        ax3.plot(tt, stoc_model[2] + offset_model[2] - y3_offset, **kw)

        if show_vsys:
            kw = dict(alpha=0.1, color='r', ls='--')
            if res.multi:
                for j in range(res.n_instruments):
                    instrument_mask = res.data.obs == j + 1
                    start = t[instrument_mask].min()
                    end = t[instrument_mask].max()
                    m = np.where( (tt > start) & (tt < end) )
                    ax1.plot(tt[m], offset_model[0][m] - y_offset, **kw)
                    ax2.plot(tt[m], offset_model[1][m] - y2_offset, **kw)
            else:
                ax1.plot(tt, offset_model[0] - y_offset, **kw)
                ax2.plot(tt, offset_model[1] - y2_offset, **kw)

        if full_plot:
            from gatspy.periodic import LombScargleMultiband
            from astropy.timeseries import LombScargle
            r = res.residuals(samples[i], full=True)
            freq = LombScargle(res.data.t, r[0], res.data.e).autofrequency()

            kwl = dict(color='k', alpha=0.2, ls='--')
            for i, ax in enumerate((ax1r, ax2r)):
                # option 1
                # gls = LombScargle(res.t, r[i], res.data.e)
                # freq, power = gls.autopower()
                # option 2
                gls = LombScargleMultiband(Nterms_base=1, Nterms_band=0)
                gls.fit(res.data.t, r[i], (res.data.e, res.data.e2)[i],
                        filts=res.data.obs)
                power = gls.periodogram(1 / freq)
                ax.semilogy(power, 1 / freq, 'k', alpha=alpha)

                gls = LombScargle(res.data.t, r[i] - gls.ymean_,
                                  (res.data.e, res.data.e2)[i])
                fap001 = gls.false_alarm_level(0.01)
                ax.axvline(fap001, **kwl)
                # kwt = dict(color='k', alpha=0.3, rotation=90, ha='left', va='top', fontsize=8)
                # fap001 = gls.false_alarm_level(0.01)
                # ax.axvline(fap001, **kwl)
                # ax.text(0.98*fap001, 1/freq.min(), '1%', **kwt)

                # fap01 = gls.false_alarm_level(0.1)
                # ax.axvline(fap01, **kwl)
                # ax.text(0.98*fap01, 1/freq.min(), '10%', **kwt)

                # ax.set(xlabel='residual power', ylabel='Period [days]')
                # ax.invert_xaxis()
                # ax.yaxis.tick_right()
                # ax.yaxis.set_label_position('right')

    # ## plot the data
    # if res.multi:
    #     for j in range(res.inst_offsets.shape[1] // 2 + 1):
    #         inst = res.instruments[j]
    #         m = res.data.obs == j + 1

    #         kw = dict(fmt='o', ms=3, color=colors[j], label=inst)
    #         kw.update(**kwargs)
    #         ax1.errorbar(t[m], y[m] - y_offset, yerr[m], **kw)
    #         ax2.errorbar(t[m], res.y2[m] - y2_offset, res.e2[m], **kw)

    #     ax1.legend(loc='upper left', fontsize=8)

    # else:
    #     ax1.errorbar(t, y - y_offset, yerr, fmt='o')
    #     ax2.errorbar(t, y2 - y2_offset, res.e2, fmt='o')

    if full_plot:
        kwl = dict(color='k', alpha=0.2, ls='--')
        gls = LombScargleMultiband(Nterms_base=1, Nterms_band=0)
        gls.fit(res.data.t, res.data.y, res.data.e, filts=res.data.obs)
        power = gls.periodogram(1 / freq)
        ax1p.semilogy(power, 1 / freq, 'r', alpha=1)
        gls = LombScargle(res.data.t, res.data.y - gls.ymean_, res.data.e)
        # kwt = dict(color='k', alpha=0.3, rotation=90, ha='left', va='top', fontsize=8)
        fap001 = gls.false_alarm_level(0.01)
        ax1p.axvline(fap001, **kwl)
        # ax.text(0.98*fap001, 1/freq.min(), '1%', **kwt)

        gls = LombScargleMultiband(Nterms_base=1, Nterms_band=0)
        gls.fit(res.data.t, res.data.y2, res.data.e2, filts=res.data.obs)
        power = gls.periodogram(1 / freq)
        ax2p.semilogy(power, 1 / freq, 'r', alpha=1)
        gls = LombScargle(res.data.t, res.data.y2 - gls.ymean_, res.data.e2)
        fap001 = gls.false_alarm_level(0.01)
        ax2p.axvline(fap001, **kwl)


    if res.arbitrary_units:
        ylabel = 'Q [arbitrary]'
    else:
        ylabel = 'RV [m/s]'

    ax1.set(ylabel=ylabel, xlabel='Time [days]')
    # ax1r.set(ylabel='', xlabel='Time [days]')
    if not just_rvs:
        ax2.set(ylabel='FWHM [m/s]', xlabel='Time [days]')
    # ax2r.set(ylabel='', xlabel='Time [days]')
    # if full_plot:
    # ax2.set(xlabel='Time [days]', ylabel=f'FWHM [m/s]')

    if full_plot:
        ax1p.set(xlabel='Power', ylabel='Period [days]')
        ax2p.set(xlabel='Power', ylabel='Period [days]')
        ax1r.set(xlabel='Residual Power')
        ax2r.set(xlabel='Residual Power')
        for ax in (ax1p, ax1r, ax2p, ax2r):
            ax.invert_xaxis()

    # ax.yaxis.tick_right()
    # ax.yaxis.set_label_position('right')



    # fig.tight_layout()

    if res.save_plots:
        filename = 'kima-showresults-fig6.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig

    return fig, ax1, ax2


def plot_random_samples_transit(res, ncurves=50, samples=None, over=0.1,
                                show_vsys=False, ntt=5000,
                                isolate_known_object=True, full_plot=False,
                                **kwargs):

    import batman

    if samples is None:
        samples = res.posterior_sample
    else:
        samples = np.atleast_2d(samples)
    mask = np.ones(samples.shape[0], dtype=bool)

    t = res.data.t.copy()
    M0_epoch = res.M0_epoch
    if t[0] > 24e5:
        t -= 24e5
        M0_epoch -= 24e5

    tt = res._get_tt(ntt, over)
    if res.has_gp:
        ttGP = res._get_ttGP()

    # if t.size > 100:
    #     ncurves = min(10, ncurves)

    ncurves = min(ncurves, samples.shape[0])

    if samples.shape[0] == 1:
        ii = np.zeros(1, dtype=int)
    elif ncurves == samples.shape[0]:
        ii = np.arange(ncurves)
    else:
        try:
            # select `ncurves` indices from the 70% highest likelihood samples
            lnlike = res.posterior_lnlike[:, 1]
            sorted_lnlike = np.sort(lnlike)[::-1]
            mask_lnlike = lnlike > np.percentile(sorted_lnlike, 70)
            ii = np.random.choice(np.where(mask & mask_lnlike)[0], ncurves)
        except ValueError:
            ii = np.arange(ncurves)

    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
        fig = ax.figure
    else:
        fig, ax = plt.subplots(1, 1)

    _, y_offset = plot_transit_data(res, ax, **kwargs)

    ## plot the Keplerian curves
    alpha = 0.1 if ncurves > 1 else 1

    cc = kwargs.get('curve_color', 'k')
    gpc = kwargs.get('gp_color', 'plum')

    params = batman.TransitParams()
    params.inc = 90.                     #orbital inclination (in degrees)
    params.limb_dark = "quadratic"       #limb darkening model

    for icurve, i in enumerate(ii):
        sample = samples[i]
        params.u = sample[res.indices['u']]  # limb darkening coefficients [u1, u2]
        flux = np.full_like(tt, sample[-1])
        planet_pars = sample[res.indices['planets']]

        for j in range(res._mc):
            params.per = planet_pars[j::res._mc][0]                      #orbital period
            params.rp = planet_pars[j::res._mc][1]                       #planet radius (in units of stellar radii)
            params.a = planet_pars[j::res._mc][2]                        #semi-major axis (in units of stellar radii)
            params.ecc = planet_pars[j::res._mc][3]                      #eccentricity
            params.w = planet_pars[j::res._mc][4]            #eccentricity
            params.t0 = planet_pars[j::res._mc][5]                       #time of inferior conjunction

            m = batman.TransitModel(params, tt)    #initializes model
            flux += -1.0 + m.light_curve(params)          #calculates light curve

        #     stoc_model = np.atleast_2d(res.stochastic_model(sample, tt))
        #     model = np.atleast_2d(res.eval_model(sample, tt))
        #     offset_model = res.eval_model(sample, tt, include_planets=False)

        #     if res.multi:
        #         model = res.burst_model(sample, tt, model)
        #         offset_model = res.burst_model(sample, tt, offset_model)

        ax.plot(tt, flux - y_offset, color=cc, alpha=alpha, zorder=10)
        #     if res.has_gp:
        #         ax.plot(tt, (stoc_model + offset_model).T - y_offset, color=gpc,
        #                 alpha=alpha)

        #     if show_vsys:
        #         kw = dict(alpha=alpha, color='r', ls='--')
        #         if res.multi:
        #             for j in range(res.n_instruments):
        #                 instrument_mask = res.data.obs == j + 1
        #                 start = t[instrument_mask].min()
        #                 end = t[instrument_mask].max()
        #                 m = np.where((tt > start) & (tt < end))
        #                 ax.plot(tt[m], offset_model[m] - y_offset, **kw)
        #         else:
        #             ax.plot(tt, offset_model - y_offset, **kw)

        if res.KO and isolate_known_object:
            # flux = np.full_like(tt, sample[-1])
            KO_pars = sample[res.indices['KOpars']]
            for j in range(0, res.nKO):
                print(KO_pars[j::res.nKO])
                params.per = KO_pars[j::res.nKO][0]                      #orbital period
                params.rp = KO_pars[j::res.nKO][1]                       #planet radius (in units of stellar radii)
                params.a = KO_pars[j::res.nKO][2]                        #semi-major axis (in units of stellar radii)
                params.ecc = KO_pars[j::res.nKO][3]                      #eccentricity
                params.w = KO_pars[j::res.nKO][4]        #longitude of periastron (in degrees)
                params.t0 = KO_pars[j::res.nKO][5]                       #time of inferior conjunction

                m = batman.TransitModel(params, tt)    #initializes model
                flux = m.light_curve(params)          #calculates light curve
                # kepKO = res.eval_model(res.posterior_sample[i], tt,
                #                        single_planet=-k)
                ax.plot(tt, flux - y_offset, color=f'C{j}', alpha=alpha)

    if res.save_plots:
        filename = 'kima-showresults-fig6.png'
        print('saving in', filename)
        fig.savefig(filename)

    if res.return_figs:
        return fig


def orbit(res, sample=None, n=10, star_mass=1.0, sortP=False):
    from analysis import get_planet_mass
    from utils import mjup2msun
    import rebound

    if sample is None:
        if sortP:
            sample = res.get_sorted_planet_samples(full=True)
            ind = np.random.choice(np.arange(sample.shape[0]), n)
            pp = sample[ind]
        else:
            pp = res.posterior_sample[:n]
    else:
        pp = np.atleast_2d(sample)

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111, aspect="equal")

    for p in pp:
        sim = rebound.Simulation()
        sim.G = 0.00029591  # AU^3 / solMass / day^2
        sim.add(m=star_mass)

        nplanets = int(p[res.index_component])
        pars = p[res.indices['planets']]
        for i in range(nplanets):
            P, K, φ, ecc, w = pars[i::res.max_components]
            # print(P, K, φ, ecc, w)
            m = get_planet_mass(P, K, ecc, star_mass=star_mass)[0]
            m *= mjup2msun
            sim.add(P=P, m=m, e=ecc, omega=w, M=φ, inc=0)
            # res.move_to_com()

        if res.KO:
            pars = p[res.indices['KOpars']]
            for i in range(res.nKO):
                P, K, φ, ecc, w = pars[i::res.nKO]
                # print(P, K, φ, ecc, w)
                m = get_planet_mass(P, K, ecc, star_mass=star_mass)[0]
                m *= mjup2msun
                sim.add(P=P, m=m, e=ecc, omega=w, M=φ, inc=0)
                # res.move_to_com()

        kw = dict(
            fig=fig,
            color=True,
            show_particles=False,
        )
        rebound.plotting.OrbitPlot(sim, **kw)

    if len(pp) == 1:
        return sim


def simulation(results, sample):
    res, p = results, sample
    star_mass = 0.12
    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.ri_whfast.safe_mode = 0
    sim.dt = 1
    sim.G = 0.00029591  # AU^3 / solMass / day^2
    sim.add(m=star_mass)

    # periods = []
    # eccentricities = []

    nplanets = int(p[res.index_component])
    pars = p[res.indices['planets']]
    for i in range(nplanets):
        P, K, φ, ecc, w = pars[i::res.max_components]
        # print(P, K, φ, ecc, w)
        m = get_planet_mass(P, K, ecc, star_mass=star_mass)[0]
        m *= mjup2msun
        sim.add(P=P, m=m, e=ecc, omega=w, M=φ, inc=0)

        # periods.append(P)
        # eccentricities.append(ecc)

        # res.move_to_com()

    if res.KO:
        pars = p[res.indices['KOpars']]
        for i in range(res.nKO):
            P, K, φ, ecc, w = pars[i::res.nKO]
            # print(P, K, φ, ecc, w)
            m = get_planet_mass(P, K, ecc, star_mass=star_mass)[0]
            m *= mjup2msun
            sim.add(P=P, m=m, e=ecc, omega=w, M=φ, inc=0)
            # periods.append(P)
            # eccentricities.append(ecc)

    sim.move_to_com()
    sim.init_megno()
    sim.exit_max_distance = 20.
    try:
        # integrate for 100 years, integrating to the nearest timestep for
        # each output to keep the timestep constant and preserve WHFast's
        # symplectic nature
        sim.integrate(100 * 365, exact_finish_time=0)
        megno = sim.calculate_megno()
        return megno
        # ax.semilogx(periods, eccentricities, 'g.')
    except rebound.Escape:
        # At least one particle got ejected, returning large MEGNO.
        return 10
        # ax.semilogx(periods, eccentricities, 'rx')


from .analysis import get_planet_mass
import rebound


def megno(res, star_mass=1.0, samples=None):

    if samples is None:
        samples = res.posterior_sample

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111) #, aspect="equal")



    from rebound.interruptible_pool import InterruptiblePool
    pool = InterruptiblePool()
    results = pool.map(simulation, samples)
    return results
    # return np.array(M)


def plot_parameter_samples(res):
    sample = res.sample.copy()

    _i = res.indices['vsys'] - 1
    sample = np.delete(sample, _i, axis=1)
    _i = res.indices['np'] - 1, res.indices['np'] - 2
    sample = np.delete(sample, _i, axis=1)

    sample = np.c_[res.sample_info[:, 1], sample]

    s = np.argsort(res.sample_info[:, 1])
    sample = sample[s, :]

    fig, axs = plt.subplots(sample.shape[1], constrained_layout=True,
                            figsize=(6, 12), sharex=True)

    for p, ax in zip(sample.T, axs):
        ax.plot(p)
    # axs[0].set(yscale='symlog')
    axs[-1].set(xlabel='sample')
    return fig, axs


def interactive_plotter(res):

    def on_pick(event):
        artist = event.artist
        ind = event.ind
        # xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
        # x, y = artist.get_xdata(), artist.get_ydata()
        # print('Artist picked:', event.artist)
        # print('{} vertices picked'.format(len(ind)))
        # print('Pick between vertices {} and {}'.format(min(ind), max(ind)+1))
        # print('x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse))
        # print('Data point:', x[ind[0]], y[ind[0]])
        # print()
        i = ind[0]
        print(i)
        res.print_sample(res.posterior_sample[i])
        print()
        res.phase_plot(res.posterior_sample[i])
        p, k, e = res.posteriors.P[ind[0]], res.posteriors.K[ind[0]], res.posteriors.e[ind[0]]
        k, e, p = k[np.argsort(p)], e[np.argsort(p)], p[np.argsort(p)]
        line1.set_data(p, k)
        line2.set_data(p, e)
        #print(line in axs[0].lines)
        fig.canvas.draw()

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    line1, = axs[0].semilogx([], [], '-ok')
    line2, = axs[1].semilogx([], [], '-ok')
    axs[0].semilogx(res.posteriors.P, res.posteriors.K, '.', ms=1, picker=5)
    axs[1].semilogx(res.posteriors.P, res.posteriors.e, '.', ms=1)
    fig.canvas.callbacks.connect('pick_event', on_pick)


def report(res):
    from .utils import distribution_support

    short_instruments = [
        i.replace('ESPRESSO', 'E').replace('HARPS', 'H').replace('CORALIE', 'C')
        for i in res.instruments
    ]

    fig, axs = plt.subplot_mosaic('aaab\npppt\nccco\nddde', figsize=(8.3, 11.7),
                                  constrained_layout=True)

    res.plot_random_samples(ax=axs['a'], ncurves=20,
                            clip_curves_to_data=True)
    axs['a'].set(title='')
    axs['a'].legend(ncol=np.ceil(res.n_instruments / 2), fontsize=8,
                    loc='lower right', bbox_to_anchor=(1.0, 1.0))

    if res.fix:
        axs['b'].axis('off')
    else:
        res.plot1(ax=axs['b'], show_ESS=False, verbose=False)
        axs['b'].set(ylabel='posterior', title='', yticks=[])


    if res.max_components == 0 and not res.KO and not res.TR:
        axs['p'].axis('off')
        axs['c'].axis('off')
        axs['d'].axis('off')
    else:
        res.plot2(ax=axs['p'], alpha=0.6)
        axs['p'].set(title='', ylabel='posterior', xlabel='')

        from .analysis import FIP
        f_width = 1 / res.priors['Pprior'].upper
        ax_ = axs['p'].twinx()
        FIP(res, f_width=f_width, ax=ax_, just_tip=True, show_ESS=False,
            color='k', alpha=0.4)
        
        axs['p'].set_ylim(ax_.get_ylim())

        res.plot3(ax1=axs['c'], ax2=axs['d'])
        if res.ESS < 1000:
            for line in axs['c'].get_lines():
                line.set_alpha(1.0)
            for line in axs['d'].get_lines():
                line.set_alpha(1.0)

        for ax in (axs['c'], axs['d']):
            ax.set_title('')
            ax.sharex(axs['p'])
        # axs['c'].set_ylim(distribution_support(res.priors['Kprior']))
        axs['d'].set_ylim(0, 1)

    # vsys, offsets violin plots
    if res.n_instruments > 1:
        axs['o'].violinplot(res.posteriors.offset, showmedians=True,
                            showextrema=False, vert=False)
        axs['o'].xaxis.minorticks_on()
        axs['o'].set_xlabel('offsets [m/s]')
        axs['o'].yaxis.tick_right()
        axs['o'].set_yticks(range(1, res.n_instruments))
        axs["o"].set_title(f"relative to {short_instruments[-1]}", loc="left",
                        fontsize=10)
        labels = short_instruments[:-1]
        axs['o'].set_yticklabels(labels)
        estimates = [percentile68_ranges_latex(o) for o in res.posteriors.offset.T]
        xlim = axs['o'].get_xlim()
        axs['o'].margins(x=0.4, tight=False)
        axs['o'].set_xlim(xlim[0], None)
        xlim = axs['o'].get_xlim()
        for i, est in enumerate(estimates):
            axs["o"].text(xlim[1] - 0.05*np.ptp(xlim), i + 1, est, 
                          ha="right", va="center", fontsize=10)
    else:
        axs['o'].axis('off')

    # jitter violin plots
    axs['e'].violinplot(res.jitter[:, 1:], showmedians=True,
                        showextrema=False, vert=False)
    axs['e'].xaxis.minorticks_on()
    axs['e'].set_xlabel('jitter [m/s]')
    axs['e'].yaxis.tick_right()
    axs['e'].set_yticks(range(1, len(res.instruments) + 1))
    labels = short_instruments
    axs['e'].set_yticklabels(labels)
    estimates = [percentile68_ranges_latex(j) for j in res.jitter[:, 1:].T]

    xlim = axs['e'].get_xlim()
    axs['e'].margins(x=0.4, tight=False)
    axs['e'].set_xlim(xlim[0], None)
    xlim = axs['e'].get_xlim()

    for i, est in enumerate(estimates, start=1):
        axs['e'].text(xlim[1] - 0.05*np.ptp(xlim), i, est, 
                      ha='right', va='center', fontsize=10, )

    axs['t'].axis('off')
    y = 0
    axs['t'].text(0, y, str(res.model).replace('MODELS.', '')); y -= 1
    axs['t'].text(0, y, f'logZ: {res.evidence:.2f}'); y -= 1
    axs['t'].text(0, y, f'ESS: {res.ESS}'); y -= 1
    axs['t'].text(0, y, f'fix: {res.fix}, $N_{{p, max}}: {res.npmax}$'); y -= 1
    if res.KO:
        axs['t'].text(0, y, 'KO: True'); y -= 1
    if res.TR:
        axs['t'].text(0, y, 'TR: True'); y -= 1
    if res.trend:
        axs['t'].text(0, y, f'trend: True, degree: {res.trend_degree}'); y -= 1
    if res.studentt:
        axs['t'].text(0, y, 'student-t: True'); y -= 1
    axs['t'].set(ylim=(y-1, 1))


    return fig, axs
