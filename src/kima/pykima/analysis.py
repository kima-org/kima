from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .results import KimaResults

import warnings
from collections import namedtuple
from typing import Tuple, Union

import numpy as np
from scipy.stats import norm, t as T, binned_statistic

from .. import MODELS
from .utils import mjup2mearth, mjup2msun, distribution_support

def np_most_probable(results: KimaResults):
    """
    Return the value of Np with the highest posterior probability.

    Arguments:
        results (KimaResults): A results instance
    """
    Nplanets = results.posterior_sample[:, results.index_component].astype(int)
    values, counts = np.unique(Nplanets, return_counts=True)
    return values[counts.argmax()]


def np_bayes_factor_threshold(results: KimaResults, threshold: float = 150):
    """
    Return the value of Np supported by the data considering a posterior ratio
    (Bayes factor) threshold.

    Arguments:
        results (KimaResults): A results instance
        threshold (float): Posterior ratio threshold.
    """
    bins = np.arange(results.max_components + 2)
    n, _ = np.histogram(results.Np, bins=bins)

    def Np_calc(T):
        Np = 0
        for i in range(bins[-2]):
            num, den = n[i + 1], n[i]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if num / den >= T:
                    Np = i + 1
        return Np

    if isinstance(threshold, (int, float)):
        Np = Np_calc(threshold)
    elif isinstance(threshold, (list, np.ndarray)):
        threshold = np.atleast_2d(threshold).T
        Np = np.apply_along_axis(Np_calc, 1, threshold)

    return Np
    # ratios = results.ratios
    # if isinstance(threshold, (int, float)):
    #     above = ratios > threshold
    #     return np.where(above, np.arange(results.npmax), -1).max() + 1
    # elif isinstance(threshold, (list, np.ndarray)):
    #     threshold = np.atleast_2d(threshold).T
    #     above = ratios > threshold
    #     return np.where(above, np.arange(results.npmax), -1).max(axis=1) + 1


def np_posterior_threshold(results: KimaResults, threshold: float = 0.9):
    """
    Return the value of Np supported by the data considering an absolute
    posterior probability threshold.

    Arguments:
        results (KimaResults): A results instance
        threshold (float): Posterior probability threshold
    """
    values = np.arange(results.npmax + 1)
    bins = np.arange(results.npmax + 2) - 0.5
    counts, _ = np.histogram(results.Np, bins=bins)
    # values, counts = np.unique(results.Np, return_counts=True)
    if isinstance(threshold, float):
        above = (counts / results.ESS) > threshold
        print(above)
        return np.where(above, np.arange(values.size), 0).max()
    elif isinstance(threshold, (list, np.ndarray)):
        threshold = np.atleast_2d(threshold).T
        above = (counts / results.ESS) > threshold
        return np.where(above, np.arange(values.size), 0).max(axis=1)


def compute_values_from_ratios(S, ratios):
    """
    Given a value for the sum S and a list of ratios between consecutive terms,
    returns the values of all terms a_1, a_2, ..., a_n.

    Args:
        S (float): sum of all terms
        ratios (list): 
            [r1, r2, ..., r_{n-1}] such that:
                a_2 = r1 * a_1,
                a_3 = r2 * a_2,
                ...
                a_n = r_{n-1} * a_{n-1}
    Returns:
        [a_1, a_2, ..., a_n] (list):
            individual values
    """
    terms = [1]  # T_0 = 1
    for r in ratios:
        terms.append(terms[-1] * r)
    sum_T = sum(terms)
    a1 = S / sum_T
    values = [a1 * t for t in terms]
    return values


def get_planet_mass(P: Union[float, np.ndarray], K: Union[float, np.ndarray],
                    e: Union[float, np.ndarray], star_mass: Union[float, Tuple] = 1.0,
                    full_output=False):
    r"""
    Calculate the planet (minimum) mass, $M_p \sin i$, given orbital period `P`,
    semi-amplitude `K`, eccentricity `e`, and stellar mass. If `star_mass` is a
    tuple with (estimate, uncertainty), this (Gaussian) uncertainty will be
    taken into account in the calculation. Note that this ignores the mass of the 
    planet in the calculation, if this is significant compared to the mass of the star, 
    then this method is not suitable.

    Args:
        P (Union[float, ndarray]):
            orbital period [days]
        K (Union[float, ndarray]):
            semi-amplitude [m/s]
        e (Union[float, ndarray]):
            orbital eccentricity
        star_mass (Union[float, Tuple]):
            stellar mass, or (mass, uncertainty) [Msun]

    This function returns different results depending on the inputs.

    !!! note "If `P`, `K`, and `e` are floats and `star_mass` is a float"

    Returns:
        Msini (float): planet mass, in $M_{\rm Jup}$
        Msini (float): planet mass, in $M_{\rm Earth}$

    !!! note "If `P`, `K`, and `e` are floats and `star_mass` is a tuple"

    Returns:
        Msini (tuple): planet mass and uncertainty, in $M_{\rm Jup}$
        Msini (tuple): planet mass and uncertainty, in $M_{\rm Earth}$

    !!! note "If `P`, `K`, and `e` are arrays and `full_output=True`"

    Returns:
        m_Msini (float):
            posterior mean for the planet mass, in $M_{\rm Jup}$
        s_Msini (float):
            posterior standard deviation for the planet mass, in $M_{\rm Jup}$
        Msini (array):
            posterior samples for the planet mass, in $M_{\rm Jup}$

    !!! note "If `P`, `K`, and `e` are arrays and `full_output=False`"

    Returns:
        m_Msini (float):
            posterior mean for the planet mass, in $M_{\rm Jup}$
        s_Msini (float):
            posterior standard deviation for the planet mass, in $M_{\rm Jup}$
        m_Msini (float):
            posterior mean for the planet mass, in $M_{\rm Earth}$
        s_Msini (float):
            posterior standard deviation for the planet mass, in $M_{\rm Earth}$
    """
    C = 4.919e-3

    try:
        P = float(P)
        # calculate for one value of the orbital period
        # then K, e, and star_mass should also be floats
        assert isinstance(K, float) and isinstance(e, float)
        uncertainty_star_mass = False
        if isinstance(star_mass, tuple) or isinstance(star_mass, list):
            star_mass = np.random.normal(star_mass[0], star_mass[1], 20000)
            uncertainty_star_mass = True

        m_mj = C * star_mass**(2. / 3) * P**(1. / 3) * K * np.sqrt(1 - e**2)
        m_me = m_mj * mjup2mearth
        if uncertainty_star_mass:
            return (m_mj.mean(), m_mj.std()), (m_me.mean(), m_me.std())
        else:
            return m_mj, m_me

    except TypeError:
        # calculate for an array of periods
        P = np.atleast_1d(P)
        if isinstance(star_mass, tuple) or isinstance(star_mass, list):
            # include (Gaussian) uncertainty on the stellar mass
            star_mass = np.random.normal(star_mass[0], star_mass[1], P.shape)

        m_mj = C * star_mass**(2. / 3) * P**(1. / 3) * K * np.sqrt(1 - e**2)
        m_me = m_mj * mjup2mearth

        if full_output:
            return m_mj.mean(), m_mj.std(), m_mj
        else:
            return (m_mj.mean(), m_mj.std(), m_me.mean(), m_me.std())
        
def get_planet_mass_GAIA(P: Union[float, np.ndarray], a0: Union[float, np.ndarray]
                         , parallax: Union[float, np.ndarray], star_mass: Union[float, Tuple] = 1.0
                         , full_output=False):
    r"""
    Calculate the planet (minimum) mass, $M_p \sin i$, given orbital period `P`,
    semi-major axis of photocentre `a0`, parallax 'parallax', and stellar mass. If `star_mass` is a
    tuple with (estimate, uncertainty), this (Gaussian) uncertainty will be
    taken into account in the calculation. Note that this ignores the mass of the planet in the calculation,
    if this is significant compared to the mass of the star, then this method is not suitable.

    Args:
        P (Union[float, ndarray]):
            orbital period [days]
        a0 (Union[float, ndarray]):
            semi-amplitude [mas]
        parallax (Union[float, ndarray]):
            parallax [mas]
        star_mass (Union[float, Tuple]):
            stellar mass, or (mass, uncertainty) [Msun]

    This function returns different results depending on the inputs.

    !!! note "If `P`, `a0`, and `parallax` are floats and `star_mass` is a float"

    Returns:
        Msini (float): planet mass, in $M_{\rm Jup}$
        Msini (float): planet mass, in $M_{\rm Earth}$

    !!! note "If `P`, `a0`, and `parallax` are floats and `star_mass` is a tuple"

    Returns:
        Msini (tuple): planet mass and uncertainty, in $M_{\rm Jup}$
        Msini (tuple): planet mass and uncertainty, in $M_{\rm Earth}$

    !!! note "If `P`, `a0`, and `parallax` are arrays and `full_output=True`"

    Returns:
        m_Msini (float):
            posterior mean for the planet mass, in $M_{\rm Jup}$
        s_Msini (float):
            posterior standard deviation for the planet mass, in $M_{\rm Jup}$
        Msini (array):
            posterior samples for the planet mass, in $M_{\rm Jup}$

    !!! note "If `P`, `a0`, and `parallax` are arrays and `full_output=False`"

    Returns:
        m_Msini (float):
            posterior mean for the planet mass, in $M_{\rm Jup}$
        s_Msini (float):
            posterior standard deviation for the planet mass, in $M_{\rm Jup}$
        m_Msini (float):
            posterior mean for the planet mass, in $M_{\rm Earth}$
        s_Msini (float):
            posterior standard deviation for the planet mass, in $M_{\rm Earth}$
    """
    # C = 4.919e-3

    try:
        P = float(P)/365.25 #convert period to years
        # calculate for one value of the orbital period
        # then K, e, and star_mass should also be floats
        assert isinstance(a0, float) and isinstance(parallax, float)
        uncertainty_star_mass = False
        if isinstance(star_mass, tuple) or isinstance(star_mass, list):
            star_mass = np.random.normal(star_mass[0], star_mass[1], 20000)
            uncertainty_star_mass = True

        m_ms = (star_mass/P)**(2. / 3) * a0/parallax
        m_mj = m_ms/mjup2msun
        m_me = m_mj * mjup2mearth
        if uncertainty_star_mass:
            return (m_mj.mean(), m_mj.std()), (m_me.mean(), m_me.std())
        else:
            return m_mj, m_me

    except TypeError:
        # calculate for an array of periods
        P = np.atleast_1d(P)/365.25 #convert period to years
        if isinstance(star_mass, tuple) or isinstance(star_mass, list):
            # include (Gaussian) uncertainty on the stellar mass
            star_mass = np.random.normal(star_mass[0], star_mass[1], P.shape)
        try:
            m_ms = (star_mass/P)**(2. / 3) * a0/parallax
        except ValueError:
            parallax_new = parallax[:, np.newaxis].copy()
            m_ms = (star_mass/P)**(2. / 3) * a0/parallax_new

        m_mj = m_ms/mjup2msun
        m_me = m_mj * mjup2mearth

        if full_output:
            return m_mj.mean(), m_mj.std(), m_mj
        else:
            return (m_mj.mean(), m_mj.std(), m_me.mean(), m_me.std())


def get_planet_semimajor_axis(P: Union[float, np.ndarray], K: Union[float, np.ndarray],
                              star_mass: Union[float, tuple] = 1.0, full_output=False):
    r"""
    Calculate the semi-major axis of the planet's orbit given orbital period
    `P`, semi-amplitude `K`, and stellar mass.

    Args:
        P (Union[float, ndarray]):
            orbital period [days]
        K (Union[float, ndarray]):
            semi-amplitude [m/s]
        star_mass (Union[float, Tuple]):
            stellar mass, or (mass, uncertainty) [Msun]

    This function returns different results depending on the inputs.

    !!! note "If `P` and `K` are floats and `star_mass` is a float"

    Returns:
        a (float): planet semi-major axis, in AU

    !!! note "If `P` and `K` are floats and `star_mass` is a tuple"

    Returns:
        a (tuple): semi-major axis and uncertainty, in AU

    !!! note "If `P` and `K` are arrays and `full_output=True`"

    Returns:
        m_a (float):
            posterior mean for the semi-major axis, in AU
        s_a (float):
            posterior standard deviation for the semi-major axis, in AU
        a (array):
            posterior samples for the semi-major axis, in AU

    !!! note "If `P` and `K` are arrays and `full_output=False`"

    Returns:
        m_a (float):
            posterior mean for the semi-major axis, in AU
        s_a (float):
            posterior standard deviation for the semi-major axis, in AU
    """
    # gravitational constant G in AU**3 / (Msun * day**2), to the power of 1/3
    f = 0.0666378476025686

    if isinstance(P, float):
        # calculate for one value of the orbital period
        # then K and star_mass should also be floats
        assert isinstance(K, float)
        uncertainty_star_mass = False
        if isinstance(star_mass, tuple) or isinstance(star_mass, list):
            star_mass = np.random.normal(star_mass[0], star_mass[1], 20000)
            uncertainty_star_mass = True

        a = f * star_mass**(1. / 3) * (P / (2 * np.pi))**(2. / 3)

        if uncertainty_star_mass:
            return a.mean(), a.std()

        return a  # in AU

    else:
        if isinstance(star_mass, tuple) or isinstance(star_mass, list):
            star_mass = star_mass[0] + star_mass[1] * np.random.randn(P.size)
        a = f * star_mass**(1. / 3) * (P / (2 * np.pi))**(2. / 3)

        if full_output:
            return a.mean(), a.std(), a
        else:
            return a.mean(), a.std()


def get_planet_mass_and_semimajor_axis(P, K, e, star_mass=1.0,
                                       full_output=False, verbose=False):
    """
    Calculate the planet (minimum) mass Msini and the semi-major axis given
    orbital period `P`, semi-amplitude `K`, eccentricity `e`, and stellar mass.
    If star_mass is a tuple with (estimate, uncertainty), this (Gaussian)
    uncertainty will be taken into account in the calculation.

    Units:
        P [days]
        K [m/s]
        e []
        star_mass [Msun]
    Returns:
        (M, A) where
            M is the output of get_planet_mass
            A is the output of get_planet_semimajor_axis
    """
    # this is just a convenience function for calling
    # get_planet_mass and get_planet_semimajor_axis

    if verbose:
        print('Using star mass = %s solar mass' % star_mass)

    mass = get_planet_mass(P, K, e, star_mass, full_output)
    a = get_planet_semimajor_axis(P, K, star_mass, full_output)
    return mass, a


def get_bins(res, start=None, end=None, nbins=100):
    try:
        _start, _end = distribution_support(res.priors['Pprior'])
        if _start in (0.0, -np.inf):
            raise KeyError
            # _start = res.posteriors.P[res.posteriors.P > 0].min()
        if _end in (np.inf, 0.0):
            raise KeyError
            # _end = res.posteriors.P.max()
    except KeyError:
        _start, _end = 1e-1, 1e7

    if start is None:
        start = _start
    if end is None:
        end = _end
    
    bins = 10**np.linspace(np.log10(start), np.log10(end), nbins)

    return bins

def aliases(P):
    yearly, solar_dayly, sidereal_dayly, synodic_monthly = 1 / 365.25, 1.0, 1 + 1 / 365.25, 1 / 29
    alias_year = [abs(1 / (yearly + i / P)) for i in [1, -1]]
    alias_solar_day = [abs(1 / (solar_dayly + i / P)) for i in [1, -1]]
    alias_sidereal_day = [abs(1 / (sidereal_dayly + i / P)) for i in [1, -1]]
    # alias_synodic_month = [abs(1 / (synodic_monthly + i / P)) for i in [1, -1]]
    return alias_year, alias_solar_day, alias_sidereal_day #, alias_synodic_month

# def FIP(results, plot=True, show_peaks=True, oversampling=1, include_aliases=False,
#         include_known_object=False, include_transiting_planet=False,
#         peak_ndigits=None, show_ESS=True, just_tip=False, fig=None):
#     """ Calculate (and plot) the True and False Inclusion Probability (TIP/FIP)

#     Args:
#         res (kima.KimaResults):
#             The `KimaResults` instance
#         plot (bool, optional): 
#             Plot the TIP and FIP. Defaults to True.
#         show_peaks (bool, optional): 
#             Identify and show prominent TIP peaks. Defaults to True.
#         oversampling (int, optional): 
#             Oversampling factor for the period binning. Defaults to 1.

#     Returns:
#         bins (np.ndarray):
#             The period bins
#         FIP (np.ndarray):
#             The False Inclusion Probability
#     """
#     Tobs = np.ptp(results.data.t)
#     Δf = 1 / Tobs / oversampling
#     bins = 1 / np.arange(Δf, 1.0, Δf / oversampling)
#     bins = bins[:5]

#     # # alias limits, in frequency
#     # a, b = 1/bins - Δf, 1/bins + Δf
#     # a_alias_year1, a_alias_year2 = 1 / np.array(kima.pykima.analysis.aliases(1/a)[0])
#     # b_alias_year1, b_alias_year2 = 1 / np.array(kima.pykima.analysis.aliases(1/b)[0])

#     P = results.posteriors.P.copy()

#     if include_known_object:
#         P = np.c_[P, results.posteriors.KO.P.copy()]
#     if include_transiting_planet:
#         P = np.concatenate([P, results.posteriors.TR.P.ravel()])

#     if include_aliases:
#         P = np.array(aliases(P)).reshape(-1, P.shape[1])

#     TIP = np.zeros_like(bins)
#     dig = np.digitize(P, bins)
#     i, c = np.unique(dig, return_counts=True)
#     TIP[i-1] = c
#     with np.errstate(divide='ignore'):
#         dig = np.digitize(P, 1/(1/bins - Δf))
#     i, c = np.unique(dig, return_counts=True)
#     TIP[i - 1] += c

#     TIP[-1] = 0

#     TIP /= results.ESS

#     # TIP /= TIP.max()

#     # very memory intensive and slower:
#     # with np.errstate(divide='ignore'):
#     #     c = np.logical_and(
#     #         # 
#     #             a[:, None, None] < 1.0 / P, 
#     #             1.0 / P < b[:, None, None]
#     #         # 
#     #     ).any(axis=2).sum(axis=1)
#     # TIP = c / results.ESS
#     # _1 = np.logical_and(a[:, None, None] < 1 / results.posteriors.P, 1 / results.posteriors.P < b[:, None, None])
#     # _21 = np.logical_and(a_alias_year1[:,None,None] < 1/results.posteriors.P, 1/results.posteriors.P < b_alias_year1[:,None,None])
#     # _22 = np.logical_and(a_alias_year2[:,None,None] < 1/results.posteriors.P, 1/results.posteriors.P < b_alias_year2[:,None,None])
#     # c1 = (_1 | _21 | _22).any(axis=2).sum(axis=1)

#     FIP = 1.0 - TIP
#     FIP = np.clip(FIP, 0, 1)
#     with np.errstate(divide='ignore'):
#         mlog10FIP = -np.log10(FIP)

#     if plot:
#         import matplotlib.pyplot as plt
#         from scipy.signal import find_peaks
#         if fig is None:
#             if just_tip:
#                 fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6, 3))
#                 axs = [axs]
#             else:
#                 fig, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
#         else:
#             axs = fig.axes

#         axs[0].semilogx(bins, TIP)
#         if not just_tip:
#             axs[1].semilogx(bins, mlog10FIP)

#         if show_peaks:
#             ndigits = peak_ndigits or len(str(results.ESS))
#             peaks, _peaks_info = find_peaks(TIP, height=0.9)
#             for peak in peaks:
#                 axs[0].plot(bins[peak], TIP[peak], 'ro', ms=3)
#                 axs[0].annotate(f'{bins[peak]:.2f}\nTIP: {TIP[peak]:.{ndigits}f}', 
#                                 xy=(bins[peak], TIP[peak]), xytext=(5, 0), 
#                                 xycoords='data', textcoords='offset points', 
#                                 ha='left', va='top')
#         if show_ESS:
#             ess = f'{results.ESS:,}'.replace(',', r'$\,$')
#             axs[0].set_title(f'ESS: {ess}', loc='right')
#         axs[0].set(xlabel='Period [days]', ylabel='TIP', ylim=(0, 1.05))

#         if not just_tip:
#             axs[1].set(xlabel='Period [days]', ylabel='-log$_{10}$ FIP', ylim=(0, None))

#         return bins, FIP, (fig, axs)

#     return bins, FIP


def FIP(results, f_center=None, f_width=None, oversampling=5, 
        plot=True, show_peaks=True, peak_indices=None, include_aliases=False,
        include_known_object=False, include_transiting_planet=False,
        peak_ndigits=None, show_ESS=True, just_tip=False, ax=None, **kwargs):
    r"""
    Compute and plot the TIP and FIP (true / false inclusion probability) based
    on (weighted) posterior samples.

    Arguments:
        results (KimaResults): A results instance
        f_center (ndarray, optional):
            Array of (linear) frequencies at which to compute the T/FIP.
        f_width (float, optional):
            Width of the frequency bins.
    """
    if f_center is None and f_width is None:
        Tobs = np.ptp(results.data.t)
        f_width = 1.0 / Tobs
        f_center = np.arange(f_width, 1.0, f_width / oversampling)
    
    if f_center is None:
        f_center = np.arange(f_width, 1.0, f_width / oversampling)
    
    P = results.posteriors.P.copy()

    if include_known_object:
        P = np.c_[P, results.posteriors.KO.P.copy()]
    if include_transiting_planet:
        P = np.concatenate([P, results.posteriors.TR.P.ravel()])

    if np.any(np.diff(f_center) > f_width):
        msg = "Non-overlapping windows in FIP calculation. "
        msg += "Some samples might be completely ignored."
        raise Exception(msg)

    FIP = np.ones_like(f_center)
    fa = f_center - f_width / 2
    fb = f_center + f_width / 2
    weights = np.ones(P.shape[0])
    weights = weights / np.sum(weights)

    with np.errstate(divide='ignore'):
        for fk, wk in zip(1 / P, weights):
            beg = np.atleast_1d(np.searchsorted(fb, fk, "right"))
            end = np.atleast_1d(np.searchsorted(fa, fk, "left"))
            listind = []
            for bi, ei in zip(beg, end):
                listind += range(bi, ei)
            FIP[listind] -= wk

    TIP = 1.0 - FIP
    mlog10FIP = -np.log10(FIP)
    pcenter = 1.0 / f_center

    if plot:
        import matplotlib.pyplot as plt
        from scipy.signal import find_peaks
        if ax is None:
            if just_tip:
                fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6, 3))
                axs = [axs]
            else:
                fig, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
        else:
            if just_tip:
                axs = [ax]
                fig = ax.figure
            else:
                ax = np.atleast_1d(ax)
                assert len(ax) == 2, "axs must have length 2"
                axs = ax

        axs[0].semilogx(pcenter, TIP, **kwargs)
        if not just_tip:
            axs[1].semilogx(pcenter, mlog10FIP, **kwargs)

        if show_peaks:
            ndigits = peak_ndigits or len(str(results.ESS))

            peaks, _peaks_info = find_peaks(TIP, height=0.9)
            if peak_indices is not None:
                peaks = peaks[peak_indices]

            for peak in peaks:
                axs[0].plot(pcenter[peak], TIP[peak], 'ro', ms=3)
                axs[0].annotate(f'{pcenter[peak]:.2f}\nTIP: {TIP[peak]:.{ndigits}f}', 
                                xy=(pcenter[peak], TIP[peak]), xytext=(5, 0), 
                                xycoords='data', textcoords='offset points', 
                                ha='left', va='top')
        if show_ESS:
            ess = f'{results.ESS:,}'.replace(',', r'$\,$')
            axs[0].set_title(f'ESS: {ess}', loc='right')
        axs[0].set(xlabel='Period [days]', ylabel='TIP', ylim=(0, 1.05))

        if not just_tip:
            axs[1].set(xlabel='Period [days]', ylabel='-log$_{10}$ FIP', ylim=(0, None))

        return pcenter, FIP, (fig, axs)

    return pcenter, FIP


def FIP_count_detections(results, alpha=0.05, Ptrue=None):
    if Ptrue is not None:
        from interval import Interval
        bins, fip = FIP(results, plot=False)
        ftrue = 1 / Ptrue
        Tobs = np.ptp(results.t)
        ffip = 1 / bins[fip.argmin()]
        return ftrue in Interval(ffip - 1 / Tobs, ffip + 1 / Tobs)

    bins, fip = FIP(results, oversampling=1, plot=False,
                    adjust_oversampling=False)

    alpha = np.atleast_2d(alpha).T
    below = fip < alpha

    if Ptrue is None:  # just return the number of bins with FIP < alpha
        return np.array([bins[b].size for b in below])

    low = np.array([bins[b] for b in below])
    #     return low.size
    # # if the true periods are given, check if they are recovered
    # upp = np.roll(bins, -1)[fip < alpha]
    # nfip = low.size  # detected by the FIP
    # n_mistakes = abs(nfip - len(Ptrue))
    # n_wrong_period = sum([
    #     0 if L < P and P < U else 1 for L, U, P in zip(low, upp, Ptrue)
    # ])
    # return nfip, n_mistakes, n_wrong_period


def true_within_hdi(results, truths, hdi_prob=0.95, only_periods=False,
                    cluster_before=True):
    # from arviz.stats import hdi
    from .hdi import hdi, hdi2
    from interval import Interval
    truths = np.atleast_2d(truths)
    nplanets = truths.shape[0]
    assert truths.shape[1] == 5, 'should provide 5 parameters per planet'

    regions = hdi2(results.T, hdi_prob)
    # if cluster_before:
    #     from scipy.cluster.vq import kmeans2
    #     centroids, labels = kmeans2(results.T, results.npmax)
    #     regions = []
    #     for i, _ in enumerate(centroids):
    #         regions.append(hdi(results.T[labels == i], skipna=True))
    # else:
    #     regions = 10**hdi(np.log10(results.T), hdi_prob,
    #                       multimodal=results.npmax > 1, skipna=True)
    regions = np.atleast_2d(regions)
    nregions = len(regions)

    if len(regions) < nplanets:
        print(f'Found less density regions ({nregions})', end=' ')
        print(f'than true planets ({nplanets})')
    if len(regions) > nplanets:
        print(f'Found more density regions ({nregions})', end=' ')
        print(f'than true planets ({nplanets})')

    intervals = [Interval(*peak) for peak in regions]
    print(intervals)

    nfound = 0
    for i in range(nplanets):
        P = truths[i][0]
        found = [P in interval for interval in intervals]
        if any(found):
            print(f'Found {P=:<5.4f} d in one of the {hdi_prob} density regions')
            nfound += 1
        else:
            print(f'!! Did not find {P=:<5.4f} d in any density region')

    if only_periods:
        return {'found': nfound, 'hdi': regions}

    nfound = 0
    withins = []
    for i in range(nplanets):
        P = truths[i][0]
        found = [P in interval for interval in intervals]
        if any(found):
            region = regions[np.where(found)[0][0]]

            # the period was found
            within = {'P': True}
            found = [True]

            mask = (results.T > region[0]) & (results.T < region[1])

            Kint = Interval(*hdi(results.A[mask], hdi_prob))
            K = truths[i][1]
            print(K, Kint)
            found.append(K in Kint)
            within['K'] = K in Kint

            # if the eccentricity is not fixed
            if 'Fixed' not in str(results.priors['eprior']):
                eccint = Interval(*hdi(results.E[mask], hdi_prob))
                ecc = truths[i][2]
                found.append(ecc in eccint)
                within['ecc'] = ecc in eccint

            # if the argument of periastron is not fixed
            if 'Fixed' not in str(results.priors['wprior']):
                wint = Interval(*hdi(results.Omega[mask], hdi_prob))
                w = truths[i][3]
                found.append(w in wint)
                within['w'] = w in wint

            #! ignore M0
            # M0int = Interval(*hdi(results.phi[mask], hdi_prob))
            # M0 = truths[i][4]
            # found.append(M0 in M0int)
            # within['M0'] = M0 in M0int

            if all(found):
                print(f'For {P=:<5.2f} d, the other parameters also match')
                nfound += 1

            withins.append(within)


    return {'found': nfound, 'hdi': regions, 'withins': withins}

def reorder_P(res, replace=False, passes=1):
    from .results import posterior_holder

    # if res.max_components >= 4:
    #     raise NotImplementedError('Only 2- or 3-planet solutions are supported')

    new_posterior = posterior_holder()
    new_posterior.P = res.posteriors.P.copy()
    new_posterior.K = res.posteriors.K.copy()
    new_posterior.e = res.posteriors.e.copy()
    new_posterior.w = res.posteriors.w.copy()
    new_posterior.φ = res.posteriors.φ.copy()

    mask = res.Np > 1
    print(mask.sum())

    for i in range(passes):
        P = new_posterior.P.copy()
        m, s = np.median(P[mask, 0]), np.std(P[mask, 0])
        maybe_wrong_order = np.abs(P[:, 0] - m) > s
        new_posterior.P[mask & maybe_wrong_order] = new_posterior.P[mask & maybe_wrong_order][:, ::-1]
        new_posterior.K[mask & maybe_wrong_order] = new_posterior.K[mask & maybe_wrong_order][:, ::-1]
        new_posterior.e[mask & maybe_wrong_order] = new_posterior.e[mask & maybe_wrong_order][:, ::-1]
        new_posterior.w[mask & maybe_wrong_order] = new_posterior.w[mask & maybe_wrong_order][:, ::-1]
        new_posterior.φ[mask & maybe_wrong_order] = new_posterior.φ[mask & maybe_wrong_order][:, ::-1]

    if replace:
        res.posteriors = new_posterior

    return new_posterior


def reorder_P2(res, replace=False, passes=1):
    from tqdm import trange, tqdm
    from .results import posterior_holder

    # if res.max_components >= 4:
    #     raise NotImplementedError('Only 2- or 3-planet solutions are supported')

    new_posterior = posterior_holder()
    new_posterior.P = res.posteriors.P.copy()
    new_posterior.K = res.posteriors.K.copy()
    new_posterior.e = res.posteriors.e.copy()
    new_posterior.w = res.posteriors.w.copy()
    new_posterior.φ = res.posteriors.φ.copy()
    if res.model is MODELS.RVHGPMmodel:
        new_posterior.i = res.posteriors.i.copy()
        new_posterior.W = res.posteriors.W.copy()

    mask = res.Np > 1
    Np = res.Np.copy().astype(int)

    s = np.std(new_posterior.P, axis=0)
    s_old = s.copy()

    passes_done = 0
    keep_going = True
    while keep_going:
        for i in tqdm(np.where(mask)[0]):
            for j in range(Np[i] - 1):
                ind = [j, j+1]
                # print(new_posterior.P[i])
                new_posterior.P[i, ind] = new_posterior.P[i, ind][::-1]
                # print(new_posterior.P[i])
                _s = np.std(new_posterior.P, axis=0)
                # print(_s, (_s < s)[j])
                if (_s < s)[j]:
                    new_posterior.K[i, ind] = new_posterior.K[i, ind][::-1]
                    new_posterior.e[i, ind] = new_posterior.e[i, ind][::-1]
                    new_posterior.w[i, ind] = new_posterior.w[i, ind][::-1]
                    new_posterior.φ[i, ind] = new_posterior.φ[i, ind][::-1]
                    s = _s
                else:
                    new_posterior.P[i, ind] = new_posterior.P[i, ind][::-1]
                # input()

        passes_done += 1
        if passes != -1 and passes_done >= passes:
            keep_going = False
        if passes == -1:
            if (s.round(2) < s_old.round(2)).any():
                s_old = s
                keep_going = True
            else:
                keep_going = False
        # print(s)

    if replace:
        res.posteriors.P = new_posterior.P
        res.posteriors.K = new_posterior.K
        res.posteriors.e = new_posterior.e
        res.posteriors.w = new_posterior.w
        res.posteriors.φ = new_posterior.φ

    return new_posterior
    from .results import posterior_holder

    new_posterior = posterior_holder()
    new_posterior.P = res.posteriors.P.copy()
    new_posterior.K = res.posteriors.K.copy()
    new_posterior.e = res.posteriors.e.copy()
    new_posterior.w = res.posteriors.w.copy()
    new_posterior.φ = res.posteriors.φ.copy()

    for i in range(res.max_components, 1, -1):
        can_calculate_distance = (res.Np == i - 1).any() or reference_periods
        if can_calculate_distance:
            mask_i = res.Np == i
            mask_im1 = res.Np == i - 1
            r = np.where(mask_i)[0]

            if reference_periods is not None:
                reference = reference_periods[:i - 1]
                # print(reference)
            else:
                reference = new_posterior.P[mask_im1, i - 2].mean()

            dist_sort = np.abs(new_posterior.P[mask_i] - reference).argsort()

            new_posterior.P[mask_i] = new_posterior.P[r, dist_sort.T].T
            new_posterior.K[mask_i] = new_posterior.K[r, dist_sort.T].T
            new_posterior.e[mask_i] = new_posterior.e[r, dist_sort.T].T
            new_posterior.w[mask_i] = new_posterior.w[r, dist_sort.T].T
            new_posterior.φ[mask_i] = new_posterior.φ[r, dist_sort.T].T

    if replace:
        res.posteriors.P = new_posterior.P
        res.posteriors.K = new_posterior.K
        res.posteriors.e = new_posterior.e
        res.posteriors.w = new_posterior.w
        res.posteriors.φ = new_posterior.φ
        res.posterior_sample[:, res.indices['planets.P']] = new_posterior.P
        res.posterior_sample[:, res.indices['planets.K']] = new_posterior.K
        res.posterior_sample[:, res.indices['planets.e']] = new_posterior.e
        res.posterior_sample[:, res.indices['planets.w']] = new_posterior.w
        res.posterior_sample[:, res.indices['planets.φ']] = new_posterior.φ

    return new_posterior


def sort_planet_samples(res, byP=True, replace=False):
    # here we sort the planet_samples array by the orbital period
    # this is a bit difficult because the organization of the array is
    # P1 P2 K1 K2 ....
    from .results import posterior_holder
    P = res.posteriors.P.copy()
    sortP = np.argsort(P, axis=1)

    new_posterior = posterior_holder()
    fields = (
        'K', 'e', 'w', 'φ',
        'i', 'i_deg', 'W', 'W_deg', 'Ω', 'Ω_deg',
    )

    if byP:
        n = np.arange(P.shape[0])[:, None]
        new_posterior.P = P[n, sortP].copy()
        for field in fields:
            try:
                setattr(new_posterior, field, 
                        getattr(res.posteriors, field)[n, sortP].copy())
            except AttributeError:
                pass
        # new_posterior.K = res.posteriors.K[n, sortP].copy()
        # new_posterior.e = res.posteriors.e[n, sortP].copy()
        # new_posterior.w = res.posteriors.w[n, sortP].copy()
        # new_posterior.φ = res.posteriors.φ[n, sortP].copy()
        # if res.model is MODELS.RVHGPMmodel:
        #     new_posterior.i = res.posteriors.i[n, sortP].copy()
        #     new_posterior.W = res.posteriors.W[n, sortP].copy()

    if replace:
        res.posteriors.P = new_posterior.P
        res.posterior_sample[:, res.indices['planets.P']] = new_posterior.P

        for field in fields:
            try:
                setattr(res.posteriors, field, getattr(new_posterior, field))
            except AttributeError:
                pass
            try:
                res.posterior_sample[:, res.indices['planets.' + field]] = getattr(new_posterior, field)
            except (AttributeError, KeyError):
                pass

    return new_posterior


def planet_parameters(results, star_mass=1.0, sample=None, printit=True):
    if sample is None:
        sample = results.maximum_likelihood_sample(printit=printit)

    mass_errs = False
    if isinstance(star_mass, (tuple, list)):
        mass_errs = True

    format_tuples = lambda data: r'{:10.5f} \pm {:.5f}'.format(
        data[0], data[1]) if mass_errs else '{:12.5f}'.format(data)

    if printit:
        print()
        final_string = star_mass if not mass_errs else r'{} \pm {}'.format(
            star_mass[0], star_mass[1])
        print('Calculating planet masses with Mstar = {} Msun'.format(
            final_string))

    indices = results.indices
    mc = results.max_components

    # planet parameters
    pars = sample[indices['planets']].copy()
    # number of planets in this sample
    nplanets = (pars[:mc] != 0).sum()

    if printit:
        extra_padding = 12 * ' ' if mass_errs else ''
        print(20 * ' ' + ('%12s' % 'Mp [Mearth]') + extra_padding +
              ('%12s' % 'Mp [Mjup]'))

    masses = []
    for j in range(int(nplanets)):
        P = pars[j + 0 * mc]
        if P == 0.0:
            continue
        K = pars[j + 1 * mc]
        # phi = pars[j + 2 * mc]
        # t0 = t[0] - (P * phi) / (2. * np.pi)
        ecc = pars[j + 3 * mc]
        # w = pars[j + 4 * mc]
        m_mjup, m_mearth = get_planet_mass(P, K, ecc, star_mass=star_mass)

        if printit:
            s = 18 * ' '
            s += format_tuples(m_mearth)
            s += format_tuples(m_mjup)
            masses.append(m_mearth)
            print(s)
    return np.array(masses)


def compare_planet_parameters(results1, results2, star_mass=1.0):
    import pygtc
    pars1 = results1.posterior_sample[:, results1.indices['planets']]
    pars2 = results2.posterior_sample[:, results2.indices['planets']]
    assert pars1.shape[1] == pars2.shape[1], 'Number of planets is different!'
    figs = []
    mc = results1._mc
    for i in range(mc):
        GTC = pygtc.plotGTC(chains=[pars1[:, i::mc], pars2[:, i::mc]])
        figs.append(GTC)


def find_outliers(results, sample, threshold=10, full_output=False):
    """
    Estimate which observations are outliers, for a model with a Student t
    likelihood. This function first calculates the residuals given the
    parameters in `sample`. Then it computes the relative probability of each
    residual point given a Student-t (Td) and a Gaussian (Gd) likelihoods. If
    the probability Td is larger than Gd (by a factor of `threshold`), the point
    is flagged as an outlier. The function returns an "outlier mask". If 
    `full_output` is True, it also returns the ratio Td/Gd.
    """
    res = results
    # the model must have studentt = true
    if not res.studentt:
        raise ValueError('studentt option is off, cannot estimate outliers')

    # calculate residuals for this sample
    resid = res.residuals(sample)

    # this sample's jitter and degrees of freedom
    J = sample[res.indices['jitter']]
    if isinstance(J, float):
        pass
    elif J.shape[0] > 1:
        # one jitter per instrument
        J = J[(res.data.obs - 1).astype(int)]

    nu = sample[res.indices['nu']]

    # probabilities within the Gaussian and Student-t likelihoods
    Gd = norm(loc=0, scale=np.hypot(res.data.e, J)).pdf(resid)
    Td = T(df=nu, loc=0, scale=np.hypot(res.data.e, J)).pdf(resid)

    # take care when probabilities are zero (very strong outliers)
    Gd = np.clip(Gd, 1e-15, None)

    # if Td/Gd > threshold, the point is an outlier, in the sense that it is
    # more likely within the Student-t likelihood than it would have been within
    # the Gaussian likelihood
    ratio = Td / Gd
    outlier = ratio > threshold

    if full_output:
        return outlier, ratio

    return outlier


def detection_limits(results, star_mass: Union[float, Tuple] = 1.0,
                     Np: int = None, bins: Union[int, np.ndarray] = 200,
                     plot=True, ax=None, semi_amplitude=False, semi_major_axis=False,
                     logX=True, return_mask=False, remove_nan=True,
                     show_eccentricity=False, K_lines=(5, 3, 1), smooth=False,
                     smooth_degree: int = 3):
    """
    Calculate detection limits using samples with more than `Np` planets. By
    default, this function uses the value of `Np` which passes the posterior
    probability threshold.

    Arguments:
        star_mass (Union[float, Tuple]):
            Stellar mass and optionally its uncertainty [in solar masses].
        Np (int):
            Consider only posterior samples with more than `Np` planets.
        bins (int):
            Number of bins at which to calculate the detection limits. The
            period ranges from the minimum to the maximum orbital period in the
            posterior.
        plot (bool):
            Whether to plot the detection limits
        semi_amplitude (bool):
            Show the detection limits for semi-amplitude, instead of planet mass
        semi_major_axis (bool):
            Show semi-major axis in the x axis, instead of orbital period
        logX (bool):
            Should X-axis bins be logarithmic?
        return_mask (bool):
        remove_nan (bool):
            remove bins with no counts (no posterior samples)
        smooth (bool):
            Smooth the binned maximum with a polynomial
        smooth_degree (int):
            Degree of the polynomial used for smoothing

    Returns:
        P,K,E,M (ndarray):
            Orbital periods, semi-amplitudes, eccentricities, and planet masses
            used in the calculation of the detection limits. These correspond to
            all the posterior samples with more than `Np` planets
        s (namedtuple):
            Detection limits result, with attributes `max` and `bins`. The `max`
            array is in units of Earth masses, `bins` is in days.
    """
    res = results

    if isinstance(results, list):
        print(f'combining samples from {len(results)} results')
        P = np.vstack([r.posteriors.P for r in results])
        K = np.vstack([r.posteriors.K for r in results])
        E = np.vstack([r.posteriors.e for r in results])
    else:
        if Np is None:
            Np = np_bayes_factor_threshold(res)

        if res.verbose:
            print(f'Using samples with Np > {Np}')

        P = res.posteriors.P[:, Np:]
        K = res.posteriors.K[:, Np:]
        E = res.posteriors.e[:, Np:]

    inds = np.nonzero(P)

    P = P[inds]
    K = K[inds]
    E = E[inds]

    M, A = get_planet_mass_and_semimajor_axis(P, K, E, star_mass=star_mass,
                                              full_output=True)
    M = M[2]
    A = A[2]

    # Mjit = get_planet_mass(P, np.hypot(K, J), E, star_mass=star_mass,
    #                        full_output=True)
    # Mjit = Mjit[2]

    if semi_major_axis:
        start, end = A.min(), A.max()
    else:
        start, end = P.min(), P.max()

    if isinstance(bins, int):
        if logX:
            bins = 10**np.linspace(np.log10(start), np.log10(end), bins)

    # bins_start = bins[:-1]# - np.ediff1d(bins)/2
    # bins_end = bins[1:]# + np.ediff1d(bins)/2
    # bins_start = np.append(bins_start, bins_end[-1])
    # bins_end = np.append(bins_end, P.max())

    DLresult = namedtuple('DLresult', ['max', 'bins'])

    if semi_amplitude:
        if semi_major_axis:
            s = binned_statistic(A, K, statistic='max', bins=bins)
            # sj = binned_statistic(A,
            #                       np.hypot(K, J),
            #                       statistic='max',
            #                       bins=bins)
        else:
            s = binned_statistic(P, K, statistic='max', bins=bins)
            # sj = binned_statistic(P,
            #                       np.hypot(K, J),
            #                       statistic='max',
            #                       bins=bins)
    else:
        if semi_major_axis:
            s = binned_statistic(A, M * mjup2mearth, statistic='max',
                                 bins=bins)
            # sj = binned_statistic(A,
            #                       Mjit * mjup2mearth,
            #                       statistic='max',
            #                       bins=bins)
        else:
            s = binned_statistic(P, M * mjup2mearth, statistic='max',
                                 bins=bins)
            # sj = binned_statistic(P,
            #                       Mjit * mjup2mearth,
            #                       statistic='max',
            #                       bins=bins)

    if remove_nan:
        w = ~np.isnan(s.statistic)
    else:
        w = np.full(s.statistic.size, True)

    bins = s.bin_edges[:-1] + np.ediff1d(s.bin_edges) / 2

    if isinstance(bins, np.ndarray):
        s = DLresult(max=s.statistic[w], bins=bins[w])
    # sj = DLresult(max=sj.statistic[w], bins=bins[w])

    # s99 = binned_statistic(P, M, statistic=lambda x: np.percentile(x, 99),
    #                        bins=bins)
    # s99 = DLresult(max=s99.statistic,
    #                bins=s99.bin_edges[:-1] + np.ediff1d(s99.bin_edges) / 2)

    if plot:
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
        else:
            fig = ax.figure

        if isinstance(star_mass, tuple):
            Ms = star_mass[0]
        else:
            Ms = star_mass

        if show_eccentricity:
            plotf = ax.scatter
        else:
            plotf = ax.plot

        lege = []
        kw_points = dict(color='k', ms=2, alpha=0.3, zorder=-1)
        if semi_amplitude:  # K [m/s] in the y axis
            if semi_major_axis:
                plotf(A, K, '.', **kw_points)
            else:
                plotf(P, K, '.', **kw_points)

        else:  # M [M⊕] in the y axis
            if semi_major_axis:
                sX = np.sort(A)
            else:
                sX = np.sort(P)

            one_ms = 4.919e-3 * Ms**(2. / 3) * sX**(1. / 3) * 1
            kw_lines = dict(color='C0', alpha=1, zorder=3)
            ls = (':', '--', '-')
            for i, f in enumerate(K_lines):
                ax.loglog(sX, f * one_ms * mjup2mearth, ls=ls[i], **kw_lines)
                lege.append(f'{f} m/s')

            if semi_major_axis:
                plotf(A, M * mjup2mearth, '.', **kw_points)
            else:
                plotf(P, M * mjup2mearth, '.', **kw_points)

        ax.set_xscale('log')
        ax.set_yscale('log')

        if show_eccentricity:
            ax.colorbar()

        if smooth:
            p = np.polyfit(np.log10(s.bins), np.log10(s.max), smooth_degree)
            ax.loglog(s.bins, 10**np.polyval(p, np.log10(s.bins)), color='m')
        else:
            ax.loglog(s.bins, s.max, '-o', ms=3, color='m')

        if not semi_major_axis:
            kwargs = dict(ls='--', lw=2, alpha=0.5, zorder=-1, color='C5')
            if isinstance(res, list):
                for r in res:
                    ax.axvline(np.ptp(r.data.t), 0.5, 1, **kwargs)
            else:
                ax.axvline(np.ptp(res.data.t), 0.5, 1, **kwargs)

        # ax.hlines(s.max, bins_start, bins_end, lw=2)
        # ax.loglog(s99.bins, s99.max * mjup2mearth)

        lege += ['posterior samples', 'binned maximum', 'time span']
        if smooth:
            lege[-2] = '(smoothed) ' + lege[-2]

        # if add_jitter:
        #     ax.fill_between(s.bins, s.max, s.max + sj.max, color='r',
        #                     alpha=0.1, lw=0)

        ax.legend(lege, ncol=2, frameon=False)
        # ax.set(ylim=(0.5, None))

        if semi_amplitude:
            Ly = r'Semi-amplitude [m/s]'
        else:
            Ly = r'Planet mass [M$_\oplus$]'

        if semi_major_axis:
            Lx = 'Semi-major axis [AU]'
        else:
            Lx = 'Orbital period [days]'

        ax.set(xlabel=Lx, ylabel=Ly)
        try:
            ax.set_xlim(results.priors['Pprior'].support())
        except (AttributeError, KeyError):
            pass

        try:
            ax.set_title(res.star)
        except AttributeError:
            pass

        return fig, P, K, E, M, s

    return P, K, E, M, s


def parameter_clusters(results, n_clusters=None, method='KMeans',
                       include_ecc=False, scale=False, plot=True, downsample=1,
                       **kwargs):
    try:
        import sklearn.cluster
        from sklearn import preprocessing
        from sklearn.metrics import silhouette_samples, silhouette_score
    except ImportError:
        print('Please install the sklearn package in order to use this function')
        return

    from scipy.spatial.distance import cdist

    res = results

    if include_ecc:
        # data = res.posterior_sample[:, res.indices['planets']]
        data = np.c_[res.posteriors.P.ravel(), res.posteriors.K.ravel(), res.posteriors.e.ravel()]
    else:
        # data = np.c_[res.T, res.A, res.Omega, res.phi]
        data = np.c_[
            res.posteriors.P.ravel().copy(), 
            res.posteriors.K.ravel().copy()
        ]

    data = data[::downsample, :]

    if scale:
        scaler = preprocessing.StandardScaler(copy=True, with_mean=True,
                                              with_std=True)
        scaler.fit(data)
        data_orig = data.copy()
        data = scaler.transform(data, copy=True)

    if n_clusters is None:
        k = res.max_components
    else:
        k = n_clusters

    min_samples = data.shape[0] // 100

    if method == 'KMeans':
        clustering = sklearn.cluster.KMeans(n_clusters=k, **kwargs)

    elif method == 'OPTICS':
        kwargs.setdefault('min_samples', min_samples)
        kwargs.setdefault('n_jobs', 4)
        clustering = sklearn.cluster.OPTICS(**kwargs)

    elif method == 'DBSCAN':
        kwargs.setdefault('n_jobs', -1)
        kwargs.setdefault('eps', 1)
        kwargs.setdefault('min_samples', min_samples)
        clustering = sklearn.cluster.DBSCAN(**kwargs)

    elif method == 'ward':
        clustering = sklearn.cluster.AgglomerativeClustering(
            linkage='ward', n_clusters=k, **kwargs)

    pred = clustering.fit_predict(data)
    # centroids, labels = kmeans2(data, k, minit='++')

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        kw = dict(
            figsize=(6, 3),
            constrained_layout=True,
            #   gridspec_kw={'sharex': True}
        )
        if include_ecc:
            fig, axs = plt.subplot_mosaic('ab\ndc', **kw)
        else:
            fig, axs = plt.subplot_mosaic('a', **kw)

        ax = axs['a']
        kw = dict(s=2, alpha=0.1, cmap='Paired')
        if scale:
            sc = ax.scatter(data_orig[:, 0], data_orig[:, 1], c=pred, **kw)
        else:
            sc = ax.scatter(data[:, 0], data[:, 1], c=pred, **kw)
        plt.colorbar(sc)
        ax.set(xscale='log', xlabel='P [days]', ylabel='K [m/s]')

        # if include_ecc:
        #     ax = axs['a']
        #     ax.scatter(data[:, 0], data[:, 2], c=pred, s=2, alpha=0.1)
        #     ax.set(xscale='log', xlabel='P [days]', ylabel=r'$\omega$')

        #     ax = axs['c']
        #     ax.scatter(data[:, 0], data[:, 3], c=pred, s=2, alpha=0.1)
        #     ax.set(xscale='log', xlabel='P [days]', ylabel=r'$\phi$')

        #     ax = axs['d']
        #     ax.scatter(data[:, 0], data[:, 4], c=pred, s=2, alpha=0.1)
        #     ax.set(xscale='log', xlabel='P [days]', ylabel='ecc')

        if False and method == 'OPTICS':
            space = np.arange(len(data))
            reachability = clustering.reachability_[clustering.ordering_]
            labels = clustering.labels_[clustering.ordering_]

            # Reachability plot
            ax1 = axs['a']
            for klass in range(0, k):
                Xk = space[labels == klass]
                Rk = reachability[labels == klass]
                ax1.plot(Xk, Rk, '.', ms=1, alpha=0.3)

            ax1.plot(space[labels == -1], reachability[labels == -1], 'k.',
                     alpha=0.3)
            ax1.set_ylabel('Reachability (epsilon distance)')

        if False and method == 'KMeans':
            # k means determine k
            distortions = []
            K = range(1, 10)
            for k in K:
                kmeanModel = sklearn.cluster.KMeans(n_clusters=k).fit(data)
                distortions.append(
                    sum(
                        np.min(
                            cdist(data, kmeanModel.cluster_centers_,
                                  'euclidean'), axis=1)) / data.shape[0])
            ax1 = axs['a']
            ax1.plot(K, distortions, '-o')

        if False and method == 'KMeans':
            # k means determine k
            silhouettes = []
            K = range(2, 10)
            for k in K:
                pred = sklearn.cluster.KMeans(n_clusters=k).fit_predict(data)
                score = silhouette_score(data, pred)
                silhouettes.append(score)
                print("For k =", k, "the average silhouette_score is :", score)

            ax1 = axs['a']
            ax1.plot(K, silhouettes, '-o')

        if False:
            # silhouette plot
            ax1 = axs['a']
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (k+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(data) + (k + 1) * 10])

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(data, pred)
            print("For k =", k, "The average silhouette_score is :",
                  silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(data, pred)

            y_lower = 10
            for i in range(k):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[pred == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / k)
                ax1.fill_betweenx(np.arange(y_lower, y_upper), 0,
                                  ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_xlabel("silhouette coefficients")
            ax1.set_ylabel("cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    return clustering


def full_model_table(res, sample, instruments=None, star_mass=1.0):
    from urepr import uformatul
    from tabletexifier import Table
    ind = res.indices
    instruments = instruments or res.instruments

    lower, median, upper = np.percentile(res.posterior_sample, [16, 50, 84],
                                         axis=0)

    x = Table(['Parameter', '$N_p=2$ model', 'Units'], table_style='T')

    # planets
    x.add_row([r'\multicolumn{3}{c}{Keplerians}', '', ''])
    parnames = ['P', 'K', 'M_0', 'e', r'\omega']
    units = ['days', 'm/s', '', '', '']
    planet_names = ['b', 'd']
    pl = sample[ind['planets']]
    samples = sort_planet_samples(res, res.posterior_sample[:, ind['planets']])
    lowerpl, medianpl, upperpl = np.percentile(samples, [16, 50, 84], axis=0)

    for pli in range(res.max_components):
        x.add_row([planet_names[pli], '', ''])
        isamples = samples[:, pli::res.max_components]
        this_planet_par = pl[pli::res.max_components]
        this_planet_low = lowerpl[pli::res.max_components]
        this_planet_med = medianpl[pli::res.max_components]
        this_planet_upp = upperpl[pli::res.max_components]
        for name, par, parlow, parmed, parhigh, unit in zip(
                parnames, this_planet_par, this_planet_low, this_planet_med,
                this_planet_upp, units):
            v = uformatul(par, parhigh - parmed, parmed - parlow, 'L')
            if name == 'e':
                if par - (parmed - parlow) < 0.0:
                    v = uformatul(par, parhigh - parmed, par, 'L')
            v = v.center(len(v) + 2, '$')
            x.add_row([f'${name}$', v, unit])

        x.add_row(3 * [''])

        _P, _K, _phi, _ecc, _w = this_planet_par
        if isinstance(star_mass, tuple):
            sm = star_mass[0]
        else:
            sm = star_mass
        _Mpar, _Apar = get_planet_mass_and_semimajor_axis(
            _P, _K, _ecc, star_mass=sm)
        _Mpar = _Mpar[1]

        _M = get_planet_mass(isamples[:, 0], isamples[:, 1], isamples[:, 3],
                             star_mass, full_output=True)[-1]
        _M *= mjup2mearth
        _Mlow, _Mmed, _Mupp = np.percentile(_M, [16, 50, 84], axis=0)
        v = uformatul(_Mpar, _Mupp - _Mmed, _Mmed - _Mlow, 'L')
        v = v.center(len(v) + 2, '$')
        x.add_row([r'$M_p \,\sin\!i$', v, r'$M_\oplus$'])

        _A = get_planet_semimajor_axis(isamples[:, 0], isamples[:, 1],
                                       star_mass, full_output=True)[-1]
        _Alow, _Amed, _Aupp = np.percentile(_A, [16, 50, 84], axis=0)
        v = uformatul(_Apar, _Aupp - _Amed, _Amed - _Alow, 'L')
        v = v.center(len(v) + 2, '$')
        x.add_row(['$a$', v, 'au'])

    x.add_row(3 * [''])
    x.add_row([r'\multicolumn{3}{c}{GP}', '', ''])
    # GP
    if res.has_gp:
        parnames = [r'$\eta_1$', r'$\eta_2$', r'$\eta_3$', r'$\eta_4$']
        units = ['m/s', 'days', 'days', '']
        if res.model == 'RVFWHMmodel':
            parnames[0] = r'$\eta_1$ RV'
            parnames.insert(1, r'$\eta_1$ FWHM')
            units.insert(1, 'm/s')
        η = sample[ind['GPpars']]
        ηlow, ηmed, ηupp = lower[ind['GPpars']], median[ind['GPpars']], upper[
            ind['GPpars']]
        for ηi, low, med, upp, name, unit in zip(η, ηlow, ηmed, ηupp, parnames,
                                                 units):
            v = uformatul(ηi, upp - med, med - low, 'L')
            v = v.center(len(v) + 2, '$')
            x.add_row([f'{name}', v, unit])

    # jitters
    x.add_row(3 * [''])
    x.add_row([r'\multicolumn{3}{c}{Noise}', '', ''])
    J = sample[ind['jitter']]
    Jlow, Jmed, Jupp = lower[ind['jitter']], median[ind['jitter']], upper[
        ind['jitter']]
    for j, low, med, upp, inst in zip(J, Jlow, Jmed, Jupp, instruments):
        v = uformatul(j, upp - med, med - low, 'L')
        if j - (med - low) < 0.0:
            v = uformatul(j, upp - med, j, 'L')
        v = v.center(len(v) + 2, '$')
        x.add_row([f'$j^{{RV}}_{{\\rm {inst}}}$', v, 'm/s'])

    if res.model == 'RVFWHMmodel':
        n = res.n_instruments
        for j, low, med, upp, inst in zip(J[n:], Jlow[n:], Jmed[n:], Jupp[n:],
                                          instruments):
            v = uformatul(j, upp - med, med - low, 'L')
            if j - (med - low) < 0.0:
                v = uformatul(j, upp - med, j, 'L')
            v = v.center(len(v) + 2, '$')
            x.add_row([f'$j^{{FWHM}}_{{\\rm {inst}}}$', v, 'm/s'])

    x.add_row(3 * [''])
    x.add_row([r'\multicolumn{3}{c}{Background}', '', ''])

    if res.trend:
        parnames = ('RV slope', 'RV quadr', 'RV cubic')
        units = ['m/s/yr', 'm/s/yr²', 'm/s/yr³']
        trend = sample[ind['trend']].copy()
        trend *= 365.25**np.arange(1, res.trend_degree + 1)
        trendlow, trendmed, trendupp = lower[ind['trend']], median[
            ind['trend']], upper[ind['trend']]
        trendlow *= 365.25**np.arange(1, res.trend_degree + 1)
        trendmed *= 365.25**np.arange(1, res.trend_degree + 1)
        trendupp *= 365.25**np.arange(1, res.trend_degree + 1)
        for t, tlow, tmed, tupp, name, unit in zip(trend, trendlow, trendmed,
                                                   trendupp, parnames, units):
            v = uformatul(t, tupp - tmed, tmed - tlow, 'L')
            v = v.center(len(v) + 2, '$')
            x.add_row([f'{name}', v, unit])

    # instrument offsets
    O = sample[ind['inst_offsets']]
    Olow, Omed, Oupp = lower[ind['inst_offsets']], median[
        ind['inst_offsets']], upper[ind['inst_offsets']]
    for o, low, med, upp, inst in zip(O, Olow, Omed, Oupp, instruments[:-1]):
        v = uformatul(o, upp - med, med - low, 'L')
        v = v.center(len(v) + 2, '$')
        x.add_row([f'RV offset {inst}-{instruments[-1]}', v, 'm/s'])

    if res.model == 'RVFWHMmodel':
        n = O.size // 2
        for o, low, med, upp, inst in zip(O[n:], Olow[n:], Omed[n:], Oupp[n:],
                                          instruments[:-1]):
            v = uformatul(o, upp - med, med - low, 'L')
            v = v.center(len(v) + 2, '$')
            x.add_row([f'FWHM offset {inst}-{instruments[-1]}', v, 'm/s'])

    vsys = sample[ind['vsys']]
    vsyslow, vsysmed, vsysupp = lower[ind['vsys']], median[ind['vsys']], upper[
        ind['vsys']]
    v = uformatul(vsys, vsysupp - vsysmed, vsysmed - vsyslow, 'L')
    v = v.center(len(v) + 2, '$')
    x.add_row([r'$v_{\rm sys}$', v, 'm/s'])

    fsys = sample[ind['C2']]
    fsyslow, fsysmed, fsysupp = lower[ind['C2']], median[ind['C2']], upper[
        ind['C2']]
    v = uformatul(fsys, fsysupp - fsysmed, fsysmed - fsyslow, 'L')
    v = v.center(len(v) + 2, '$')
    x.add_row([r'$f_{\rm sys}$', v, 'm/s'])

    print(x)
    return x


def _column_dynamic_ranges(results):
    """ Return the range of each column in the posterior file """
    return results.posterior_sample.ptp(axis=0)


def _columns_with_dynamic_range(results):
    """ Return the columns in the posterior file which vary """
    dr = _column_dynamic_ranges(results)
    return np.nonzero(dr)[0]



def hpd_grid(sample, alpha=0.05, bw_method=None, roundto=2):
    """Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI). The function
    works for multimodal distributions, returning more than one mode.

    Warning:
        This function is experimental.

    Args:
        sample (array):
            An array containing MCMC samples
        alpha (float, optional):
            Desired probability of type I error (defaults to 0.05)
        bw_method (float, optional):
            The method used to estimate the KDE bandwidth. See
            `scipy.stats.gaussian_kde`
        roundto (integer):
            Number of digits after the decimal point for the results
    """
    import scipy.stats.kde as kde
    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]
    # get upper and lower bounds
    l = np.min(sample)
    u = np.max(sample)
    print('doing KDE')
    density = kde.gaussian_kde(sample, bw_method)
    print('done')
    x = np.linspace(l, u, max(2000, sample.size//5))
    y = density.evaluate(x)
    #y = density.evaluate(x, l, u) waitting for PR to be accepted
    xy_zipped = zip(x, y/np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1-alpha):
            break
    hdv.sort()
    diff = (u-l)/100  # differences of 1%
    hpd = []
    hpd.append(round(min(hdv), roundto))
    for i in range(1, len(hdv)):
        if hdv[i]-hdv[i-1] >= diff:
            hpd.append(round(hdv[i-1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))
    ite = iter(hpd)
    hpd = list(zip(ite, ite))
    modes = []
    for value in hpd:
         x_hpd = x[(x > value[0]) & (x < value[1])]
         y_hpd = y[(x > value[0]) & (x < value[1])]
         modes.append(round(x_hpd[np.argmax(y_hpd)], roundto))
    return hpd, x, y, modes