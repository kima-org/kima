""" Small (but sometimes important) utility functions """

from datetime import datetime
import time
import os
import re
import contextlib
import math
import configparser
from warnings import WarningMessage

import numpy as np
from scipy import stats
from astropy.units import R_sun, AU

import urepr
from loguniform import ModifiedLogUniform
from kumaraswamy import kumaraswamy

# CONSTANTS
mjup2mearth = 317.8284065946748       # 1 Jupiter mass in Earth masses
mjup2msun = 0.0009545942339693249     # 1 Jupiter mass in solar masses
mearth2msun = 3.0034893488507934e-06  # 1 Earth mass in solar masses


def sounds(on=True):
    import chime
    chime.theme('material')
    import kima
    kima._SOUNDS_ = on

def sounds_off():
    sounds(False)

def maybe_success_sound():
    import kima
    if kima._SOUNDS_:
        import chime
        chime.success(sync=False)

def maybe_error_sound():
    import kima
    if kima._SOUNDS_:
        import chime
        chime.error(sync=False)

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def get_kima_dir():
    here = os.path.dirname(os.path.dirname(__file__))
    return here

def get_model_id(model, star=None, add_timestamp=True):
    from .. import RVmodel, GPmodel, RVFWHMmodel, RVFWHMRHKmodel
    if star is None:
        id = 'kima_'
    else: 
        id = star + '_'

    id += f'k{model.npmax}_' if model.fix else f'k0{model.npmax}_'
    id += f'd{model.degree}_' if model.trend else ''
    try:
        id += f'studentt_' if model.studentt else ''
    except AttributeError:
        pass
    id += 'GP_' if isinstance(model, GPmodel) else ''
    id += 'RVFWHM_' if isinstance(model, RVFWHMmodel) else ''
    id += 'RVFWHMRHK_' if isinstance(model, RVFWHMRHKmodel) else ''
    id += f'KO{model.n_known_object}_' if model.known_object else ''
    id += f'TR{model.n_transiting_planet}_' if model.transiting_planet else ''
    if add_timestamp:
        id += get_timestamp()
    if id.endswith('_'):
        id = id[:-1]
    return id

def read_model_setup(filename='kima_model_setup.txt'):
    setup = configparser.ConfigParser()
    setup.optionxform = str
    setup.read(filename)
    # if sys.version_info < (3, 0):
    #     setup = setup._sections
    #     # because we cheated, we need to cheat a bit more...
    #     setup['kima']['GP'] = setup['kima'].pop('gp')
    return setup


def read_datafile(datafile, skip):
    """
    Read data from `datafile` for multiple instruments.
    Can be str, in which case the last column is assumed to contain an integer
    identifier of the instrument.
    Or list, in which case each element will be one different filename
    containing three columns each.
    """
    if isinstance(datafile, list):
        data = np.empty((0, 3))
        obs = np.empty((0, ))
        for i, df in enumerate(datafile):
            d = np.loadtxt(df, usecols=(0, 1, 2), skiprows=skip, ndmin=2)
            data = np.append(data, d, axis=0)
            obs = np.append(obs, (i + 1) * np.ones((d.shape[0])))
        return data, obs
    else:
        data = np.loadtxt(datafile, usecols=(0, 1, 2), skiprows=skip)
        obs = np.loadtxt(datafile, usecols=(-1, ), skiprows=skip, dtype=int)
        uobs = np.unique(obs)
        if uobs.min() > 0:
            uobs -= uobs.min()
        return data, obs


def read_datafile_rvfwhm(datafile, skip):
    """
    Read data from `datafile` for multiple instruments and RV-FWHM data.
    """
    if isinstance(datafile, list):
        data = np.empty((0, 5))
        obs = np.empty((0, ))
        for i, df in enumerate(datafile):
            d = np.loadtxt(df, usecols=range(5), skiprows=skip, ndmin=2)
            data = np.append(data, d, axis=0)
            obs = np.append(obs, (i + 1) * np.ones((d.shape[0])))
        return data, obs
    else:
        raise NotImplementedError

def read_datafile_rvfwhmrhk(datafile, skip):
    """
    Read data from `datafile` for multiple instruments and RV-FWHM-RHK data.
    """
    if isinstance(datafile, list):
        data = np.empty((0, 7))
        obs = np.empty((0, ))
        for i, df in enumerate(datafile):
            d = np.loadtxt(df, usecols=range(7), skiprows=skip, ndmin=2)
            data = np.append(data, d, axis=0)
            obs = np.append(obs, (i + 1) * np.ones((d.shape[0])))
        return data, obs
    else:
        raise NotImplementedError


@contextlib.contextmanager
def hide_stdout():
    """
    A simple context manager to hide stdout temporarily

    Examples:
        ```python
        with hide_stdout():
            print('hello') # no output
        ```
    
    """
    from io import StringIO
    hidden = StringIO()
    try:
        with contextlib.redirect_stdout(hidden):
            yield
    finally:
        pass


@contextlib.contextmanager
def chdir(dir, create=True):
    """ A simple context manager to switch directories temporarily """
    curdir = os.getcwd()
    if not os.path.isdir(dir):
        if create:
            os.makedirs(dir)
        else:
            raise FileNotFoundError(f'directory does not exist: {dir}')
    try:
        os.chdir(dir)
        yield
    finally:
        os.chdir(curdir)


def show_tips():
    """ Show a few tips on how to use kima (but only sometimes) """
    tips = (
        "Press Ctrl+C in any of kima's plots to copy the figure.",
        "Run 'kima-showresults all' to plot every figure.",
        "Use the 'kima-template' script to create a new bare-bones directory.")
    if np.random.rand() < 0.2:  # only sometimes, otherwise it's annoying :)
        tip = np.random.choice(tips)
        print('[kima TIP] ' + tip)


def _show_kima_setup():
    def isnotebook():
        try:
            shell = get_ipython().__class__.__name__
            return shell == 'ZMQInteractiveShell'  # notebook or qtconsole
        except NameError:
            return False
    if isnotebook():
        from pygments import highlight
        from pygments.lexers import CppLexer
        from pygments.formatters import HtmlFormatter
        import IPython

        with open('kima_setup.cpp') as f:
            code = f.read()

        formatter = HtmlFormatter()
        return IPython.display.HTML(
            '<style type="text/css">{}</style>{}'.format(
                formatter.get_style_defs('.highlight'),
                highlight(code, CppLexer(), formatter)))
    else:
        with open('kima_setup.cpp') as f:
            print(f.read())


class Interval:
    def __init__(self, low, high):
        self.low = low
        self.high = high
    
    @classmethod
    def from_array(cls, arr: np.ndarray):
        return cls(arr.min(), arr.max())

    def __repr__(self):
        return f'Interval({self.low}, {self.high})'

    def __contains__(self, x):
        if isinstance(x, np.ndarray):
            return (self.low <= x).any() and (x <= self.high).any()
        return self.low <= x <= self.high
    
    def mask(self, arr: np.ndarray):
        return (self.low <= arr) & (arr <= self.high)


def rms(array):
    """ Root mean square of array """
    return np.sqrt(np.sum(array**2) / array.size)


def wrms(array, weights):
    """ Weighted root mean square of array, given weights """
    mu = np.average(array, weights=weights)
    return np.sqrt(np.sum(weights * (array - mu)**2) / np.sum(weights))


def apply_argsort(arr1, arr2, axis=-1):
    """
    Apply `arr1`.argsort() on `arr2`, along `axis`.
    """
    # check matching shapes
    assert arr1.shape == arr2.shape, "Shapes don't match!"

    i = list(np.ogrid[[slice(x) for x in arr1.shape]])
    i[axis] = arr1.argsort(axis)
    return arr2[i]


def percentile68_ranges(a, min=None, max=None):
    """
    Calculate the 16th and 84th percentiles of values in `a`, clipped between
    `min` and `max`.

             minus     median    plus     
        -------==========|==========------
        16%    |        68%        |    16%

    Returns:
        median (float):
            Median value of `a`.
        plus (float):
            The 84th percentile minus the median.
        minus (float):
            The median minus the 16th percentile.
    """
    if min is None and max is None:
        mask = np.ones_like(a, dtype=bool)
    elif min is None:
        mask = a < max
    elif max is None:
        mask = a > min
    else:
        mask = (a > min) & (a < max)
    lp, median, up = np.percentile(a[mask], [16, 50, 84])
    return (median, up - median, median - lp)


def percentile68_ranges_latex(a, min=None, max=None, collapse=True,
                              include_dollar=True):
    r"""
    Return a LaTeX-formatted string of the 68% range of values in `a`, clipped
    between `a` and `b`, in the form

    $ median ^{+ plus} _{- minus} $
    
    Args:
        collapse (bool, optional):
            If True and plus=minus, return $ median \pm plus $
        include_dollar (bool, optional):
            Whether to include dollar signs in the output, so it's a valid LaTeX
            math mode string
    """
    median, plus, minus = percentile68_ranges(a, min, max)
    dollar = '$' if include_dollar else ''
    if collapse:
        high = urepr.uformat(median, plus, 'L').split(r'\pm')[1]
        low = urepr.uformat(median, minus, 'L').split(r'\pm')[1]
        if high == low:
            return dollar + urepr.uformat(median, plus, 'L') + dollar
    return dollar + urepr.uformatul(median, plus, minus, 'L') + dollar


def percentile_ranges(a, percentile=68, min=None, max=None):
    """
    Calculate the given percentile range of values in `a`, clipped between `min`
    and `max`.

    Returns:
        median (float):
            Median value of `a`.
        plus (float):
            The 84th percentile minus the median.
        minus (float):
            The median minus the 16th percentile.
    """
    if min is None and max is None:
        mask = np.ones_like(a, dtype=bool)
    elif min is None:
        mask = a < max
    elif max is None:
        mask = a > min
    else:
        mask = (a > min) & (a < max)
    half = percentile / 2
    lp, median, up = np.percentile(a[mask], [50 - half, 50, 50 + half])
    return (median, up - median, median - lp)


def percentile_ranges_latex(a, percentile, min=None, max=None):
    median, plus, minus = percentile_ranges(a, percentile, min, max)
    return '$' + urepr.core.uformatul(median, plus, minus, 'L') + '$'


def clipped_mean(arr, min, max):
    """ Mean of `arr` between `min` and `max` """
    mask = (arr > min) & (arr < max)
    return np.mean(arr[mask])


def clipped_std(arr, min, max):
    """ Standard deviation of `arr` between `min` and `max` """
    mask = (arr > min) & (arr < max)
    return np.std(arr[mask])


def distribution_rvs(dist, size=1):
    """
    Generate random samples from a distribution.
    
    Args:
        dist (kima.distributions.Distribution or scipy.stats.rv_continuous):
            The distribution to sample from
        size (int, optional):
            Number of samples to generate
    """
    if hasattr(dist, 'rvs'):
        try:
            return dist.rvs(size)
        except TypeError:
            return np.array([dist.rvs() for _ in range(size)])
    u = np.random.rand(size)
    return np.array([dist.ppf(uu) for uu in u])


def distribution_support(dist):
    """ Return the support of a distribution """
    return dist.ppf(0.0), dist.ppf(1.0)


def find_constrained_invgamma(low, upp, low_tail=0.01, upp_tail=0.01):
    """ 
    Find the parameters of the inverse gamma distribution that is constrained
    between `low` and `upp` with tail probabilities `low_tail` and `upp_tail`.
    """
    from scipy.stats import invgamma
    from scipy.optimize import root
    def tail_condition(theta, low, upp, low_tail, upp_tail):
        return [
            invgamma(a=theta[0], scale=theta[1]).cdf(low) - low_tail,
            invgamma(a=theta[0], scale=theta[1]).cdf(upp) - (1 - upp_tail),
        ]
    def guess(low, upp, delta=0.75):
        alpha_guess = (delta * (low + upp) / (upp - low))**2 + 2
        beta_guess = (alpha_guess - 1.0) * 0.5 * (low + upp)
        return [alpha_guess, beta_guess]
    result = root(tail_condition, x0=guess(low, upp), args=(low, upp, low_tail, upp_tail), method='lm')
    if result.success:
        return result.x
    else:
        import warnings
        warnings.warn('Could not find constrained inverse gamma distribution')
        return result



def get_gaussian_prior_vsys(data, use_ptp=True, use_std=False, use_value=None):
    """
    Get an informative Gaussian prior for the systemic velocity using the data.

    Args:
        data (kima.RVData): 
            The RV dataset 
        use_ptp (bool, optional):
            Width of the prior is the range of the data.
        use_std (bool, optional):
            Width of the prior is the standard deviation of the data.
        use_value (float, optional):
            Width of the prior is provided directly.

    Returns:
        prior (kima.distributions.Gaussian):
            A Gaussian prior for the systemic velocity

    Raises:
        ValueError:
            If _both_ `use_ptp` and `use_std` are True
    """
    from kima.distributions import Gaussian
    obsi = np.array(data.obsi)
    u_obsi = np.unique(obsi)
    y = np.array(data.y)[obsi == u_obsi[-1]]
    if use_value is not None:
        assert isinstance(use_value, (float, int)), '`use_value` should be a float'
        return Gaussian(np.median(y), use_value)
    if use_ptp and not use_std:
        return Gaussian(np.median(y), np.ptp(y))
    elif use_std:
        return Gaussian(np.median(y), np.std(y))
    else:
        raise ValueError('either `use_ptp` or `use_std` should be True')


def get_gaussian_priors_individual_offsets(data, use_ptp=True, use_std=False, use_value=None):
    """
    Get a informative Gaussian priors for the between-instrument offsets using
    the data.

    Args:
        data (kima.RVData): 
            The RV dataset (must have multiple instruments)
        use_ptp (bool, optional):
            Width of the prior is the range of the data.
        use_std (bool, optional):
            Width of the prior is the standard deviation of the data.
        use_value (float, optional):
            Width of the prior is provided directly.

    Returns:
        prior (kima.distributions.Gaussian):
            Gaussian priors appropriate for the between-instrument offsets

    Raises:
        ValueError:
            - If `data.multi` is False
            - If _both_ `use_ptp` and `use_std` are True
    """
    from kima.distributions import Gaussian
    if not data.multi:
        raise ValueError('data is not from multiple instruments')
    loc_scale = []
    obsi = np.array(data.obsi)
    u_obsi = np.unique(obsi)
    y_last = np.array(data.y)[obsi == u_obsi[-1]]
    for i in u_obsi[::-1][1:]:
        y = np.array(data.y)[obsi == i]
        if use_value is not None:
            assert isinstance(use_value, (float, int)), '`use_value` should be a float'
            loc_scale.append((np.median(y) - np.median(y_last), use_value))
            continue
        if use_ptp and not use_std:
            loc_scale.append((np.median(y) - np.median(y_last), np.ptp(y)))
        elif use_std:
            loc_scale.append((np.median(y) - np.median(y_last), np.std(y)))
        else:
            raise ValueError('either `use_ptp` or `use_std` should be True')
    return [Gaussian(loc, scale) for loc, scale in loc_scale[::-1]]


def get_prior_monotransits(T0_1, T0_2, T0_1_err=0.1, T0_2_err=0.1, lower_limit=1.0):
    from kima.distributions import GaussianMixture
    periods = [T0_2 - T0_1]
    errors = [float(np.hypot(T0_1_err, T0_2_err))]
    i = 2
    while (periods[0] / i) > lower_limit:
        periods.append(periods[0] / i)
        errors.append(errors[0] / i)
        i += 1
    return GaussianMixture(periods, errors, lower_limit, np.inf)




# def get_planet_mass_latex(P, K, e, units='earth', **kwargs):
#     """ A simple wrapper around `get_planet_mass` to provide LaTeX strings

#     Args:
#         P : Same argument as for `get_planet_mass`
#         K : Same argument as for `get_planet_mass`
#         e : Same argument as for `get_planet_mass`
#         units (str, optional): Planet mass units, either 'earth' or 'jup'
#         **kwargs : Provided directly to `get_planet_mass`

#     Returns:
#         s (str): LaTeX string with planet mass and possibly uncertainty
#     """
#     out = get_planet_mass(P, K, e, **kwargs)

#     if isinstance(P, float):
#         if units.lower() == 'earth':
#             return '$%f$' % out[1]
#         else:
#             return '$%f$' % out[0]
#     else:
#         if units.lower() == 'earth':
#             return percentile68_ranges_latex(out[2] * mjup2mearth)
#         else:
#             return percentile68_ranges_latex(out[2])


# def get_planet_semimajor_axis_latex(P, K, **kwargs):
#     """
#     A simple wrapper around `get_planet_semimajor_axis` to provide LaTeX strings

#     Args:
#         P : Same argument as for `get_planet_semimajor_axis`
#         K : Same argument as for `get_planet_semimajor_axis`
#         **kwargs : Provided directly to `get_planet_semimajor_axis`

#     Returns:
#         s (str): LaTeX string with planet semi-major axis
#     """
#     out = get_planet_semimajor_axis(P, K, **kwargs)
#     if isinstance(P, float):
#         return '$%f$' % out
#     else:
#         return '$%f$' % out[0]


def get_planet_teq(Tstar: float = 5777, Rstar: float = 1, a: float = 1,
                   A: float = 0, f: float = 1):
    """ Calculate the planet's equlibrium temperature

    Args:
        Tstar (float, optional): Stellar effective temperature [K]. Defaults to 5777.
        Rstar (float, optional): Stellar radius [Rsun]. Defaults to 1.0.
        a (float, optional): Planet semi-major axis [AU]. Defaults to 1.0.
        A (float, optional): Bond albedo. Defaults to 0.
        f (float, optional): Redistribution factor. Defaults to 1.

    Returns:
        Teq (float): Planet equiolibrium temperature [K]
    """
    a = a * AU.to('km')
    Rstar = 1 * R_sun.to('km')
    Teq = Tstar * (f * (1 - A))**(1 / 4) * (Rstar / (2 * a))**(1 / 2)
    return Teq


def get_transit_probability(Rstar=1.0, a=1.0):
    """
    Transit probability, simple. Eq 6.1 in Exoplanet Handbook
    Rstar: stellar radius [Rsun]
    a: semi-major axis [au]
    """
    return Rstar / (a * AU).to(R_sun).value


def get_mean_longitude(M0, w, fold=False, deg=False):
    """ 
    Return the mean longitude λ for a given mean anomaly M0 and longitude of
    periastron w (given by λ = M0 + w).

    Args:
        M0 (float or ndarray):
            Mean anomaly (M0=0 at periastron) [rad]
        w (float or ndarray):
            Longitude of periastron [rad]
        fold (bool, optional):
            Fold the mean longitude into the range [-π, π) [default: False]
        deg (bool, optional):
            Return mean longitude in degrees [default: False]

    Returns:
        λ (float or ndarray):
            Mean longitude [rad or deg]
    """
    λ = M0 + w
    if fold:
        λ = np.angle(np.exp(1j * λ))
    if deg:
        λ = np.rad2deg(λ)
    return λ

def mean_anomaly_from_time_periastron(t, P, Tp, deg=False):
    """ 
    Calculate mean anomaly for times t, given period P and time of periastron Tp.

    Args:
        t (float, ndarray): Times at which to calculate mean anomaly [days]
        P (float): Orbital period [days]
        Tp (float): Time of periastron [days]
        deg (bool, optional): Return mean anomaly in degrees [default: False]

    Returns:
        M (float, ndarray): Mean anomaly [rad or deg]
    """
    M = 2 * np.pi * (t - Tp) / P
    if deg:
        M = np.rad2deg(M)
    return M

def mean_anomaly_from_epoch(t, P, M0, epoch):
    """
    Calculate mean anomaly for times t, given period P, mean anomaly at the
    epoch M0, and epoch.

    Args:
        t (float, ndarray): Times at which to calculate mean anomaly [days]
        P (float): Orbital period [days]
        M0 (float): Mean anomaly at the epoch [rad]
        epoch (float): Epoch [days]

    Returns:
        M (float, ndarray): Mean anomaly [rad]
    """
    return 2 * np.pi * (t - epoch) / P + M0


def lighten_color(color, amount=0.5):
    """
    Lightens the given `color` by multiplying (1-luminosity) by the given
    `amount`. Input can be a matplotlib color string, hex string, or RGB tuple.

    Examples:
        ```python
        lighten_color('g', 0.3)
        lighten_color('#F034A3', 0.6)
        lighten_color((.3, .55, .1), 0.5)
        ```
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


class Fixed:
    """ Not a real distribution, just a parameter fixed at a given `value` """
    def __init__(self, value: float):
        self.value = value

    def rvs(self) -> float:
        """ Random sample (always returns `value`)"""
        return self.value

    def pdf(self, x: float) -> float:
        """ Probability density function: 1.0 if x==value, otherwise 0.0"""
        return 1.0 if x == self.value else 0.0

    def logpdf(self, x: float) -> float:
        """ Logarithm of the probability density function """
        return 0.0 if x == self.value else -np.inf


class ZeroDist:
    """ A dummy probability distribution which always returns 0.0 """
    def __init__(self):
        pass

    def rvs(self) -> float:
        return 0.0

    def pdf(self, x: float) -> float:
        return 0.0

    def logpdf(self, x: float) -> float:
        return 0.0


def _prior_to_dist():
    """ Convert a prior name to a prior object """
    d = {
        'Uniform': stats.uniform,
        'LogUniform': stats.reciprocal,
        'ModifiedLogUniform': ModifiedLogUniform,
        'Gaussian': stats.norm,
        'TruncatedGaussian': stats.truncnorm,
        'Exponential': stats.expon,
        'Kumaraswamy': kumaraswamy,
        'Laplace': stats.laplace,
        'Cauchy': stats.cauchy,
        'InvGamma': lambda shape, scale: stats.invgamma(shape, scale=scale),
        'Fixed': Fixed,
    }
    return d


def _get_prior_parts(prior):
    if not isinstance(prior, str):
        raise ValueError('`prior` should be a string, got', prior)

    try:
        inparens = re.search(r'\((.*?)\)', prior).group(1)
    except AttributeError:
        # print('Cannot decode "%s", seems badly formed' % prior)
        return '', '', prior
    try:
        truncs = re.search(r'\[(.*?)\]', prior).group(1)
    except AttributeError:
        truncs = ''

    name = prior[:prior.find('(')]

    return inparens, truncs, name


def find_prior_limits(prior):
    """
    Find lower and upper limits of a prior from the kima_model_setup.txt file.
    """
    inparens, truncs, name = _get_prior_parts(prior)
    # print(inparens, truncs, name)

    if 'Truncated' in name:
        return tuple(float(v) for v in truncs.split(','))

    if name == 'ModifiedLogUniform':
        return (0.0, float(inparens.split(';')[1]))

    if name == 'Uniform':
        v1, v2 = inparens.split(';')
        return (float(v1), float(v2) - float(v1))

    if name == 'LogUniform':
        return tuple(float(v) for v in inparens.split(';'))

    if name == 'Gaussian':
        return (-np.inf, np.inf)

    if name == 'InvGamma':
        return (0, np.inf)

    if name == 'Kumaraswamy':
        return (0.0, 1.0)


def find_prior_parameters(prior):
    """ Find parameters of a prior from the kima_model_setup.txt file. """
    inparens, _, name = _get_prior_parts(prior)

    truncated = False
    if 'Truncated' in name:
        inparens = ';'.join(inparens.split(';')[:-1])
        name = name.replace('Truncated', '')
        truncated = True

    twopars = ('LogUniform', 'ModifiedLogUniform', 'Gaussian', 'Kumaraswamy',
               'Cauchy', 'InvGamma')

    if name in twopars:
        r = [float(v) for v in inparens.split(';')]
    elif name == 'Uniform':
        v1, v2 = inparens.split(';')
        r = [float(v1), float(v2) - float(v1)]
    elif name in ('Exponential', 'Fixed'):
        r = [float(inparens), ]
    else:
        r = [
            np.nan,
        ]

    if truncated:
        r.append('T')

    return tuple(r)


def get_prior(prior):
    """ Return a scipt.stats-like prior from a kima_model_setup.txt prior """
    _, _, name = _get_prior_parts(prior)
    pars = find_prior_parameters(prior)

    if 'T' in pars:
        a, b = find_prior_limits(prior)
        loc, scale, _ = pars
        a, b = (a - loc) / scale, (b - loc) / scale

    try:
        d = _prior_to_dist()
        if 'T' in pars:
            return d[name](a=a, b=b, loc=loc, scale=scale)
        else:
            return d[name](*pars)
    except KeyError:
        return None


def hyperprior_samples(size):
    pi, log, rng = np.pi, np.log, np.random
    U = rng.rand(size)
    muP = 5.901 + np.tan(pi * (0.97 * rng.rand(size) - 0.485))
    wP = 0.1 + 2.9 * rng.rand(size)
    P = np.where(U < 0.5, muP + wP * log(2. * U), muP - wP * log(2. - 2. * U))
    return np.exp(P)


def priors_latex(results, title='Priors', label='tab:1'):
    priors = results.priors

    tab1 = rf"""\\begin{{table}}
    \r\caption{{{title}}}
    \r\label{{{label}}}
    \r\centering
    \r\\begin{{tabular}}{{l l c}}
    \r  \hline\hline
    \r  Parameter & Units & Prior \\\\
    \r  \hline
    """

    tab2 = r"""  \hline
    \r\end{tabular}
    \r\end{table}
    """

    print(tab1)

    parameter_names = dict(
        Vprior=r'$v_{\rm sys}$',
        C2prior=r'$$',
        Jprior=r'$j^{\rm RV}$',
        J2prior=r'$j^{\rm FWHM}$',
        #
        Pprior='$P$',
        Kprior='$K$',
        eprior='$e$',
        wprior=r'$\omega$',
        phiprior='$M_0$',
        #
        eta1_prior=r'$\eta_1$',
        eta1_1_prior=r'$\eta_1$ RV',
        eta1_2_prior=r'$\eta_1$ FWHM',
        eta2_1_prior=r'$\eta_2$',
        eta3_1_prior=r'$\eta_3$',
        eta4_1_prior=r'$\eta_4$'
    )

    units = dict(
        Vprior=r'\ms',
        C2prior=r'\ms',
        Jprior=r'\ms',
        J2prior=r'\ms',
        #
        Pprior='days',
        Kprior=r'\ms',
        #
        eta1_prior=r'\ms',
        eta1_1_prior=r'\ms',
        eta1_2_prior=r'\ms',
        eta2_1_prior='days',
        eta3_1_prior='days',
    )

    dist_symbols = {
        'uniform': r'$\mathcal{U}$',
        'reciprocal': r'$\mathcal{L}\mathcal{U}$',
        'ModifiedLogUniform': r'$\mathcal{M}\mathcal{L}\mathcal{U}$',
    }
    translation = {
        np.pi: r'$\pi$',
        2 * np.pi: r'$2\pi$',
        results.y.min(): r'$\min v$',
        results.y.max(): r'$\max v$',
        np.ptp(results.t): r'$\Delta t$',
        np.ptp(results.y): r'$\Delta RV$',
        -np.ptp(results.y): r'$-\Delta RV$'
    }
    if results.model == 'RVFWHMmodel':
        translation.update({
            results.y2.min(): r'$\min$ FWHM',
            results.y2.max(): r'$\max$ FWHM',
            np.ptp(results.y2): r'$\Delta$ FWHM',
            -np.ptp(results.y2): r'$-\Delta$ FWHM'
        })

    def translate_value(value):
        for v, name in translation.items():
            if np.isclose(value, v):
                return name
        return str(value).replace('.0', '')

    for k, prior in priors.items():
        try:
            name = prior.dist.name
        except AttributeError:
            name = prior.__class__.__name__

        try:
            v1, v2 = prior.support()
            v1 = translate_value(v1)
            v2 = translate_value(v2)
            value = ', '.join([v1, v2])
        except AttributeError:
            v = translate_value(prior.val)
            value = v + ' (fixed)'

        s = '  '\
            f'{parameter_names.get(k, k)} & ' \
            f'{units.get(k, "")} & ' \
            f'{dist_symbols.get(name, "")} ({value}) \\\\'
        print(s)
    print()

    print(tab2)


def get_star_name(data_file):
    """ Find star name (usually works for approx. standard filenames) """
    bn = os.path.basename(data_file)
    try:
        pattern = '|'.join([
            r'HD\d+',
            r'HIP\d+',
            r'HR\d+',
            r'BD-\d+',
            r'CD-\d+',
            r'NGC\d+No\d+',
            r'GJ[\d.]+',
            r'Gl\d+',
            r'Proxima',
            r'Barnard',
            r'Ross128',
        ])
        return re.findall(pattern, bn)[0]

    except IndexError:  # didn't find correct name
        if bn.endswith('_harps.rdb'):
            return bn[:-10]

    return 'unknown'


def get_instrument_name(data_file):
    """ Try to find the instrument name from a filename """
    bn = os.path.basename(data_file)
    try:
        pattern = '|'.join([
            # 'ESPRESSO',
            # 'ESPRESSO*[\d+]*',
            r'ESPRESSO*[\d+_\w+]*',
            r'HARPS*[\d+_\w+]*',
            r'HIRES[^\W_]*[\d+]*',
            r'APF',
            r'CORALIE[\d+]*',
            r'HJS',
            r'ELODIE',
            r'KECK',
            r'HET',
            r'LICK',
            r'HRS',
            r'SOPHIE',
            r'HAMILTON',
            r'NIRPS',
            r'PFS[\d+]',
            r'UCLES',
            r'MIKE',
        ])
        return re.findall(pattern, bn, re.IGNORECASE)[0]

    except IndexError:
        try:
            return re.findall(pattern, bn.upper())[0]
        except IndexError:
            try:  # at the very least, try removing the file type
                return os.path.splitext(bn)[0]
            except IndexError:  # couldn't do anything, just try basename
                return bn


def read_big_file(filename, names=None):
    # return np.genfromtxt(filename)
    # if cache and os.path.exists(filename + '.npy'):
    #     return np.load(filename + '.npy')

    try:
        import pandas as pd
        if names is None:
            names = open(filename).readline().strip().replace('#', '').split()
        try:
            data = pd.read_csv(filename, sep=r'\s+', comment='#',
                               names=names, dtype=float).values
        except pd.errors.ParserError:
            raise ValueError

        # pandas.read_csv has problems with 1-line files
        if data.shape[0] == 0:
            data = np.genfromtxt(filename)

    # no pandas or some other issue, use np.genfromtxt
    except (ImportError, ValueError):  
        data = np.genfromtxt(filename)

    # if cache:
    #     np.save(filename + '.npy', data)

    return data



def kima_to_george_GP(η1, η2, η3, η4, η5=None, η6=None, η7=None, build_kernel=True):
    _1, _2, _3, _4 = η1**2, η2**2, np.log(η3), 2/η4**2
    if η5 is not None:
        _5, _6, _7 = η5, np.log(η6), 2/η7**2

    if build_kernel:
        from george.kernels import ExpSquaredKernel, ExpSine2Kernel
        if η5 is None:
            return _1 * ExpSquaredKernel(_2) * ExpSine2Kernel(_4, _3)
        else:
            return _1 * ExpSquaredKernel(_2) * ExpSine2Kernel(_4, _3) + _5 * ExpSine2Kernel(_7, _6)
    else:
        if η5 is None:
            return _1, _2, _3, _4
        else:
            return _1, _2, _3, _4, _5, _6, _7


def get_timestamp():
    return datetime.now().strftime('%Y_%m_%dT%H_%M_%S')

# covert dates to/from Julian days and MJD
# originally from https://gist.github.com/jiffyclub/1294443
# author: Matt Davis (http://github.com/jiffyclub)


def date_mjd_to_jd(mjd):
    """ Convert Modified Julian Day to Julian Day.

    Args:
        mjd (float): Modified Julian Day

    Returns
        jd (float): Julian Day
    """
    return mjd + 2400000.5


def date_jd_to_mjd(jd):
    """ Convert Julian Day to Modified Julian Day

    Args:
        jd (float): Julian Day

    Returns
        mjd (float): Modified Julian Day
    """
    return jd - 2400000.5


def date_date_to_jd(year, month, day):
    """ Convert a date (year, month, day) to Julian Day.

    Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet',
        4th ed., Duffet-Smith and Zwart, 2011.

    Args:
        year (int): 
            Year as integer. Years preceding 1 A.D. should be 0 or negative. The
            year before 1 A.D. is 0, 10 B.C. is year -9.
        month (int): Month as integer, Jan = 1, Feb. = 2, etc.
        day (float): Day, may contain fractional part.

    Returns:
        jd (float): Julian Day

    Examples:
        Convert 6 a.m., February 17, 1985 to Julian Day
        >>> date_to_jd(1985, 2, 17.25)
        2446113.75
    """
    if month == 1 or month == 2:
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month

    # this checks where we are in relation to October 15, 1582, the beginning
    # of the Gregorian calendar.
    if (year < 1582 or (year == 1582 and month < 10)
            or (year == 1582 and month == 10 and day < 15)):
        # before start of Gregorian calendar
        B = 0
    else:
        # after start of Gregorian calendar
        A = math.trunc(yearp / 100.)
        B = 2 - A + math.trunc(A / 4.)

    if yearp < 0:
        C = math.trunc((365.25 * yearp) - 0.75)
    else:
        C = math.trunc(365.25 * yearp)

    D = math.trunc(30.6001 * (monthp + 1))

    jd = B + C + D + day + 1720994.5

    return jd


def date_jd_to_date(jd):
    """ Convert Julian Day to date.

    Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet',
        4th ed., Duffet-Smith and Zwart, 2011.

    Args:
        jd (float): Julian Day

    Returns:
        year (int): 
            Year as integer. Years preceding 1 A.D. should be 0 or negative. The
            year before 1 A.D. is 0, 10 B.C. is year -9.
        month (int): Month as integer, Jan = 1, Feb. = 2, etc.
        day (float): Day, may contain fractional part.

    Examples:
        Convert Julian Day 2446113.75 to year, month, and day.
        >>> jd_to_date(2446113.75)
        (1985, 2, 17.25)
    """
    jd = jd + 0.5

    frac, integ = math.modf(jd)
    integ = int(integ)

    A = math.trunc((integ - 1867216.25) / 36524.25)

    if integ > 2299160:
        B = integ + 1 + A - math.trunc(A / 4.)
    else:
        B = integ

    C = B + 1524

    D = math.trunc((C - 122.1) / 365.25)

    E = math.trunc(365.25 * D)

    G = math.trunc((C - E) / 30.6001)

    day = C - E + frac - math.trunc(30.6001 * G)

    if G < 13.5:
        month = G - 1
    else:
        month = G - 13

    if month > 2.5:
        year = D - 4716
    else:
        year = D - 4715

    return year, month, day


def date_hmsm_to_days(hour=0, min=0, sec=0, micro=0):
    """
    Convert hours, minutes, seconds, and microseconds to fractional days.

    Args:
        hour (int, optional): Hour number. Defaults to 0.
        min (int, optional): Minute number. Defaults to 0.
        sec (int, optional): Second number. Defaults to 0.
        micro (int, optional): Microsecond number. Defaults to 0.

    Returns:
        days (float): Fractional days

    Examples:
        >>> hmsm_to_days(hour=6)
        0.25
    """
    days = sec + (micro / 1.e6)

    days = min + (days / 60.)

    days = hour + (days / 60.)

    return days / 24.


def date_days_to_hmsm(days):
    """
    Convert fractional days to hours, minutes, seconds, and microseconds.
    Precision beyond microseconds is rounded to the nearest microsecond.

    Args:
        days (float): A fractional number of days. Must be less than 1.

    Returns:
        hour (int): Hour number
        min (int): Minute number
        sec (int): Second number
        micro (int): Microsecond number

    Raises:
        ValueError: If `days` is >= 1.

    Examples:
        >>> days_to_hmsm(0.1)
        (2, 24, 0, 0)
    """
    hours = days * 24.
    hours, hour = math.modf(hours)

    mins = hours * 60.
    mins, min = math.modf(mins)

    secs = mins * 60.
    secs, sec = math.modf(secs)

    micro = round(secs * 1.e6)

    return int(hour), int(min), int(sec), int(micro)


def date_datetime_to_jd(date: datetime):
    """ Convert a `datetime.datetime` object to Julian day.

    Args:
        date (datetime): the date to be converted to Julian day

    Returns:
        jd (float): Julian day

    Examples:
        >>> d = datetime.datetime(1985, 2, 17, 6)
        >>> d
        datetime.datetime(1985, 2, 17, 6, 0)
        >>> jdutil.datetime_to_jd(d)
        2446113.75
    """
    days = date.day + date_hmsm_to_days(date.hour, date.minute, date.second,
                                        date.microsecond)

    return date_date_to_jd(date.year, date.month, days)


# def styleit(func):
#     from matplotlib.pyplot import style
#     here = os.path.dirname(__file__)
#     def wrapper(*args, **kwargs):
#         with style.context([os.path.join(here, 'simple.mplstyle')]):
#             return func(*args, **kwargs)
#     return wrapper

def show_kima_setup(lang: str = 'cpp', output: str = 'html', prefix: str = ''):
    from textwrap import indent
    try:
        from pygments import highlight
        from pygments.lexers import PythonLexer, CppLexer
        from pygments.formatters import HtmlFormatter
        pyg = True
    except ImportError as e:
        # pyg = False
        # WarningMessage('pygments is not installed')
        raise e

    if pyg:
        if lang == 'cpp':
            lex = CppLexer
            code = open('kima_setup.cpp').read()
        elif lang == 'py':
            lex = PythonLexer
            code = open('kima_setup.py').read()

        if output == 'html':
            out = highlight(code, lex(), HtmlFormatter())

    out = indent(out, prefix)
    print(out)

# print(
#     '<style>\n' +
#     HtmlFormatter().get_style_defs('.highlight') +
#     '</style>\n'
# )
# print(
#     pass



class SimpleTimer:    
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

