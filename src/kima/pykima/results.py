"""
This module defines the `KimaResults` class to hold results from a run.
"""

from functools import lru_cache
import os
import sys
from typing import List, Union
from typing_extensions import Self
import zipfile
import tempfile
from string import ascii_lowercase
from dataclasses import dataclass, field
from io import StringIO
from contextlib import redirect_stdout
from copy import copy, deepcopy


from .. import __models__, MODELS
from ..kepler import keplerian
from ..postkepler import post_keplerian, post_keplerian_sb2, period_correction, a0fromK, Kfroma0
from kima import distributions
from .classic import postprocess
from .GP import (GP as GaussianProcess, ESPkernel, EXPkernel, MEPkernel, RBFkernel, Matern32kernel, SHOkernel, 
                 QPkernel, QPCkernel, PERkernel, QPpCkernel,
                 QPpMAGCYCLEkernel, mixtureGP)

from .analysis import get_planet_mass, get_planet_mass_GAIA, get_planet_mass_and_semimajor_axis, get_planet_semimajor_axis, np_bayes_factor_threshold
from .utils import (distribution_rvs, read_datafile, read_datafile_rvfwhm, read_datafile_rvfwhmrhk, read_model_setup,
                    get_star_name, mjup2mearth, get_instrument_name, SimpleTimer, get_timestamp,
                    _show_kima_setup, read_big_file, rms, wrms, chdir)

from . import display

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm, t as students_t, gaussian_kde, randint as discrete_uniform

try:  # only available in scipy 1.1.0
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None

pathjoin = os.path.join
colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

COMPRESSED_FILE_EXT = {
    "bz2": ".bz",
    "gzip": ".gz",
    "lz4": ".lz4",
    "lzma": ".lzma",
    "pickle": "",
    "zipfile": ".zip",
}



def _read_priors(res, setup=None):
    if setup is None:
        setup = read_model_setup()

    priors = list(setup['priors.general'].values())
    prior_names = list(setup['priors.general'].keys())
    try:
        for section in ('priors.planets', 'priors.hyperpriors', 'priors.GP'):
            try:
                priors += list(setup[section].values())
                prior_names += list(setup[section].keys())
            except KeyError:
                continue
    except KeyError:
        pass

    try:
        priors += list(setup['priors.known_object'].values())
        prior_names += ['KO_' + k for k in setup['priors.known_object'].keys()]
    except KeyError:
        pass

    try:
        priors += list(setup['priors.transiting_planet'].values())
        prior_names += ['TR_' + k for k in setup['priors.transiting_planet'].keys()]
    except KeyError:
        pass

    prior_dists = []
    for p in priors:
        p = p.replace(';', ',').replace('[', '').replace(']', '')
        p = p.replace('inf', 'np.inf')
        if 'UniformAngle' in p:
            p = 'UniformAngle()'
        prior_dists.append(eval('distributions.' + p))

    priors = {n: v for n, v in zip(prior_names, prior_dists)}
    # priors = {
    #     n: v
    #     for n, v in zip(prior_names, [get_prior(p) for p in priors])
    # }

    return priors


class named_array(np.ndarray):
    columns = []
    def set_columns(self, columns):
        self.columns = columns

    def __getattr__(self, name):
        if name in self.columns:
            return self[:, self.columns.index(name)]
        return super().__getattribute__(name)
    
    # def __repr__(self):
    #     if self.columns is not None:
    #         return f'{super().__repr__()}\n(attributes={self.columns})'
    #     return super(np.ndarray).__repr__()

@dataclass
class data_holder:
    """ A simple class to hold the RV datasets used in kima

    Attributes:
        t (ndarray): The observation times
        y (ndarray): The observed radial velocities
        e (ndarray): The radial velocity uncertainties
        obs (ndarray): Identifier for the instrument of each observation
        N (int): Total number of observations
    """
    t: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    e: np.ndarray = field(init=False)
    obs: np.ndarray = field(init=False)
    N: int = field(init=False)
    instrument: str = field(init=False)

    def __repr__(self):
        return f'data_holder(N={self.N}, t, y, e, obs)'

    @property
    def tmiddle(self):
        return self.t.min() + 0.5 * np.ptp(self.t)


@dataclass
class astrometric_data_holder:
    """ A simple class to hold the astrometric datasets used in kima

    Attributes:
        t (ndarray): The observation times
        w (ndarray): 
        sigw (ndarray): 
        psi (ndarray): 
        pf (ndarray): 
        N (int): Total number of observations
    """
    t: np.ndarray = field(init=False)
    w: np.ndarray = field(init=False)
    sigw: np.ndarray = field(init=False)
    psi: np.ndarray = field(init=False)
    pf: np.ndarray = field(init=False)
    N: int = field(init=False)
    instrument: str = 'Gaia'

    def __repr__(self):
        return f'data_holder(N={self.N}, t, w, sigw, psi, pf)'
    
@dataclass
class ETV_data_holder:
    """ A simple class to hold the ETV datasets used in kima

    Attributes:
        epoch (ndarray): The eclipse epochs
        et (ndarray): The observed eclipse times
        etsig (ndarray): The eclipse timing uncertainties
        N (int): Total number of observations
    """
    epoch: np.ndarray = field(init=False)
    et: np.ndarray = field(init=False)
    etsig: np.ndarray = field(init=False)
    N: int = field(init=False)
    instrument: str = 'eclipse_times'

    def __repr__(self):
        return f'data_holder(N={self.N}, epoch, et, etsig)'


@dataclass
class posterior_holder:
    """ A simple class to hold the posterior samples

    Attributes:
        P (ndarray): Orbital period(s)
        K (ndarray): Semi-amplitude(s)
        e (ndarray): Orbital eccentricities(s)
        w (ndarray): Argument(s) of pericenter
        φ (ndarray): Mean anomaly(ies) at the epoch
        --
        jitter (ndarray): Per-instrument jitter(s)
        stellar_jitter (ndarray): Global jitter
        offset (ndarray): Between-instrument offset(s)
        vsys (ndarray): Systemic velocity
        slope, quadr, cubic (ndarray): Trend parameters (up to 3rd degree)
        --
        η1 - η6 (ndarray): GP hyperparameters
        --
        TR: TRansiting planet parameters
        KO: Known Object parameters
    """
    Np: np.ndarray = field(init=False)
    P: np.ndarray = field(init=False)
    K: np.ndarray = field(init=False)
    e: np.ndarray = field(init=False)
    w: np.ndarray = field(init=False)
    φ: np.ndarray = field(init=False)
    Tc: np.ndarray = field(init=False)
    i: np.ndarray = field(init=False)
    Ω: np.ndarray = field(init=False)
    # 
    jitter: named_array = field(init=False)
    stellar_jitter: np.ndarray = field(init=False)
    offset: np.ndarray = field(init=False)
    vsys: np.ndarray = field(init=False)
    slope: np.ndarray = field(init=False)
    quadr: np.ndarray = field(init=False)
    cubic: np.ndarray = field(init=False)
    # 
    outlier_mean: np.ndarray = field(init=False)
    outlier_sigma: np.ndarray = field(init=False)
    outlier_Q: np.ndarray = field(init=False)
    #
    η1: np.ndarray = field(init=False)
    η2: np.ndarray = field(init=False)
    η3: np.ndarray = field(init=False)
    η4: np.ndarray = field(init=False)
    η5: np.ndarray = field(init=False)
    η6: np.ndarray = field(init=False)
    alpha: np.ndarray = field(init=False)
    beta: np.ndarray = field(init=False)
    # 
    TR: Self = field(init=False)
    KO: Self = field(init=False)

    def _get_set_fields(self):
        fields = list(self.__dataclass_fields__.keys())
        check_hasattr = lambda f: hasattr(self, f)
        check_isph = lambda f: isinstance(getattr(self, f), posterior_holder)
        check_size = lambda f: getattr(self, f).size > 0
        fields = [
            f for f in fields 
            if check_hasattr(f) and (check_isph(f) or check_size(f))
        ]
        return fields

    def __repr__(self):
        fields = self._get_set_fields()
        fields = ', '.join(fields)
        return f'posterior_holder({fields})'

    def _all(self):
        fields = self._get_set_fields()
        return np.hstack([getattr(self, f) for f in fields if f not in ('TR', 'KO')])
    
    def msini(self, star_mass=1.0, units=None,GAIA=False):
        """ planet minimum mass [Mjup by default], if from gaia data then true planet mass """
        allowed = ('mj', 'mjup', 'jupiter', 'jup', 'me', 'mearth', 'earth')
        if units and units.lower() not in allowed:
            raise ValueError(f'`units` must be one of {allowed}')
        units = units or 'mjup'
        if GAIA:
            m = get_planet_mass_GAIA(self.P, self.a0, self.plx, star_mass, full_output=True)[-1]
        else:
            m = get_planet_mass(self.P, self.K, self.e, star_mass, full_output=True)[-1]
        if units.lower() in ('me', 'mearth', 'earth'):
            m *= mjup2mearth
        return m

    def asini(self, star_mass=1.0, GAIA=False):
        """ planet semi-major axis [AU] """
        if GAIA:
            try:
                return self.a0/self.plx
            except ValueError:
                plx_new = self.plx[:, np.newaxis].copy()
                return self.a0/plx_new
        else:
            return get_planet_semimajor_axis(self.P, self.K, star_mass,
                                         full_output=True)[-1]
    
    def λ0(self, fold=True):
        """ mean longitude at the epoch [rad] """
        from .utils import get_mean_longitude
        return get_mean_longitude(self.φ, self.w, fold=fold)


def _get_pdf(prior, x=None, N=300):
    if x is None:
        if np.isneginf(_min := prior.ppf(0.0)):
            _min = prior.ppf(1e-6)
        if np.isinf(_max := prior.ppf(1.0)):
            _max = prior.ppf(1.0 - 1e-6)
        support = _max - _min
        x = np.linspace(_min - 0.1 * support, _max + 0.1 * support, N)
        return x, np.exp(np.vectorize(prior.logpdf)(x))
    return np.exp(np.vectorize(prior.logpdf)(x))


@dataclass
class prior_holder:
    """ A simple class to hold the priors

    Attributes:
        P (ndarray): Orbital period(s)
        K (ndarray): Semi-amplitude(s)
        e (ndarray): Orbital eccentricities(s)
        w (ndarray): Argument(s) of pericenter
        φ (ndarray): Mean anomaly(ies) at the epoch
        --
        jitter (ndarray): Per-instrument jitter(s)
        stellar_jitter (ndarray): Global jitter
        offset (ndarray): Between-instrument offset(s)
        vsys (ndarray): Systemic velocity
        slope, quadr, cubic (ndarray): Trend parameters (up to 3rd degree)
        --
        η1 - η6 (ndarray): GP hyperparameters
        --
        TR: TRansiting planet parameters
        KO: Known Object parameters
    """
    # Np: np.ndarray = field(init=False)
    # P: np.ndarray = field(init=False)
    # K: np.ndarray = field(init=False)
    # e: np.ndarray = field(init=False)
    # w: np.ndarray = field(init=False)
    # φ: np.ndarray = field(init=False)
    # Tc: np.ndarray = field(init=False)
    # 
    jitter: distributions.Distribution = field(init=False)
    # stellar_jitter: np.ndarray = field(init=False)
    # offset: np.ndarray = field(init=False)
    vsys: distributions.Distribution = field(init=False)
    # slope: np.ndarray = field(init=False)
    # quadr: np.ndarray = field(init=False)
    # cubic: np.ndarray = field(init=False)
    # 
    # outlier_mean: np.ndarray = field(init=False)
    # outlier_sigma: np.ndarray = field(init=False)
    # outlier_Q: np.ndarray = field(init=False)
    #
    η1: np.ndarray = field(init=False)
    η2: np.ndarray = field(init=False)
    η3: np.ndarray = field(init=False)
    η4: np.ndarray = field(init=False)
    η5: np.ndarray = field(init=False)
    η6: np.ndarray = field(init=False)
    # alpha: np.ndarray = field(init=False)
    # beta: np.ndarray = field(init=False)
    # 
    TR: Self = field(init=False)
    KO: Self = field(init=False)

    def _get_set_fields(self):
        fields = list(self.__dataclass_fields__.keys())
        check_hasattr = lambda f: hasattr(self, f)  # noqa: E731
        check_isph = lambda f: isinstance(getattr(self, f), prior_holder)  # noqa: E731
        check_dist = lambda f: isinstance(getattr(self, f), distributions.Distribution)  # noqa: E731
        fields = [
            f for f in fields 
            if check_hasattr(f) and (check_isph(f) or check_dist(f))
        ]
        return fields

    def __repr__(self):
        fields = self._get_set_fields()
        fields = ', '.join(fields)
        return f'prior_holder({fields})'

    def get_samples(self, field, N=1):
        prior = getattr(self, field)
        u = np.random.rand(N)
        return np.vectorize(prior.ppf)(u)

    def get_pdf(self, field, x=None):
        prior = getattr(self, field)
        return _get_pdf(prior, x)


def load_results(model_or_file, data=None, diagnostic=False, verbose=True,
                 moreSamples=1, n_resample_logX=1):
    """ Load the results from a kima run 

    Args:
        model_or_file (str or Model):
            If a string, load results from a pickle or zip file. If a model
            (e.g. `kima.RVmodel`), load results from that particular model and
            from the directory where it ran
        data (kima.RVData, optional):
            An instance of `kima.RVData` to use instead of `model.data`
            Warning: untested!
        diagnostic (bool, optional):
            Show the diagnostic plots
        verbose (bool, optional):
            Print some information about the results
        moreSamples (int, optional):
            The total number of posterior samples will be equal to 
            ESS * moreSamples
        n_resample_logX (int, optional):
            Number of times to resample the logX values in order to estimate the
            NS uncertainty in logZ, I, etc.

    Raises:
        FileNotFoundError:
            If `model_or_file` is a string and the file does not exist

    Returns:
        res (KimaResults):
            An instance of `KimaResults` holding the results of the run
    """
    # load from a pickle or zip file
    if isinstance(model_or_file, str):
        if not os.path.exists(model_or_file):
            raise FileNotFoundError(model_or_file)
        res = KimaResults.load(model_or_file)

    elif isinstance(model_or_file, __models__):
        if hasattr(model_or_file, 'directory') and model_or_file.directory != '':
            with chdir(model_or_file.directory):
                res = KimaResults(model_or_file, data,
                                  diagnostic=diagnostic, verbose=verbose,
                                  moreSamples=moreSamples, n_resample_logX=n_resample_logX)
        else:
            res = KimaResults(model_or_file, data,
                              diagnostic=diagnostic, verbose=verbose,
                              moreSamples=moreSamples, n_resample_logX=n_resample_logX)

    return res


class KimaResults:
    r""" A class to hold, analyse, and display the results from kima

    Attributes:
        model (str):
            The type of kima model
        priors (dict):
            A dictionary with the priors used in the model
        ESS (int):
            Effective sample size
        evidence (float):
            The log-evidence ($\ln Z$) of the model
        information (float):
            The Kullback-Leibler divergence between prior and posterior

        data (data_holder): The data
        posteriors (posterior_holder): The marginal posterior samples
    """

    data: data_holder
    model: str
    priors: dict
    GPmodel: bool

    evidence: float
    information: float
    ESS: int

    _star = None
    _debug = False

    def __init__(self, model, data=None, diagnostic=False, 
                 moreSamples=1, n_resample_logX=1, cache_files=True,
                 save_plots=False, return_figs=True, verbose=False, _debug=False):
        self.save_plots = save_plots
        self.return_figs = return_figs
        self.verbose = verbose
        self._debug = _debug

        self.setup = setup = read_model_setup()

        hidden = StringIO()
        stdout = sys.stdout if verbose else hidden

        with redirect_stdout(stdout):
            try:
                out = postprocess(plot=diagnostic, moreSamples=moreSamples,
                                  numResampleLogX=n_resample_logX)
                if diagnostic:
                    evidence, H, logx_samples, P_samples, figs = out
                else:
                    evidence, H, logx_samples, P_samples = out
            except FileNotFoundError as e:
                if e.filename == 'levels.txt':
                    msg = f'No levels.txt file found in {os.getcwd()}. Did you run the model?'
                    raise FileNotFoundError(msg) from None
                raise e
            except IndexError:
                raise ValueError('Something went wrong reading the posterior samples. Try again')

        self.model = MODELS(model.__class__.__name__)

        self.fix = model.fix
        self.npmax = model.npmax
        self.evidence = self.logZ = evidence
        self.information = self.KL = H

        with SimpleTimer() as timer:
            self.posterior_sample = np.atleast_2d(read_big_file('posterior_sample.txt'))
        if self._debug:
            print(f'Loading "posterior_sample.txt" took {timer.interval:.2f} seconds')

        self._ESS = self.posterior_sample.shape[0]

        #self.priors = {}
        self.priors = _read_priors(self, setup)


        ##### Issue here where data is assumed to be RV data, could do cases for each type, but what if there are multiple types?
        self.data_type = 'RV' #default to being RV data

        if self.model is MODELS.GAIAmodel:
            if data is None:
                data = model.data
            self.GAIAdata = astrometric_data_holder()
            self.data = self.GAIAdata
            self.GAIAdata.t = np.copy(data.t)
            self.GAIAdata.w = np.copy(data.w)
            self.GAIAdata.wsig = np.copy(data.wsig)
            self.GAIAdata.psi = np.copy(data.psi)
            self.GAIAdata.pf = np.copy(data.pf)
            self.GAIAdata.N = data.N
            self.data_type = 'GAIA'

        elif self.model is MODELS.ETVmodel:
            if data is None:
                data = model.data
            self.ETVdata = ETV_data_holder()
            self.data = self.ETVdata
            self.ETVdata.epochs = np.array(np.copy(data.t))
            self.ETVdata.et = np.array(np.copy(data.w))
            self.ETVdata.etsig = np.array(np.copy(data.wsig))
            self.ETVdata.N = data.N
            self.data_type = 'ETV'
        elif self.model is MODELS.RVGAIAmodel:
            if data is None:
                RV_data = model.RVdata
                data = RV_data
                GAIA_data = model.GAIAdata
            self.RVdata = data_holder()
            self.data = self.RVdata
            self.data.t = np.copy(RV_data.t)
            self.data.y = np.copy(RV_data.y)
            self.data.e = np.copy(RV_data.sig)
            self.data.obs = np.copy(RV_data.obsi)
            self.data.N = RV_data.N

            self.GAIAdata = astrometric_data_holder()
            self.GAIAdata.t = np.copy(GAIA_data.t)
            self.GAIAdata.w = np.copy(GAIA_data.w)
            self.GAIAdata.wsig = np.copy(GAIA_data.wsig)
            self.GAIAdata.psi = np.copy(GAIA_data.psi)
            self.GAIAdata.pf = np.copy(GAIA_data.pf)
            self.GAIAdata.N = GAIA_data.N

        else:
            if data is None:
                data = model.data
            self.data = data_holder()
            self.data.t = np.copy(data.t)
            self.data.y = np.copy(data.y)
            self.data.e = np.copy(data.sig)
            self.data.obs = np.copy(data.obsi)
            self.data.N = data.N

        # arbitrary units?
        if self.model is MODELS.RVGAIAmodel:
            self.arbitrary_units = False #hack to avoid lack of definition of model.data in RVGAIA date since 
        else:
            if 'arb' in model.data.units:
                self.arbitrary_units = True
            else:
                self.arbitrary_units = False

        #Add extra thigns for various models
        self.series = ('RV',)

        if self.model is MODELS.BINARIESmodel:
            self.double_lined = model.double_lined
            self.eclipsing = model.eclipsing
            self.relativistic_correction = model.relativistic_correction
            self.tidal_correction = model.tidal_correction
            self.star_mass = model.star_mass
            self.binary_mass = model.binary_mass
            self.star_radius = model.star_radius
            self.binary_radius = model.binary_radius
            if self.double_lined:
                self.data.y2 = np.array(np.copy(data.y2))
                self.data.e2 = np.array(np.copy(data.sig2))
                self.series = ('RV1','RV2')
        if self.model is MODELS.GAIAmodel:
            self.thiele_innes = model.thiele_innes
            self.RA = model.RA
            self.DEC= model.DEC
        if self.model is MODELS.RVGAIAmodel:
            self.thiele_innes = False
            self.RA = model.RA
            self.DEC= model.DEC
        if self.model is MODELS.RVFWHMmodel:
            self.series = ('RV', 'FWHM')
            self.data.y2, self.data.e2, *_ = np.array(data.actind)

        if self.model is MODELS.RVFWHMRHKmodel:
            self.series = ('RV', 'FWHM', 'RHK')
            self.data.y2, self.data.e2, self.data.y3, self.data.e3, *_ = np.array(data.actind)

        if self.model is MODELS.SPLEAFmodel:
            self.nseries = int(setup['kima']['nseries'])

        if self.data_type=='RV':
            self._extra_data = np.array(np.copy(data.actind))
            self._extra_data_names = np.array(np.copy(data.indicator_names))

        if self.model is MODELS.RVHGPMmodel:
            self.pm_data = model.pm_data
        
        if self.model is MODELS.RVGAIAmodel:
            self.M0_epoch = GAIA_data.M0_epoch
        else:
            self.M0_epoch = data.M0_epoch
        if self.data_type =='RV':
            self.n_instruments = np.unique(data.obsi).size
            self.multi = data.multi
        else:
            self.n_instruments = 1
            self.multi = False #Set multi to false for other datatypes 

        if self.multi:
            self.data_file = data.datafiles
        else:
            self.data_file = data.datafile

        if self.data_type == 'RV':
            self.data.instrument = data.instrument
            if data.instrument == 's':
                self.data.instrument = 'RVdata' #Hack for double lined binaries where the if only one instrument it wasn't defined
        if self.multi and len(data.instruments) > 0:
            self.instruments = data.instruments

        try:
            self.posterior_lnlike = np.atleast_2d(
                read_big_file("posterior_sample_info.txt",
                              names=["level_assignment", "logL", "tiebreaker", "ID"])
            )
            self._lnlike_available = True
        except IOError:
            self._lnlike_available = False
            print('Could not find file "posterior_sample_info.txt",'
                  'log-likelihoods will not be available.')

        try:
            with SimpleTimer() as timer:
                self.sample = np.atleast_2d(read_big_file('sample.txt'))
            if self._debug:
                print(f'Loading "sample.txt" took {timer.interval:.2f} seconds')

            with SimpleTimer() as timer:
                self.sample_info = np.atleast_2d(
                    read_big_file('sample_info.txt', 
                                  names=['level_assignment', 'logL', 'tiebreaker', 'ID'])
                )
            if self._debug:
                print(f'Loading "sample_info.txt" took {timer.interval:.2f} seconds')

            with SimpleTimer() as timer:
                self.levels = np.atleast_2d(read_big_file('levels.txt'))
            if self._debug:
                print(f'Loading "levels.txt" took {timer.interval:.2f} seconds')

            with open("sample.txt", "r") as fs:
                header = fs.readline()
                header = header.replace("#", "").replace("  ", " ").strip()
                self.parameters = [p for p in header.split(" ") if p != ""]
                self._parameters = copy(self.parameters)
                self.parameters.pop(self.parameters.index("ndim"))
                self.parameters.pop(self.parameters.index("maxNp"))
                self.parameters.pop(self.parameters.index("staleness"))

            # different sizes can happen when running the model and sample_info
            # was updated while reading sample.txt
            if self.sample.shape[0] != self.sample_info.shape[0]:
                minimum = min(self.sample.shape[0], self.sample_info.shape[0])
                self.sample = self.sample[:minimum]
                self.sample_info = self.sample_info[:minimum]

        except IOError:
            self.sample = None
            self.sample_info = None
            self.parameters = []

        self.indices = {}
        self._current_column = 0

        #### Probably no need to specify what the jitter datatype is so okay
        # read jitters
        if self.multi and self.model in (MODELS.RVmodel, MODELS.RVHGPMmodel):
            self.n_jitters = 1  # stellar jitter
        else:
            self.n_jitters = 0

        self.n_jitters += self.n_instruments

        if self.model is MODELS.RVFWHMmodel:
            self.n_jitters *= 2
        elif self.model is MODELS.BINARIESmodel:
            if self.double_lined:
                self.n_jitters *= 2
        elif self.model is MODELS.RVFWHMRHKmodel:
            self.n_jitters *= 3
        elif self.model is MODELS.SPLEAFmodel:
            # one jitter per activity indicator per instrument
            self.n_jitters += self.n_instruments * (self.nseries - 1)
        elif self.model is MODELS.RVGAIAmodel:
            self.n_jitters += 1
        
        try:
            self.jitter_propto_indicator = model.jitter_propto_indicator
            if model.jitter_propto_indicator:
                self.n_jitters += 1
        except AttributeError:
            self.jitter_propto_indicator = False

        self._read_jitters()
        
        # read limb-darkening coefficients
        if self.model is MODELS.TRANSITmodel:
            self._read_limb_dark()

        # default value
        self.trend = False

        #### this is only for RV models
        # read trend
        if self.data_type == "RV":
            self.trend = model.trend
            self.trend_degree = model.degree

            if self.model is MODELS.RVFWHMmodel:
                self.trend_fwhm = model.trend_fwhm
                self.trend_fwhm_degree = model.degree_fwhm

            self._read_trend()

        # does the model enforce AMD stability?
        try:
            self.enforce_stability = model.enforce_stability
        except AttributeError:
            self.enforce_stability = False

        try:
            self.star_mass = model.star_mass
        except AttributeError:
            self.star_mass = 1.0

        # multiple instruments? read offsets
        self._read_multiple_instruments()
        
        # activity indicator correlations?
        self._read_actind_correlations()
        
        # find GP in the compiled model
        self._read_GP()
        
        # # find MA in the compiled model
        # self._read_MA()

        #read astrometric solution
        if self.model in (MODELS.GAIAmodel, MODELS.RVGAIAmodel):
            self._read_astrometric_solution()
        
        # find KO in the compiled model
        self.KO = model.known_object
        self.nKO = model.n_known_object
        self._read_KO()
        
        # find transiting planet in the compiled model
        try:
            self.TR = model.transiting_planet
            self.nTR = model.n_transiting_planet
        except AttributeError:
            self.TR = False
            self.nTR = 0
        self._read_TR()
        
        if self.model is MODELS.OutlierRVmodel:
            self._read_outlier()
        
        self._read_components()
        
        # staleness (ignored)
        self._current_column += 1

        try:
            self.studentt = model.studentt
            self._read_studentt()
        except AttributeError:
            self.studentt = False

        if self.model is MODELS.RVHGPMmodel:
            self._read_pm()

        if self.model in (MODELS.RVFWHMmodel, MODELS.RVFWHMRHKmodel):
            self.cfwhm = self.posterior_sample[:, self._current_column]
            self.indices['cfwhm'] = self._current_column
            self._current_column += 1

            if self.model is MODELS.RVFWHMRHKmodel:
                self.crhk = self.posterior_sample[:, self._current_column]
                self.indices['crhk'] = self._current_column
                self._current_column += 1


        if self.model is MODELS.SPLEAFmodel:
            self.n_zero_points = self.n_instruments * (self.nseries - 1)
            istart = self._current_column
            iend = istart + self.n_zero_points
            self.indices['zero_points'] = slice(istart, iend)
            self.zero_points = self.posterior_sample[:, istart:iend]
            self._current_column += self.n_zero_points

        elif self.model is MODELS.BINARIESmodel:
            if self.double_lined:
                self.vsys_sec = self.posterior_sample[:, -2]
                self.indices['vsys_sec'] = -2
        if self.data_type == 'RV':
            self.vsys = self.posterior_sample[:, -1]
            self.indices['vsys'] = -1

        # build the marginal posteriors for planet parameters
        self.get_marginals()

        if self.fix:
            self.parameters.pop(self.parameters.index('Np'))

        self._set_plots()


    @property
    def ESS(self):
        """ Effective sample size """
        try:
            return self._ESS
        except AttributeError:
            self._ESS = self.posterior_sample.shape[0]
            return self._ESS

    def __repr__(self):
        return f'KimaResults(lnZ={self.evidence:.1f}, ESS={self.ESS})'

    def _read_jitters(self):
        i1, i2 = self._current_column, self._current_column + self.n_jitters
        self.jitter = self.posterior_sample[:, i1:i2]
        self._current_column += self.n_jitters
        self.indices['jitter_start'] = i1
        self.indices['jitter_end'] = i2
        self.indices['jitter'] = slice(i1, i2)
        if self._debug:
            print(f'finished reading ({self.n_jitters}) jitters')

    def _read_astrometric_solution(self):
        self.n_astrometric_solution = 5
        i1, i2 = self._current_column, self._current_column + self.n_astrometric_solution
        self.astrometric_solution = self.posterior_sample[:, i1:i2]
        self._current_column += self.n_astrometric_solution
        self.indices['astrometric_solution_start'] = i1
        self.indices['astrometric_solution_end'] = i2
        self.indices['astrometric_solution'] = slice(i1, i2)
        if self._debug:
            print('finished reading astrometric solution')

    def _read_limb_dark(self):
        return
        # i1, i2 = self._current_column, self._current_column + 2
        # self.u = self.posterior_sample[:, i1:i2]
        # self._current_column += 2
        # self.indices['u_start'] = i1
        # self.indices['u_end'] = i2
        # self.indices['u'] = slice(i1, i2)
        # if self._debug:
        #     print('finished reading limb darkening')

    def _read_trend(self):
        if self.trend:
            n_trend = self.trend_degree
            i1 = self._current_column
            i2 = self._current_column + n_trend
            self.trendpars = self.posterior_sample[:, i1:i2]
            self._current_column += n_trend
            self.indices['trend'] = slice(i1, i2)
        
        if self.model is MODELS.RVFWHMmodel and self.trend_fwhm:
            n_trend = self.trend_fwhm_degree
            i1 = self._current_column
            i2 = self._current_column + n_trend
            self.trend_fwhm_pars = self.posterior_sample[:, i1:i2]
            self._current_column += n_trend
            self.indices['trend_fwhm'] = slice(i1, i2)

        if self._debug:
            print('finished reading trend, trend =', self.trend)

    def _read_multiple_instruments(self):
        if self.multi:
            # there are n instruments and n-1 offsets per output
            if self.model is MODELS.RVFWHMmodel:
                n_inst_offsets = 2 * (self.n_instruments - 1)
            elif self.model is MODELS.BINARIESmodel:
                if self.double_lined:
                    n_inst_offsets = 2 * (self.n_instruments - 1)
                else:
                    n_inst_offsets = self.n_instruments - 1
            elif self.model is MODELS.RVFWHMRHKmodel:
                n_inst_offsets = 3 * (self.n_instruments - 1)
            else:
                n_inst_offsets = self.n_instruments - 1

            istart = self._current_column
            iend = istart + n_inst_offsets
            ind = np.s_[istart:iend]
            self.inst_offsets = self.posterior_sample[:, ind]
            self._current_column += n_inst_offsets
            self.indices['inst_offsets_start'] = istart
            self.indices['inst_offsets_end'] = iend
            self.indices['inst_offsets'] = slice(istart, iend)
        else:
            n_inst_offsets = 0

        if self._debug:
            print('finished reading multiple instruments')

    def _read_actind_correlations(self):
        setup = self.setup
        try:
            self.indicator_correlations = setup['kima']['indicator_correlations'] == 'true'
        except KeyError:
            self.indicator_correlations = False

        if self.indicator_correlations:
            activity_indicators = setup['data']['indicators'].split(',')
            activity_indicators = list(filter(None, activity_indicators))
            self.activity_indicators = activity_indicators
            n_act_ind = len(self.activity_indicators)
            istart = self._current_column
            iend = istart + n_act_ind
            ind = np.s_[istart:iend]
            self.betas = self.posterior_sample[:, ind]
            self._current_column += n_act_ind
            self.indices['betas_start'] = istart
            self.indices['betas_end'] = iend
            self.indices['betas'] = slice(istart, iend)
        else:
            n_act_ind = 0

    def _read_components(self):
        # how many parameters per component
        self.n_dimensions = int(self.posterior_sample[0, self._current_column])
        self._current_column += 1

        # maximum number of components
        self.max_components = self.npmax
        self._current_column += 1

        # find hyperpriors in the compiled model
        self.hyperpriors = False
        #self.hyperpriors = self.setup['kima']['hyperpriors'] == 'true'

        # number of hyperparameters (muP, wP, muK)
        if self.hyperpriors:
            n_dist_print = 3
            istart = self._current_column
            iend = istart + n_dist_print
            self._current_column += n_dist_print
            self.indices['hyperpriors'] = slice(istart, iend)
        else:
            n_dist_print = 0

        # if hyperpriors, then the period is sampled in log
        self.log_period = self.hyperpriors

        # the column with the number of planets in each sample
        self.index_component = self._current_column

        if not self.fix:
            self.priors['np_prior'] = discrete_uniform(0, self.npmax + 1)
            self.priors['np_prior'].logpdf = self.priors['np_prior'].logpmf


        self.indices['np'] = self.index_component
        self._current_column += 1
        
        
        # indices of the planet parameters
        n_planet_pars = self.max_components * self.n_dimensions
        istart = self._current_column
        iend = istart + n_planet_pars
        self._current_column += n_planet_pars
        self.indices['planets'] = slice(istart, iend)
        

        if self.model in (MODELS.GAIAmodel,MODELS.RVGAIAmodel):
            if self.thiele_innes:
                for j, p in zip(range(self.n_dimensions), ('P', 'φ', 'e', 'A', 'B', 'F', 'G')):
                    iend = istart + self.max_components
                    self.indices[f'planets.{p}'] = slice(istart, iend)
                    istart += self.max_components
            else:
                for j, p in zip(range(self.n_dimensions), ('P', 'φ', 'e', 'a0', 'w', 'cosi', 'W')):
                    iend = istart + self.max_components
                    self.indices[f'planets.{p}'] = slice(istart, iend)
                    istart += self.max_components
        elif self.model is MODELS.RVHGPMmodel:
            for j, p in zip(range(self.n_dimensions), ('P', 'K', 'φ', 'e', 'w', 'i', 'W')):
                iend = istart + self.max_components
                self.indices[f'planets.{p}'] = slice(istart, iend)
                istart += self.max_components
        else:
            for j, p in zip(range(self.n_dimensions), ('P', 'K', 'φ', 'e', 'w')):
                iend = istart + self.max_components
                self.indices[f'planets.{p}'] = slice(istart, iend)
                istart += self.max_components

    def _read_studentt(self):
        if self.studentt:
            if self.model is MODELS.RVGAIAmodel:
                self.nu_GAIA = self.posterior_sample[:, self._current_column]
                self.indices['nu_GAIA'] = self._current_column
                self._current_column += 1
                self.nu_RV = self.posterior_sample[:, self._current_column]
                self.indices['nu_RV'] = self._current_column
                self._current_column += 1
            else:
                self.nu = self.posterior_sample[:, self._current_column]
                self.indices['nu'] = self._current_column
                self._current_column += 1

    def _read_pm(self):
        self.indices['pm_ra_bary'] = self._current_column
        self.indices['pm_dec_bary'] = self._current_column + 1
        self.indices['parallax'] = self._current_column + 2
        self._current_column += 3

    @property
    def _GP_par_indices(self):
        """
        indices for specific GP hyperparameters:
        eta1_RV, eta1_FWHM, eta2_RV, eta2_FWHM, eta3_RV, eta3_FWHM, eta4_RV, eta4_FWHM
        """
        if self.model is MODELS.RVFWHMmodel:
            i = [0, 1]  # eta1_rv, eta1_fwhm
            i += 2 * [i[-1] + 1] if self.share_eta2 else [i[-1] + 1, i[-1] + 2]
            i += 2 * [i[-1] + 1] if self.share_eta3 else [i[-1] + 1, i[-1] + 2]
            i += 2 * [i[-1] + 1] if self.share_eta4 else [i[-1] + 1, i[-1] + 2]
        elif self.model is MODELS.RVFWHMRHKmodel:
            i = [0, 1, 2]  # eta1_rv, eta1_fwhm, eta1_rhk
            i += 3 * [i[-1]+1] if self.share_eta2 else list(range(i[-1]+1, i[-1] + 4))
            i += 3 * [i[-1]+1] if self.share_eta3 else list(range(i[-1]+1, i[-1] + 4))
            i += 3 * [i[-1]+1] if self.share_eta4 else list(range(i[-1]+1, i[-1] + 4))
            if self.magnetic_cycle_kernel:
                i += range(i[-1] + 1, i[-1] + 1 + 5)
        else:
            i = [0, 1, 2, 3]

        return i

    def _read_GP(self):
        from .. import GP
        if self.model not in (MODELS.GPmodel, MODELS.RVFWHMmodel, MODELS.RVFWHMRHKmodel, MODELS.SPLEAFmodel):
            self.has_gp = False
            self.n_hyperparameters = 0
            return
        
        self.has_gp = True
        try:
            self.kernel = GP.KernelType(int(self.setup['kima']['kernel']))
        except KeyError:
            self.kernel = GP.KernelType(0)

        try:
            self.magnetic_cycle_kernel = self.setup['kima']['magnetic_cycle_kernel'] == 'true'
            if self.magnetic_cycle_kernel:
                self.kernel = 'standard+magcycle'
        except KeyError:
            pass

        hyperparameters_per_kernel = {
            GP.KernelType.qp: 4,
            GP.KernelType.per: 3,
            GP.KernelType.spleaf_exp: 2,
            GP.KernelType.spleaf_matern32: 2,
            GP.KernelType.spleaf_sho: 3,
            GP.KernelType.spleaf_mep: 4,
            GP.KernelType.spleaf_esp: 4,
            'standard+magcycle': 7,
        }
        if self.model is MODELS.GPmodel:
            try:
                n_hyperparameters = hyperparameters_per_kernel[self.kernel]
            except KeyError:
                raise ValueError(
                    f'GP kernel = {self.kernel} not recognized')

            self.n_hyperparameters = n_hyperparameters

        elif self.model is MODELS.RVFWHMmodel:
            _n_shared = 0
            for i in range(2, 5):
                setattr(self, f'share_eta{i}',
                        self.setup['kima'][f'share_eta{i}'] == 'true')
                if getattr(self, f'share_eta{i}'):
                    _n_shared += 1

            n_hyperparameters = 2  # at least 2 x eta1
            n_hyperparameters += 1 if self.share_eta2 else 2
            n_hyperparameters += 1 if self.share_eta3 else 2
            n_hyperparameters += 1 if self.share_eta4 else 2
            self.n_hyperparameters = n_hyperparameters
            self._n_shared_hyperparameters = _n_shared

        elif self.model is MODELS.RVFWHMRHKmodel:
            _n_shared = 0
            for i in range(2, 5):
                setattr(self, f'share_eta{i}',
                        self.setup['kima'][f'share_eta{i}'] == 'true')
                if getattr(self, f'share_eta{i}'):
                    _n_shared += 1

            n_hyperparameters = 3  # at least 3 x eta1
            n_hyperparameters += 1 if self.share_eta2 else 3
            n_hyperparameters += 1 if self.share_eta3 else 3
            n_hyperparameters += 1 if self.share_eta4 else 3
            if self.magnetic_cycle_kernel:
                n_hyperparameters += 5
            self.n_hyperparameters = n_hyperparameters
            self._n_shared_hyperparameters = _n_shared

        elif self.model is MODELS.SPLEAFmodel:
            self.n_hyperparameters = hyperparameters_per_kernel[self.kernel]

        istart = self._current_column
        iend = istart + self.n_hyperparameters
        self.etas = self.posterior_sample[:, istart:iend]

        self._current_column += self.n_hyperparameters
        self.indices['GPpars_start'] = istart
        self.indices['GPpars_end'] = iend
        self.indices['GPpars'] = slice(istart, iend)

        t, e = self.data.t, self.data.e
        kernels = {
            GP.KernelType.qp: QPkernel(1, 1, 1, 1),
            GP.KernelType.per: PERkernel(1, 1, 1),
            GP.KernelType.spleaf_exp: EXPkernel(1, 1), 
            GP.KernelType.spleaf_matern32: Matern32kernel(1, 1),
            GP.KernelType.spleaf_sho: SHOkernel(1, 1, 1),
            GP.KernelType.spleaf_mep: MEPkernel(1, 1, 1, 1),
            GP.KernelType.spleaf_esp: ESPkernel(1, 1, 1, 1, nharm=3),
            'standard+magcycle': QPpMAGCYCLEkernel(1, 1, 1, 1, 1, 1, 1),

            #'qpc': QPCkernel(1, 1, 1, 1, 1),
            #'RBF': RBFkernel(1, 1),
            #'qp_plus_cos': QPpCkernel(1, 1, 1, 1, 1, 1),
        }

        if self.model is MODELS.RVFWHMmodel:
            self.GP1 = GaussianProcess(deepcopy(kernels[self.kernel]), t, e, white_noise=0.0)
            self.GP2 = GaussianProcess(deepcopy(kernels[self.kernel]), t, self.data.e2, white_noise=0.0)

        if self.model is MODELS.RVFWHMRHKmodel:
            self.GP1 = GaussianProcess(deepcopy(kernels[self.kernel]), t, e, white_noise=0.0)
            self.GP2 = GaussianProcess(deepcopy(kernels[self.kernel]), t, self.data.e2, white_noise=0.0)
            self.GP3 = GaussianProcess(deepcopy(kernels[self.kernel]), t, self.data.e3, white_noise=0.0)

        # elif self.model is MODELS.GPmodel_systematics:
        #     X = np.c_[self.data.t, self._extra_data[:, 3]]
        #     self.GP = mixtureGP([], X, None, e)

        elif self.model is MODELS.SPLEAFmodel:
            pass
        else:
            self.GP = GaussianProcess(kernels[self.kernel], t, e, white_noise=0.0)

        if self.model is MODELS.SPLEAFmodel:
            n_alphas = n_betas = self.nseries
            istart = self._current_column
            iend = istart + n_alphas + n_betas
            alphas_betas = self.posterior_sample[:, istart:iend]
            self.alphas = alphas_betas[:, ::2]
            self.betas = alphas_betas[:, 1::2]
            self.indices['GP_alphas'] = slice(istart, iend, 2)
            self.indices['GP_betas'] = slice(istart + 1, iend, 2)
            self._current_column += n_alphas + n_betas

    def _read_MA(self):
        # find MA in the compiled model
        try:
            self.MAmodel = self.setup['kima']['MA'] == 'true'
        except KeyError:
            self.MAmodel = False

        if self.MAmodel:
            n_MAparameters = 2
            istart = self._current_column
            iend = istart + n_MAparameters
            self.MA = self.posterior_sample[:, istart:iend]
            self._current_column += n_MAparameters
        else:
            n_MAparameters = 0

    def _read_KO(self):
        if self.KO:
            if self.model is MODELS.TRANSITmodel:
                n_KOparameters = 6 * self.nKO
            elif self.model is MODELS.GAIAmodel:
                n_KOparameters = 7 * self.nKO
            elif self.model is MODELS.RVGAIAmodel:
                n_KOparameters = 7 * self.nKO
            elif self.model is MODELS.BINARIESmodel:
                if self.double_lined:
                    n_KOparameters = 8 * self.nKO
                else:
                    n_KOparameters = 7 * self.nKO
            else:
                n_KOparameters = 5 * self.nKO
            start = self._current_column
            koinds = slice(start, start + n_KOparameters)
            self.KOpars = self.posterior_sample[:, koinds]
            self._current_column += n_KOparameters
            self.indices['KOpars'] = koinds
        else:
            n_KOparameters = 0

    def _read_TR(self):
        if self.TR:
            if self.model is MODELS.TRANSITmodel:
                n_TRparameters = 6 * self.nTR
            else:
                n_TRparameters = 5 * self.nTR
            start = self._current_column
            TRinds = slice(start, start + n_TRparameters)
            self.TRpars = self.posterior_sample[:, TRinds]
            self._current_column += n_TRparameters
            self.indices['TRpars'] = TRinds
        else:
            n_TRparameters = 0

    def _read_outlier(self):
        n_outlier_parameters = 3
        start = self._current_column
        outlier_inds = slice(start, start + n_outlier_parameters)
        self.outlier_pars = self.posterior_sample[:, outlier_inds]
        self._current_column += n_outlier_parameters
        self.indices['outlier'] = outlier_inds

    def _make_named_arrays(self):
        self.posteriors.jitter = self.posteriors.jitter.view(named_array)
        columns = []
        if self.model is MODELS.RVmodel and self.multi:
            columns.append('stellar')
    
        if self.multi:
            columns += self.instruments
        else:
            columns += [self.instruments]

        if self.model is MODELS.RVFWHMmodel:
            columns += [i + '_FWHM' for i in columns]

        if hasattr(self, 'jitter_propto_indicator') and self.jitter_propto_indicator:
            columns.append('slope')
        self.posteriors.jitter.set_columns(tuple(columns))

        if self.multi:
            self.posteriors.offset = self.posteriors.offset.view(named_array)
            self.posteriors.offset.set_columns(tuple(self.instruments[:-1]))

    @property
    def _mc(self):
        """ Maximum number of Keplerians in the model """
        return self.max_components

    @property
    def _nd(self):
        """ Number of parameters per Keplerian """
        return self.n_dimensions

    @property
    def total_parameters(self):
        return len(self.parameters)

    @property
    def parameter_priors(self):
        """ A list of priors which can be indexed using self.indices """
        priors = np.full(self.posterior_sample.shape[1], None)

        if self.model is MODELS.RVFWHMmodel:
            for i in range(self.n_instruments):
                priors[i] = self.priors['Jprior']
            for i in range(self.n_instruments, 2 * self.n_instruments):
                priors[i] = self.priors['Jfwhm_prior']
        else:
            priors[self.indices['jitter']] = self.priors['Jprior']
            if self.model is MODELS.RVmodel and self.multi:
                priors[0] = self.priors['stellar_jitter_prior']

        if self.trend:
            names = ('slope_prior', 'quadr_prior', 'cubic_prior')
            trend_priors = [self.priors[n] for n in names if n in self.priors]
            priors[self.indices['trend']] = trend_priors
        
        if self.model is MODELS.RVFWHMmodel and self.trend_fwhm:
            names = ('slope_fwhm_prior', 'quadr_fwhm_prior', 'cubic_fwhm_prior')
            trend_priors = [self.priors[n] for n in names if n in self.priors]
            priors[self.indices['trend_fwhm']] = trend_priors

        if self.multi:
            no = self.n_instruments - 1
            if self.model is MODELS.RVFWHMmodel:
                prior1 = self.priors['offsets_prior']
                prior2 = self.priors['offsets_fwhm_prior']
                offset_priors = no * [prior1] + no * [prior2]
                priors[self.indices['inst_offsets']] = np.array(offset_priors)
            else:
                for i in range(no):
                    prior = self.priors[f'individual_offset_prior[{i}]']
                    priors[self.indices['inst_offsets']][i] = prior

        if self.indicator_correlations:
            prior = self.priors['beta_prior']
            priors[self.indices['betas']] = prior


        if self.has_gp:
            if self.model in (MODELS.GPmodel, MODELS.SPLEAFmodel):
                priors[self.indices['GPpars']] = [
                    self.priors[f'eta{i}_prior'] for i in range(1, 5)
                ]
            elif self.model is MODELS.RVFWHMmodel:
                i = self.indices['GPpars_start']
                priors[i] = self.priors['eta1_prior']
                i += 1
                priors[i] = self.priors['eta1_fwhm_prior']
                i += 1
                if self.share_eta2:
                    priors[i] = self.priors['eta2_prior']
                    i += 1
                else:
                    priors[i] = self.priors['eta2_prior']
                    priors[i + 1] = self.priors['eta2_fwhm_prior']
                    i += 2
                #
                if self.share_eta3:
                    priors[i] = self.priors['eta3_prior']
                    i += 1
                else:
                    priors[i] = self.priors['eta3_prior']
                    priors[i + 1] = self.priors['eta3_fwhm_prior']
                    i += 2
                #
                if self.share_eta4:
                    priors[i] = self.priors['eta4_prior']
                    i += 1
                else:
                    priors[i] = self.priors['eta4_prior']
                    priors[i + 1] = self.priors['eta4_fwhm_prior']
                    i += 2
            
            if self.model is MODELS.SPLEAFmodel:
                priors[self.indices['GP_alphas']] = [self.priors[f'alpha{i+1}_prior'] for i in range(self.nseries)]
                priors[self.indices['GP_betas']] = [self.priors[f'beta{i+1}_prior'] for i in range(self.nseries)]

        if self.KO:
            KO_priors = []
            KO_priors += [self.priors[f'KO_Pprior_{i}'] for i in range(self.nKO)]
            KO_priors += [self.priors[f'KO_Kprior_{i}'] for i in range(self.nKO)]
            KO_priors += [self.priors[f'KO_phiprior_{i}'] for i in range(self.nKO)]
            KO_priors += [self.priors[f'KO_eprior_{i}'] for i in range(self.nKO)]
            KO_priors += [self.priors[f'KO_wprior_{i}'] for i in range(self.nKO)]
            priors[self.indices['KOpars']] = KO_priors

        if self.TR:
            TR_priors = []
            TR_priors += [self.priors[f'TR_Pprior_{i}'] for i in range(self.nTR)]
            TR_priors += [self.priors[f'TR_Kprior_{i}'] for i in range(self.nTR)]
            TR_priors += [self.priors[f'TR_Tcprior_{i}'] for i in range(self.nTR)]
            TR_priors += [self.priors[f'TR_eprior_{i}'] for i in range(self.nTR)]
            TR_priors += [self.priors[f'TR_wprior_{i}'] for i in range(self.nTR)]
            priors[self.indices['TRpars']] = TR_priors

        if self.fix:
            priors[self.indices['np']] = distributions.Fixed(self.npmax)
        else:
            try:
                priors[self.indices['np']] = self.priors['np_prior']
            except KeyError:
                priors[self.indices['np']] = discrete_uniform(0, self.npmax + 1)

        if self.max_components > 0:
            planet_priors = []
            for i in range(self.max_components):
                planet_priors.append(self.priors['Pprior'])
            for i in range(self.max_components):
                planet_priors.append(self.priors['Kprior'])
            for i in range(self.max_components):
                planet_priors.append(self.priors['phiprior'])
            for i in range(self.max_components):
                planet_priors.append(self.priors['eprior'])
            for i in range(self.max_components):
                planet_priors.append(self.priors['wprior'])
            priors[self.indices['planets']] = planet_priors

        if self.studentt:
            priors[self.indices['nu']] = self.priors['nu_prior']

        priors[self.indices['vsys']] = self.priors['Cprior']
        if self.model is MODELS.RVFWHMmodel:
            priors[self.indices['cfwhm']] = self.priors['Cfwhm_prior']

        return priors

    @property
    def _parameter_priors_full(self):
        from utils import ZeroDist
        parameter_priors = self.parameter_priors
        for i, p in enumerate(parameter_priors):
            if p is None:
                parameter_priors[i] = ZeroDist()
        return parameter_priors

    @classmethod
    def load(cls, filename: str = None, diagnostic: bool = False, **kwargs):
        """
        Load a KimaResults object from the current directory, a pickle file, or
        a zip file.

        Args:
            filename (str, optional):
                If given, load the model from this file. Can be a zip or pickle
                file. Defaults to None.
            diagnostic (bool, optional):
                Whether to plot the DNest4 diagnotics. Defaults to False.
            **kwargs: Extra keyword arguments passed to `showresults`

        Returns:
            res (KimaResults): An object holding the results
        """
        if filename is None:
            from .showresults import showresults
            return showresults(force_return=True, **kwargs)

        try:
            if filename.endswith('.zip'):
                zf = zipfile.ZipFile(filename, 'r')
                names = zf.namelist()
                needs = ('sample.txt', 'levels.txt', 'sample_info.txt',
                         'kima_model_setup.txt')
                wants = ('posterior_sample.txt', 'posterior_sample_info.txt')

                for need in needs:
                    if need not in names:
                        raise ValueError('%s does not contain a "%s" file' %
                                         (filename, need))

                with tempfile.TemporaryDirectory() as dirpath:
                    for need in needs:
                        zf.extract(need, path=dirpath)

                    for want in wants:
                        try:
                            zf.extract(need, path=dirpath)
                        except FileNotFoundError:
                            pass

                    try:
                        zf.extract('evidence', path=dirpath)
                        zf.extract('information', path=dirpath)
                    except Exception:
                        pass

                    with chdir(dirpath):
                        setup = read_model_setup()

                        section = 'data' if 'data' in setup else 'kima'
                        try:
                            multi = setup[section]['multi'] == 'true'
                        except KeyError:
                            multi = False

                        if multi:
                            datafiles = setup[section]['files'].split(',')
                            datafiles = list(filter(None, datafiles))
                        else:
                            datafiles = np.atleast_1d(setup['data']['file'])

                        datafiles = list(map(os.path.basename, datafiles))
                        for df in datafiles:
                            zf.extract(df)

                        if os.path.exists(wants[0]):
                            res = cls('')
                            res.evidence = float(open('evidence').read())
                            res.information = float(open('information').read())
                            res.ESS = res.posterior_sample.shape[0]
                        else:
                            from showresults import showresults
                            res = showresults(verbose=False)

                        # from classic import postprocess
                        # postprocess()

            elif filename.endswith('.pkl'):
                import pickle
                try:
                    with open(filename, 'rb') as f:
                        res = pickle.load(f)
                except UnicodeDecodeError:
                    with open(filename, 'rb') as f:
                        res = pickle.load(f, encoding='latin1')

            elif filename.endswith(tuple(COMPRESSED_FILE_EXT.values())):
                try:
                    import compress_pickle as pickle
                except (ImportError, ModuleNotFoundError):
                    print('reading compressed file requires the `compress-pickle` package')
                    return
                
                res = pickle.load(filename)


        except Exception:
            # print('Unable to load data from ', filename, ':', e)
            raise

        res._update()
        res._set_plots()
        res.get_marginals()
        return res

    def _update(self):
        if isinstance(self.model, str):
            self.model = MODELS(self.model)

        if hasattr(self, 'studentT'):
            self.studentt = self.studentT
            del self.studentT

        if not hasattr(self, '_ESS'):
            self._ESS = self.posterior_sample.shape[0]
        
        if not hasattr(self, 'logZ'):
            self.logZ = self.evidence

        try:
            from .. import GPmodel
            if self.model is MODELS.GPmodel:
                if not hasattr(self, 'kernel'):
                    self.kernel = {
                        'standard': GPmodel.KernelType.qp
                    }[self.GPkernel]
                del self.GPkernel
        except AttributeError:
            pass

        try:
            if not hasattr(self.posteriors, 'η1'):
                for i, eta in enumerate(self.etas.T):
                    setattr(self.posteriors, f'η{i+1}', eta)
        except AttributeError:
            pass


    def show_kima_setup(self):
        return _show_kima_setup()

    def get_model_id(self, add_timestamp=True):
        if self.star in (None, '', 'unknown'):
            id = 'kima_'
        else: 
            id = self.star + '_'

        id += f'k{self.npmax}_' if self.fix else f'k0{self.npmax}_'
        id += f'd{self.trend_degree}_' if self.trend else ''
        id += 'studentt_' if self.studentt else ''
        id += 'GP_' if self.model is MODELS.GPmodel else ''
        id += 'RVFWHM_' if self.model is MODELS.RVFWHMmodel else ''
        id += 'RVFWHMRHK_' if self.model is MODELS.RVFWHMRHKmodel else ''
        id += f'KO{self.nKO}_' if self.KO else ''
        id += f'TR{self.nTR}_' if self.TR else ''
        if add_timestamp:
            id += get_timestamp()
        return id

    def save_pickle(self, filename: str=None, directory: str=None,
                    postfix: str=None, compress=False, verbose: bool=True):
        """ Pickle this KimaResults object into a file.

        Args:
            filename (str, optional):
                The name of the file where to save the model. If not given, a
                unique name will be generated from the properties of the model.
            directory (str, optional):
                The directory where to save the file.
            postfix (str, optional):
                A string to add to the filename, before the extension.
            compress (bool or str, optional):
                Compress the pickle file. Requires the `compress_pickle`
                package. If a string, use the specified compression method. If
                True, uses bz2 compression by default.

                | method | speed, size    |
                |--------|----------------|
                | bz2    | fast, small    |
                | gzip   | fast, small    |
                | lz4    | very fast, big |
                | lzma   | slow, smallest |
                | zipfile| very fast, big |

            verbose (bool, optional):
                Print a message. Defaults to True.
        Returns:
            filename (str): The name of the pickle file where the model was saved
        """

        ending = '.pkl'
        dump_kwargs = {}

        if compress is False:
            import pickle
            dump_kwargs['protocol'] = 2
        else:
            try:
                import compress_pickle as pickle
                if compress is True:
                    # use bz2 compression by default, good compromise between
                    # speed and file size
                    dump_kwargs['compression'] = 'bz2'
                    ending = '.pkl' + COMPRESSED_FILE_EXT['bz2']

                elif isinstance(compress, str):
                    available = list(filter(None, pickle.compressers.registry.get_known_compressions()))
                    if compress not in available:
                        print('available compression methods: ', available)
                        return
                    dump_kwargs['compression'] = compress
                    ending = ending + COMPRESSED_FILE_EXT[compress]

            except (ImportError, ModuleNotFoundError):
                print('compression requires the `compress-pickle` package')
                return

        if filename is None:
            filename = self.get_model_id(add_timestamp=True)

        if postfix is not None:
            filename += '_' + postfix

        if filename.endswith('.pkl'):
            filename = filename.replace('.pkl', ending)
        elif filename.endswith('.pickle'):
            ending = ending.replace('.pkl', '.pickle')
            filename = filename.replace('.pickle', ending)
        else:
            filename += ending

        if directory is not None:
            filename = os.path.join(directory, filename)

        with open(filename, 'wb') as f:
            pickle.dump(self, f, **dump_kwargs)

        if verbose:
            print('Wrote to file "%s"' % filename)

        return filename

    def save_zip(self, filename: str, verbose=True):
        """ Save this KimaResults object and the text files into a zip.

        Args:
            filename (str): The name of the file where to save the model
            verbose (bool, optional): Print a message. Defaults to True.
        """
        if not filename.endswith('.zip'):
            filename = filename + '.zip'

        zf = zipfile.ZipFile(filename, 'w', compression=zipfile.ZIP_DEFLATED)
        tosave = ('sample.txt', 'sample_info.txt', 'levels.txt',
                  'sampler_state.txt', 'posterior_sample.txt',
                  'posterior_sample_info.txt')
        for f in tosave:
            zf.write(f)

        text = open('kima_model_setup.txt').read()
        for f in self.data_file:
            text = text.replace(f, os.path.basename(f))
        zf.writestr('kima_model_setup.txt', text)

        try:
            zf.writestr('evidence', str(self.evidence))
            zf.writestr('information', str(self.information))
        except AttributeError:
            pass

        for f in np.atleast_1d(self.data_file):
            zf.write(f, arcname=os.path.basename(f))

        zf.close()
        if verbose:
            print('Wrote to file "%s"' % zf.filename)

    def get_marginals(self):
        """
        Get the marginal posteriors from the posterior_sample matrix.
        They go into self.T, self.A, self.E, etc
        """

        self.posteriors = posterior_holder()
        self._priors = prior_holder()

        # jitter(s)
        self.posteriors.jitter = self.posterior_sample[:, self.indices['jitter']]
        self.posteriors.jitter = self.posteriors.jitter.view(named_array)
        if self.model is MODELS.RVGAIAmodel:
            self.priors['jitter_RV'] = self.priors['J_RV_prior']
            self.priors['jitter_GAIA'] = self.priors['J_GAIA_prior']
        else:
            self._priors.jitter = self.priors['Jprior']
        # if self.n_jitters == 1:
        #     self.posteriors.jitter = self.posteriors.jitter.ravel()

        if self.studentt:
            if self.model is MODELS.RVGAIAmodel:
                self.posteriors.nu_RV = self.posterior_sample[:, self.indices["nu"]]
                self.posteriors.nu_GAIA = self.posterior_sample[:, self.indices["nu"]]
                self._priors.nu_RV = self.priors["nu_RV_prior"]
                self._priors.nu_GAIA = self.priors["nu_GAIA_prior"]
            else:
                self.posteriors.nu = self.posterior_sample[:, self.indices["nu"]]
                self._priors.nu = self.priors["nu_prior"]

        if self.has_gp:
            for i in range(self.n_hyperparameters):
                setattr(self.posteriors, f'η{i+1}', self.etas[:, i])
                setattr(self.posteriors, f'_eta{i+1}', self.etas[:, i])
                if self.model != MODELS.RVFWHMmodel:
                    try:
                        setattr(self._priors, f'η{i+1}', self.priors[f'eta{i+1}_prior'])
                    except KeyError:
                        pass
            
            if self.model is MODELS.SPLEAFmodel:
                self.posteriors.alpha = self.alphas
                self.posteriors.beta = self.betas
                for i in range(self.nseries):
                    setattr(self._priors, f'alpha{i+1}', self.priors[f'alpha{i+1}_prior'])
                    setattr(self._priors, f'beta{i+1}', self.priors[f'beta{i+1}_prior'])

        if self.model in (MODELS.GAIAmodel,MODELS.RVGAIAmodel):
            da, dd, mua, mud, plx = self.posterior_sample[:, self.indices['astrometric_solution']].T
            self.posteriors.da = da
            self.posteriors.dd = dd
            self.posteriors.mua = mua
            self.posteriors.mud = mud
            self.posteriors.plx = plx
            # TODO: _priors

        # instrument offsets
        if self.multi:
            self.posteriors.offset = self.posterior_sample[:, self.indices['inst_offsets']]
            # TODO: _priors
        
        if self.model != MODELS.GAIAmodel:
            # systemic velocity
            self.posteriors.vsys = self.posterior_sample[:, self.indices['vsys']].reshape(-1, 1)
            if self.model is MODELS.BINARIESmodel and self.double_lined:
                self.posteriors.vsys_sec = self.posterior_sample[:, self.indices['vsys_sec']].reshape(-1, 1)
            self._priors.vsys = self.priors['Cprior']
            if self.model is MODELS.RVFWHMmodel:
                self.posteriors.cfwhm = self.posterior_sample[:, self.indices['cfwhm']]
                self._priors.cfwhm = self.priors['Cfwhm_prior']

        if self.data_type=='RV' and self.trend:
            ind = self.indices['trend']
            ind = list(range(ind.start or 0, ind.stop or 0, ind.step or 1))
            for i, name in zip(ind, ('slope', 'quadr', 'cubic')):
                setattr(self.posteriors, name, self.posterior_sample[:, i])
                setattr(self._priors, name, self.priors[f'{name}_prior'])
        
        if self.model is MODELS.RVFWHMmodel and self.trend_fwhm:
            ind = self.indices['trend_fwhm']
            ind = list(range(ind.start or 0, ind.stop or 0, ind.step or 1))
            for i, name in zip(ind, ('slope_fwhm', 'quadr_fwhm', 'cubic_fwhm')):
                setattr(self.posteriors, name, self.posterior_sample[:, i])
                setattr(self._priors, name, self.priors[f'{name}_prior'])

        if self.model is MODELS.RVHGPMmodel:
            self.posteriors.pm_ra_bary = self.posterior_sample[:, self.indices['pm_ra_bary']]
            self.posteriors.pm_dec_bary = self.posterior_sample[:, self.indices['pm_dec_bary']]

        # parameters of the outlier model
        if self.model is MODELS.OutlierRVmodel:
            self.posteriors.outlier_mean, self.posteriors.outlier_sigma, self.posteriors.outlier_Q = \
                self.posterior_sample[:, self.indices['outlier']].T
            # TODO: _priors

        max_components = self.max_components
        index_component = self.index_component

        if max_components > 0:
            # periods
            s = self.indices["planets.P"]
            self.posteriors.P = self.posterior_sample[:, s]
            self._priors.P = self.priors["Pprior"]

            # RV semi-amplitudes
            models_with_K = (
                MODELS.RVmodel,
                MODELS.GPmodel,
                MODELS.RVHGPMmodel,
                MODELS.BINARIESmodel,
            )
            if self.model in models_with_K:
                # amplitudes
                s = self.indices["planets.K"]
                self.posteriors.K = self.posterior_sample[:, s]
                self._priors.K = self.priors["Kprior"]

            # eccentricities
            s = self.indices["planets.e"]
            self.posteriors.e = self.posterior_sample[:, s]
            self._priors.e = self.priors["eprior"]

            # phases
            s = self.indices['planets.φ']
            φ = self.posteriors.φ = self.posteriors._phi = self.posterior_sample[:, s]
            self.posteriors.φ_deg = np.rad2deg(φ)
            self._priors.φ = self.priors['phiprior']

            if self.model in (MODELS.GAIAmodel,MODELS.RVGAIAmodel):
                if self.thiele_innes:
                    s = self.indices['planets.A']
                    self.posteriors.A = self.posterior_sample[:,s]
                    self._priors.A = self.priors['Aprior']

                    s = self.indices['planets.B']
                    self.posteriors.B = self.posterior_sample[:,s]
                    self._priors.B = self.priors['Bprior']

                    s = self.indices['planets.F']
                    self.posteriors.F = self.posterior_sample[:,s]
                    self._priors.F = self.priors['Fprior']

                    s = self.indices['planets.G']
                    self.posteriors.G = self.posterior_sample[:,s]
                    self._priors.G = self.priors['Gprior']
                else:
                    #a0s 
                    s = self.indices['planets.a0']
                    self.posteriors.a0 = self.posterior_sample[:, s]
                    self._priors.a0 = self.priors['a0prior']

                    # omegas
                    s = self.indices['planets.w']
                    w = self.posteriors.w = self.posteriors.ω = self.posterior_sample[:, s]
                    self.posteriors.w_deg = self.posteriors.ω_deg = np.rad2deg(w)
                    self._priors.w = self.priors['omegaprior']

                    # cosi
                    s = self.indices['planets.cosi']
                    cosi = self.posteriors.cosi = self.posteriors.cosi = self.posterior_sample[:, s]
                    self.posteriors.i_deg = self.posteriors.i_deg = np.rad2deg(np.arccos(cosi))
                    self._priors.cosi = self.priors['cosiprior']

                    #Omegas
                    s = self.indices['planets.W']
                    W = self.posteriors.W = self.posteriors.Ω = self.posterior_sample[:, s]
                    self.posteriors.W_deg = self.posteriors.Ω_deg = np.rad2deg(W)
                    self._priors.W = self.priors['Omegaprior']

                if self.model is MODELS.RVGAIAmodel:
                    _Kfroma0 = np.vectorize(Kfroma0)
                    self.posteriors.K = _Kfroma0(self.posteriors.P, self.posteriors.a0,
                                                self.posteriors.e, self.posteriors.cosi,
                                                self.posteriors.plx.reshape(-1, 1))

            ### Also add ETV ones
            else:
                # periods
                s = self.indices['planets.P']
                self.posteriors.P = self.posterior_sample[:, s]
                self._priors.P = self.priors['Pprior']

                # amplitudes
                s = self.indices['planets.K']
                self.posteriors.K = self.posterior_sample[:, s]
                self._priors.K = self.priors['Kprior']

                # phases
                s = self.indices['planets.φ']
                φ = self.posteriors.φ = self.posteriors._phi = self.posterior_sample[:, s]
                self.posteriors.φ_deg = np.rad2deg(φ)
                self._priors.φ = self.priors['phiprior']

                # eccentricities
                s = self.indices['planets.e']
                self.posteriors.e = self.posterior_sample[:, s]
                self._priors.e = self.priors['eprior']

                # omegas
                s = self.indices['planets.w']
                w = self.posteriors.w = self.posteriors.ω = self.posterior_sample[:, s]
                self.posteriors.w_deg = self.posteriors.ω_deg = np.rad2deg(w)
                self._priors.w = self.priors['wprior']

                # times of periastron
                self.posteriors.Tp = (self.posteriors.P * self.posteriors.φ) / (2 * np.pi) + self.M0_epoch

                
                if self.model is MODELS.RVHGPMmodel:
                    s = self.indices['planets.i']
                    self.posteriors.i = self.posterior_sample[:, s]
                    self.posteriors.i_deg = np.rad2deg(self.posterior_sample[:, s])
                    self._priors.i = self.priors['iprior']

                    s = self.indices['planets.W']
                    W = self.posteriors.W = self.posteriors.Ω = self.posterior_sample[:, s]
                    self.posteriors.W_deg = self.posteriors.Ω_deg = np.rad2deg(W)
                    self._priors.W = self.priors['Omegaprior']


        if self.KO:
            self.posteriors.KO = posterior_holder()
            self._priors.KO = prior_holder()
            self.posteriors.KO.__doc__ = self._priors.KO.__doc__ = 'Known object parameters'
            self.posteriors.KO.P = self.KOpars[:, range(0*self.nKO, 1*self.nKO)]
            if self.model is MODELS.BINARIESmodel:
                self.posteriors.KO.K = self.KOpars[:, range(1*self.nKO, 2*self.nKO)]
                if self.double_lined:
                    self.posteriors.KO.q = self.KOpars[:, range(2 * self.nKO, 3 * self.nKO)]
                    self.posteriors.KO.φ = self.KOpars[:, range(3 * self.nKO, 4 * self.nKO)]
                    self.posteriors.KO.e = self.KOpars[:, range(4 * self.nKO, 5 * self.nKO)]
                    self.posteriors.KO.w = self.KOpars[:, range(5 * self.nKO, 6 * self.nKO)]
                    self.posteriors.KO.wdot = self.KOpars[:, range(6 * self.nKO, 7 * self.nKO)]
                    self.posteriors.KO.cosi = self.KOpars[:, range(7 * self.nKO, 8 * self.nKO)]
                else:
                    self.posteriors.KO.φ = self.KOpars[:, range(2*self.nKO, 3*self.nKO)]
                    self.posteriors.KO.e = self.KOpars[:, range(3*self.nKO, 4*self.nKO)]
                    self.posteriors.KO.w = self.KOpars[:, range(4*self.nKO, 5*self.nKO)]
                    self.posteriors.KO.wdot = self.KOpars[:, range(5*self.nKO, 6*self.nKO)]
                    self.posteriors.KO.cosi = self.KOpars[:, range(6*self.nKO, 7*self.nKO)]
            elif self.model in (MODELS.GAIAmodel,MODELS.RVGAIAmodel):
                self.posteriors.KO.a0 = self.KOpars[:, range(1*self.nKO, 2*self.nKO)]
                self.posteriors.KO.φ = self.KOpars[:, range(2*self.nKO, 3*self.nKO)]
                self.posteriors.KO.e = self.KOpars[:, range(3*self.nKO, 4*self.nKO)]
                self.posteriors.KO.w = self.KOpars[:, range(4*self.nKO, 5*self.nKO)]
                self.posteriors.KO.cosi = self.KOpars[:, range(5*self.nKO, 6*self.nKO)]
                self.posteriors.KO.W = self.KOpars[:, range(6*self.nKO, 7*self.nKO)]
            else:
                self.posteriors.KO.K = self.KOpars[:, range(1*self.nKO, 2*self.nKO)]
                self.posteriors.KO.φ = self.KOpars[:, range(2*self.nKO, 3*self.nKO)]
                self.posteriors.KO.e = self.KOpars[:, range(3*self.nKO, 4*self.nKO)]
                self.posteriors.KO.w = self.KOpars[:, range(4*self.nKO, 5*self.nKO)]
            for i in range(self.nKO):
                setattr(self._priors.KO, f'P{i}', self.priors[f'KO_Pprior_{i}'])

        if self.TR:
            self.posteriors.TR = posterior_holder()
            self._priors.TR = prior_holder()
            self.posteriors.TR.__doc__ = self._priors.TR.__doc__ = ("Transiting planet parameters")
            self.posteriors.TR.P = self.TRpars[:, range(0 * self.nTR, 1 * self.nTR)]
            self.posteriors.TR.K = self.TRpars[:, range(1 * self.nTR, 2 * self.nTR)]
            self.posteriors.TR.Tc = self.TRpars[:, range(2 * self.nTR, 3 * self.nTR)]
            self.posteriors.TR.e = self.TRpars[:, range(3 * self.nTR, 4 * self.nTR)]
            self.posteriors.TR.w = self.TRpars[:, range(4 * self.nTR, 5 * self.nTR)]
            for i in range(self.nTR):
                setattr(self._priors.TR, f'P{i}', self.priors[f'TR_Pprior_{i}'])
                setattr(self._priors.TR, f'K{i}', self.priors[f'TR_Kprior_{i}'])
                setattr(self._priors.TR, f'e{i}', self.priors[f'TR_eprior_{i}'])
                setattr(self._priors.TR, f'w{i}', self.priors[f'TR_wprior_{i}'])
                setattr(self._priors.TR, f'Tc{i}', self.priors[f'TR_Tcprior_{i}'])


        # # times of inferior conjunction (transit, if the planet transits)
        # f = np.pi / 2 - self.Omega
        # ee = 2 * np.arctan(
        #     np.tan(f / 2) * np.sqrt((1 - self.E) / (1 + self.E)))
        # Tc = self.Tp + self.T / (2 * np.pi) * (ee - self.E * np.sin(ee))
        # self.posteriors.Tc = Tc
        self._make_named_arrays()

    def get_medians(self):
        """ return the median values of all the parameters """
        if self.posterior_sample.shape[0] % 2 == 0:
            print(
                'Median is not a solution because number of samples is even!!')

        self.medians = np.median(self.posterior_sample, axis=0)
        self.means = np.mean(self.posterior_sample, axis=0)
        return self.medians, self.means

    def _select_samples(self, Np=None, mask=None, return_indices=False):
        if mask is None:
            mask = np.ones(self.sample.shape[0], dtype=bool)

        if Np is None:
            if return_indices:
                return np.where(mask)[0]
            return self.sample[mask].copy()
        else:
            mask_Np = self.sample[:, self.index_component] == Np
            if not mask_Np.any():
                raise ValueError(f'No samples with {Np} Keplerians')
            if return_indices:
                return np.where(mask & mask_Np)[0]
            return self.sample[mask & mask_Np].copy()

    def _select_posterior_samples(self, Np=None, mask=None, return_indices=False):
        if mask is None:
            mask = np.ones(self.ESS, dtype=bool)

        if Np is None:
            if return_indices:
                return np.where(mask)[0]
            return self.posterior_sample[mask].copy()
        else:
            mask_Np = self.posterior_sample[:, self.index_component] == Np
            if not mask_Np.any():
                raise ValueError(f'No posterior samples with {Np} Keplerians')
            if return_indices:
                return np.where(mask & mask_Np)[0]
            return self.posterior_sample[mask & mask_Np].copy()

    def log_prior(self, sample, debug=False):
        """ Calculate the log prior for a given sample

        Args:
            sample (array): sample for which to calculate the log prior
        
        Tip:
            To evaluate at all posterior samples, consider using
            
            ```python
            np.apply_along_axis(self.log_prior, 1, self.posterior_sample)
            ```
        """
        # logp = []
        # for p, v in zip(self.parameter_priors, sample):
        #     if p is None:
        #         # continue
        #         logp.append(0.0)
        #     else:
        #         try:
        #             logp.append(p.logpdf(v))
        #         except AttributeError:
        #             logp.append(p.logpmf(v))

        if debug:
            parameters = copy(self.parameters)
            for i, p in enumerate(self.parameter_priors):
                if p is None:
                    parameters.insert(i, None)

            for par, p, v in zip(parameters, self.parameter_priors, sample):
                print(f'{par:10s}' if par else '-', p, v, end='\t')
                if p:
                    print(p.logpdf(v), end='')
                print()

        logp = [
            p.logpdf(v) if p and v != 0.0 else 0.0
            for p, v in zip(self.parameter_priors, sample)
        ]

        # _np = int(sample[self.indices['np']])
        # st = self.indices['planets'].start
        # k = 0
        # for j in range(self._nd):
        #     for i in range(_np, self._mc):
        #         logp.pop(st + i + 3 * j - k)
        #         k += 1
        # return logp
        return np.sum(logp)

    def log_likelihood(self, sample, separate_instruments=False):
        if self.model != MODELS.RVmodel:
            raise NotImplementedError('only implemented for RVmodel')
        if self.multi:
            stellar_jitter = sample[self.indices['jitter']][0]
            jitter = sample[self.indices['jitter']][self.data.obs]
            var = self.data.e**2 + stellar_jitter**2 + jitter**2
        else:
            jitter = sample[self.indices['jitter']][0]
            var = self.data.e**2 + jitter**2

        model = self.full_model(sample)
        if self.studentt:
            nu = sample[self.indices['nu']]
            return students_t.logpdf(self.data.y, df=nu, loc=model, scale=np.sqrt(var)).sum()
            # TODO: why not the multivariate below?
            # return multivariate_t.logpdf(self.data.y, loc=model, shape=var, df=nu)
        else:
            if self.multi and separate_instruments:
                like = norm(loc=model, scale=np.sqrt(var)).logpdf(self.data.y)
                return np.array([like[self.data.obs == i].sum() for i in np.unique(self.data.obs)])
            else:
                return norm(loc=model, scale=np.sqrt(var)).logpdf(self.data.y).sum()
                # return multivariate_normal.logpdf(self.data.y, mean=model, cov=var)

    def log_posterior(self, sample, separate=False):
        logp = self.log_prior(sample)
        index = (self.posterior_sample == sample).sum(axis=1).argmax()
        logl = self.posterior_lnlike[index, 1]
        if separate:
            return logp + logl, logl, logp
        return logp + logl

    @lru_cache
    def map_sample(self, Np=None, mask=None, printit=True,
                   from_posterior=False):
        """
        Get the maximum a posteriori (MAP) sample.

        Note:
            This requires recalculation of the prior for all samples, so it can
            be a bit slow, depending on the number of posterior samples.

        Args:
            Np (int, optional):
                If given, select only samples with that number of planets.
            printit (bool, optional):
                Whether to print the sample.
            from_posterior (bool, optional): 
                If True, return the highest likelihood sample *from those that
                represent the posterior*. 
        """
        if from_posterior:
            samples = self._select_posterior_samples(Np, mask)
            ind = self._select_posterior_samples(Np, mask, return_indices=True)
            loglikes = self.posterior_lnlike[ind, 1]
        else:
            samples = self._select_samples(Np, mask)
            ind = self._select_samples(Np, mask, return_indices=True)
            loglikes = self.sample_info[ind, 1]

        logpriors = np.apply_along_axis(self.log_prior, 1, samples)
        logposts = logpriors + loglikes
        ind_map = logposts.argmax()
        self._map_sample = map_sample = samples[ind_map]
        logprior = logpriors[ind_map]
        loglike = loglikes[ind_map]
        logpost = logposts[ind_map]

        if printit:
            if from_posterior:
                print('Posterior sample with the highest posterior value')
            else:
                print('Sample with the highest posterior value')

            print(f'(logLike = {loglike:.2f}, logPrior = {logprior:.2f},', end=' ')
            print(f'logPost = {logpost:.2f})')

            if Np is not None:
                print(f'from samples with {Np} Keplerians only')

            msg = '-> might not be representative of the full posterior distribution\n'
            print(msg)

            self.print_sample(map_sample)

        return map_sample
    
    def maximum_likelihood(self, Np=None, from_posterior=False):
        """ Get the maximum log-likelihood value
        
        Args:
            Np (int, optional):
                If given, select only samples with that number of planets.
            from_posterior (bool, optional): 
                If True, return the highest likelihood value *from samples that
                represent the posterior*.
        """
        if self.sample_info is None and not self._lnlike_available:
            print('log-likelihoods are not available! '
                  'max_log_likelihood() doing nothing...')
            return

        if from_posterior:
            ind = self._select_posterior_samples(Np, return_indices=True)
            return self.posterior_lnlike[ind, 1].max()
        else:
            ind = self._select_samples(Np, return_indices=True)
            return self.sample_info[ind, 1].max()

    def maximum_likelihood_sample(self, Np=None, printit=True, mask=None,
                                  from_posterior=False, optimize=False):
        """
        Get the maximum likelihood sample. 
        
        By default, this is the highest likelihood sample found by DNest4.
        
        Note:
            If `from_posterior=True`, the returned sample may change, due to
            random choices, between different calls to `load_results`.

        Args:
            Np (int, optional):
                If given, select only samples with that number of planets.
            printit (bool, optional):
                Whether to print the sample
            from_posterior (bool, optional): 
                If True, return the highest likelihood sample *from those that
                represent the posterior*. 
            optimize (bool, optional):
                If True, optimize the likelihood, starting from the maximum
                likelihood sample.
        """
        if self.sample_info is None and not self._lnlike_available:
            print('log-likelihoods are not available! '
                  'maximum_likelihood_sample() doing nothing...')
            return

        if from_posterior:
            ind = self._select_posterior_samples(Np, mask, return_indices=True)
            loglike = self.posterior_lnlike[ind, 1]
            ind_maxlike = loglike.argmax()
            maxlike = loglike[ind_maxlike]
            pars = self.posterior_sample[ind][ind_maxlike]
        else:
            ind = self._select_samples(Np, mask, return_indices=True)
            loglike = self.sample_info[ind, 1]
            ind_maxlike = loglike.argmax()
            maxlike = loglike[ind_maxlike]
            pars = self.sample[ind][ind_maxlike]

        if optimize:
            # TODO: should take into account the prior (bounds)
            from scipy.optimize import minimize
            res = minimize(lambda p: -self.log_likelihood(p), pars)
            maxlike = -res.fun
            pars = res.x

        if printit:
            text = '(after optimization)' if optimize else ''

            if from_posterior:
                print(f'Posterior sample with the highest likelihood value {text}', end=' ')
            else:
                print(f'Sample with the highest likelihood value {text}', end=' ')

            print('(logL = {:.2f})'.format(maxlike))

            if Np is not None:
                print(f'from samples with {Np=} only')

            msg = '-> might not be representative '\
                  'of the full posterior distribution\n'
            print(msg)

            self.print_sample(pars)

        return pars

    def median_sample(self, Np=None, printit=True):
        """
        Get the median posterior sample.

        Warning:
            Unless *all* posteriors are Gaussian or otherwise well-behaved, the
            median sample is usually not the appropriate choice for plots, etc. 

        Args:
            Np (int, optional):
                If given, select only samples with that number of planets.
            printit (bool, optional):
                Whether to print the sample
        """

        if Np is None:
            pars = np.median(self.posterior_sample, axis=0)
        else:
            mask = self.posterior_sample[:, self.index_component] == Np
            pars = np.median(self.posterior_sample[mask, :], axis=0)

        if printit:
            print('Median posterior sample')
            if Np is not None:
                print('from samples with %d Keplerians only' % Np)
            print(
                '-> might not be representative of the full posterior distribution\n'
            )

            self.print_sample(pars)

        return pars

    def print_sample(self, p, star_mass=1.0, show_a=False, show_m=False,
                     mass_units='mjup', show_Tp=False, squeeze=False):

        if show_a or show_m:
            print('considering stellar mass:', star_mass)
            uncertainty_star_mass = False
            if isinstance(star_mass, tuple) or isinstance(star_mass, list):
                uncertainty_star_mass = True

        if self.multi:
            instruments = self.instruments
            instruments = [os.path.splitext(inst)[0] for inst in instruments]

        print('jitter:')
        if squeeze:
            if self.model is MODELS.RVFWHMmodel:
                inst = instruments + instruments
                data = self.n_instruments * ['RV'] + self.n_instruments * ['FWHM']
            else:
                inst = instruments
                data = self.n_instruments * ['']

            for i, jit in enumerate(p[self.indices['jitter']]):
                print(f'  {data[i]:5s} ({inst[i]}): {jit:.2f} m/s')
        else:
            if self.model in (MODELS.RVFWHMmodel, MODELS.RVFWHMRHKmodel):
                print(f'{"RV":>10s}', end=': ')
                print(p[self.indices['jitter']][:self.n_instruments])
                print(f'{"FWHM":>10s}', end=': ')
                print(p[self.indices['jitter']][self.n_instruments:2*self.n_instruments])

                if self.model is MODELS.RVFWHMRHKmodel:
                    print(f'{"RHK":>10s}', end=': ')
                    print(p[self.indices['jitter']][2*self.n_instruments:])
            else:
                print(' ', p[self.indices['jitter']])

        if hasattr(self, 'indicator_correlations') and self.indicator_correlations:
            print('indicator correlations:')
            c = p[self.indices['betas']]
            print(f'  {c}')

        npl = int(p[self.index_component])
        if npl > 0:
            print('number of planets: ', npl)
            print('orbital parameters: ', end='')

            if self.model in (MODELS.GAIAmodel, MODELS.RVGAIAmodel):
                if self.thiele_innes:
                    pars = ['P', 'phi', 'ecc', 'A', 'B', 'F', 'G']
                else:
                    pars = ['P', 'phi', 'ecc', 'a0', 'w', 'cosi', 'W']
            elif self.model is MODELS.RVHGPMmodel:
                pars = ['P', 'K', 'M0', 'e', 'w', 'i', 'W']
            else:
                pars = ['P', 'K', 'M0', 'e', 'w']

            n = self.n_dimensions

            if squeeze:
                print('\n' + 10 * ' ', end='')
                for i in range(npl):
                    print('%-10s' % ascii_lowercase[1 + i], end='')
                print()
                for par in pars:
                    par = 'φ' if par == 'M0' else par
                    print(f'  {par:2s}', ':    ', end='')
                    try:
                        sample_pars = p[self.indices[f'planets.{par}']]
                        for v in sample_pars:
                            print('%-10f' % v, end='')
                    except KeyError:
                        pass
                    print()
            else:
                if show_a:
                    pars.append('a')
                    n += 1
                if show_m:
                    pars.append('Mp')
                    n += 1

                print((n * ' {:>10s} ').format(*pars))

                for i in range(0, npl):
                    formatter = {'all': lambda v: f'{v:11.5f}'}
                    with np.printoptions(formatter=formatter, linewidth=1000):
                        planet_pars = p[self.indices['planets']][i::self.max_components]

                        if show_a or show_m:
                            P, K, M0, ecc, w = planet_pars
                            (m, _), a = get_planet_mass_and_semimajor_axis(
                                P, K, ecc, star_mass)

                            if uncertainty_star_mass:
                                m = m[0]
                                a = a[0]

                        if show_a:
                            planet_pars = np.append(planet_pars, a)
                        if show_m:
                            if mass_units != 'mjup':
                                if mass_units.lower() == 'mearth':
                                    m *= mjup2mearth
                            planet_pars = np.append(planet_pars, m)

                        s = str(planet_pars)
                        s = s.replace('[', '').replace(']', '')
                    s = s.rjust(20 + len(s))
                    print(s)

        if self.KO:
            print('number of known objects: ', self.nKO)
            print('orbital parameters: ', end='')
            extra_n = 0
            if self.model in (MODELS.GAIAmodel,MODELS.RVGAIAmodel):
                pars = ['P', 'a0', 'phi', 'ecc', 'w', 'cosi', 'W']
            elif self.model is MODELS.BINARIESmodel:
                if self.double_lined:
                    pars = ['P', 'K', 'q', 'M0', 'e', 'w', 'wdot','cosi']
                    extra_n = 3
                else:
                    pars = ['P', 'K', 'M0', 'e', 'w', 'wdot','cosi']
                    extra_n = 2
            else:
                pars = ('P', 'K', 'M0', 'e', 'w')
            print(((self.n_dimensions + extra_n) * ' {:>10s} ').format(*pars))

            for i in range(0, self.nKO):
                formatter = {'all': lambda v: f'{v:11.5f}'}
                with np.printoptions(formatter=formatter):
                    s = str(p[self.indices['KOpars']][i::self.nKO])
                    s = s.replace('[', '').replace(']', '')
                s = s.rjust(20 + len(s))
                print(s)

        if self.TR:
            print('number of transiting planets: ', self.nTR)
            print('orbital parameters: ', end='')

            pars = ('P', 'K', 'Tc', 'e', 'w')
            print((self.n_dimensions * ' {:>10s} ').format(*pars))

            for i in range(0, self.nTR):
                formatter = {'all': lambda v: f'{v:11.5f}'}
                with np.printoptions(formatter=formatter):
                    s = str(p[self.indices['TRpars']][i::self.nTR])
                    s = s.replace('[', '').replace(']', '')
                s = s.rjust(20 + len(s))
                print(s)

        if self.has_gp:
            print('GP parameters: ', end='')
            if self.model is MODELS.GPmodel:
                pars = ('η1', 'η2', 'η3', 'η4')
            elif self.model is MODELS.RVFWHMmodel:
                pars = ('η1 RV', 'η1 FWHM', 'η2', 'η3', 'η4')
            elif self.model is MODELS.RVFWHMRHKmodel:
                pars = ['η1 RV', 'η1 FWHM', 'η1 RHK']
                pars += ['η2'] if self.share_eta2 else ['η2 RV', 'η2 FWHM', 'η2 RHK']
                pars += ['η3'] if self.share_eta3 else ['η3 RV', 'η3 FWHM', 'η3 RHK']
                pars += ['η4'] if self.share_eta4 else ['η4 RV', 'η4 FWHM', 'η4 RHK']
            else:
                pars = self._parameters[self.indices['GPpars']]

            pars = [p.replace('eta', 'η') for p in pars]

            if squeeze:
                print()
                values = p[self.indices['GPpars']]
                for par, v in zip(pars, values):
                    print(f'  {par:8s}:', v)
            else:
                print((len(pars) * ' {:>10s} ').format(*pars))
                formatter = {'all': lambda v: f'{v:11.5f}'}
                with np.printoptions(formatter=formatter):
                    s = str(p[self.indices['GPpars']])
                    s = s.replace('[', '').replace(']', '')
                s = s.rjust(15 + len(s))
                print(s)

            if self.model is MODELS.SPLEAFmodel:
                def print_pars(pars, key, offset=0):
                    print(offset * ' ', end='')
                    print((len(pars) * ' {:>10s} ').format(*pars))
                    formatter = {'all': lambda v: f'{v:11.5f}'}
                    with np.printoptions(formatter=formatter):
                        s = str(p[self.indices[key]])
                        s = s.replace('[', '').replace(']', '')
                    s = s.rjust(15 + len(s))
                    print(s)

                names = ['RV'] + list(self._extra_data_names[::2])
                pars1 = [f'α{i} ({n})' for i, n in enumerate(names)]
                pars2 = [f'β{i} ({n})' for i, n in enumerate(names)]
                if squeeze:
                    raise NotImplementedError
                else:
                    print_pars(pars1, 'GP_alphas', 15)
                    print_pars(pars2, 'GP_betas', 15)


        if self.trend:
            names = ('slope', 'quadr', 'cubic')
            units = ['m/s/yr', 'm/s/yr²', 'm/s/yr³']
            trend = p[self.indices['trend']].copy()
            # transfrom from /day to /yr
            trend *= 365.25**np.arange(1, self.trend_degree + 1)

            for name, unit, trend_par in zip(names, units, trend):
                print(name + ':', '%-8.5f' % trend_par, unit)

        if self.model is MODELS.RVFWHMmodel and self.trend_fwhm:
            names = ('slope_fwhm', 'quadr_fwhm', 'cubic_fwhm')
            units = ['m/s/yr', 'm/s/yr²', 'm/s/yr³']
            trend = p[self.indices['trend_fwhm']].copy()
            # transfrom from /day to /yr
            trend *= 365.25**np.arange(1, self.trend_fwhm_degree + 1)

            for name, unit, trend_par in zip(names, units, trend):
                print(name + ':', '%-8.5f' % trend_par, unit)


        if self.multi:
            ni = self.n_instruments - 1
            print('instrument offsets: ', end=' ')
            # print('(relative to %s) ' % self.data_file[-1])
            print('(relative to %s) ' % instruments[-1])
            s = 20 * ' '
            s += (ni * ' {:20s} ').format(*instruments)
            print(s)

            i = self.indices['inst_offsets']
            s = 20 * ' '
            s += (ni * ' {:<20.3f} ').format(*p[i])
            print(s)

        if self.model != MODELS.GAIAmodel:
            if self.model is MODELS.RVFWHMmodel:
                print('cfwhm: ', p[self.indices['cfwhm']])
            if self.model is MODELS.RVFWHMRHKmodel:
                print('crhk: ', p[self.indices['crhk']])
            if self.model is MODELS.BINARIESmodel and self.double_lined:
                print('vsys_sec: ', p[-2])
            print('vsys: ', p[-1])


    def _sort_planets_by_amplitude(self, sample, decreasing=True):
        new_sample = sample.copy()
        ind = np.argsort(new_sample[self.indices['planets.K']])
        if decreasing:
            ind = ind[::-1]
        nd = self.n_dimensions
        mc = self.max_components
        pars = new_sample[self.indices['planets']]
        for i in range(nd):
            pars[i * mc:(i + 1) * mc] = pars[i * mc:(i + 1) * mc][ind]
        return new_sample

    def _get_tt(self, N=1000, over=0.1):
        """
        Create array for model prediction plots. This simply returns N
        linearly-spaced times from t[0]-over*Tspan to t[-1]+over*Tspan.
        """
        start = self.data.t.min() - over * np.ptp(self.data.t)
        end = self.data.t.max() + over * np.ptp(self.data.t)
        return np.linspace(start, end, N)

    def _get_ttGP(self, N=1000, over=0.1):
        """ Create array of times for GP prediction plots. """
        kde = gaussian_kde(self.data.t)
        if N > self.data.N:
            ttGP = kde.resample(N - self.data.N).reshape(-1)
        else:
            ttGP = kde.resample(N).reshape(-1)
        # constrain ttGP within observed times (+- over), to not waste
        ttGP = (ttGP + self.data.t[0]) % ((1 + over) * np.ptp(self.data.t)) + self.data.t[0]
        # add the observed times as well
        ttGP = np.r_[ttGP, self.data.t]
        ttGP.sort()  # in-place
        return ttGP

    def eval_model(self, sample, t=None, include_planets=True, 
                   include_known_object=True, include_transiting_planet=True,
                   include_indicator_correlations=True,
                   include_trend=True, single_planet: int = None,
                   except_planet: Union[int, List] = None):
        """
        Evaluate the deterministic part of the model at one posterior `sample`.

        If `t` is None, use the observed times.
        
        Note:
            Instrument offsets are only added if `t` is None, but the systemic
            velocity is always added.

        Note:
            This function does *not* evaluate the GP component of the model.

        Args:
            sample (array): 
                One posterior sample, with shape (npar,)
            t (array):
                Times at which to evaluate the model, or None to use observed
                times
            include_planets (bool):
                Whether to include the contribution from the planets
            include_known_object (bool):
                Whether to include the contribution from the known object
                planet(s)
            include_transiting_planet (bool):
                Whether to include the contribution from the transiting
                planet(s)
            include_indicator_correlations (bool):
                Whether to include the indicator correlation model
            include_trend (bool):
                Whether to include the contribution from the trend
            single_planet (int):
                Index of a single planet to *include* in the model, starting at
                1. Use positive values (1, 2, ...) for the Np planets and
                negative values (-1, -2, ...) for the known object and
                transiting planets.
            except_planet (Union[int, List]):
                Index (or list of indices) of a single planet to *exclude* from
                the model, starting at 1. Use positive values (1, 2, ...) for
                the Np planets and negative values (-1, -2, ...) for the known
                object and transiting planets.

        Tip:
            To evaluate at all posterior samples, consider using

            ```python
            np.apply_along_axis(self.eval_model, 1, self.posterior_sample)
            ```
        """
        if sample.shape[0] != self.posterior_sample.shape[1]:
            n1 = sample.shape[0]
            n2 = self.posterior_sample.shape[1]
            msg = '`sample` has wrong dimensions, expected %d got %d' % (n2, n1)
            raise ValueError(msg)

        data_t = False
        if t is None or t is self.data.t:
            t = self.data.t.copy()
            data_t = True

        ONE_D_MODELS = [MODELS.RVmodel, MODELS.GPmodel, MODELS.RVHGPMmodel, MODELS.RVGAIAmodel]

        if self.model is MODELS.RVFWHMmodel:
            v = np.zeros((2, t.size))
        elif self.model is MODELS.RVFWHMRHKmodel:
            v = np.zeros((3, t.size))
        elif self.model is MODELS.SPLEAFmodel:
            v = np.zeros((self.nseries, t.size))
        elif self.model is MODELS.BINARIESmodel:
            if self.double_lined:
                v = np.zeros((2,t.size))
            else:
                v = np.zeros_like(t)
                ONE_D_MODELS.append(MODELS.BINARIESmodel)
        else:
            v = np.zeros_like(t)

        if self.model is MODELS.RVGAIAmodel:
            da,dd,mua,mud,plx = sample[self.indices['astrometric_solution']]

        if include_planets:
            if single_planet and except_planet:
                raise ValueError("'single_planet' and 'except_planet' "
                                 "cannot be used together")
            if single_planet == 0:
                raise ValueError("'single_planet' should not be 0")

            # except_planet should be a list to exclude more than one planet
            if except_planet is not None:
                if isinstance(except_planet, int):
                    if except_planet == 0:
                        raise ValueError("'except_planet' should not be 0")
                    except_planet = [except_planet]

            # known_object ? 
            # For BINARIESmodel and double_lined especially, need to deal with
            # the extra parameters in those models and using the correct Keplerian
            # also for the RVGAIA model, converting a0 into K
            pj = 0
            if self.KO and include_known_object:
                pars = sample[self.indices['KOpars']].copy()
                for j in range(self.nKO):
                    pj += 1
                    if single_planet is not None:
                        if pj != -single_planet:
                            continue
                    if except_planet is not None:
                        if -pj in except_planet:
                            continue

                    P = pars[j + 0 * self.nKO]
                    if self.model is MODELS.RVGAIAmodel:
                        a0 = pars[j + 1 * self.nKO]
                        phi = pars[j + 2 * self.nKO]
                        ecc = pars[j + 3 * self.nKO]
                        w = pars[j + 4 * self.nKO]
                        cosi = pars[j + 5 * self.nKO]
                        K = Kfroma0(P,a0,ecc,cosi,plx)
                    # t0 = (P * phi) / (2. * np.pi) + self.M0_epoch
                    elif self.model is MODELS.BINARIESmodel:
                        K = pars[j + 1 * self.nKO]
                        if self.double_lined:
                            q = pars[j + 2 * self.nKO]
                            phi = pars[j + 3 * self.nKO]
                            ecc = pars[j + 4 * self.nKO]
                            w = pars[j + 5 * self.nKO]
                            wdot = pars[j + 6 * self.nKO]
                            cosi = pars[j + 7 * self.nKO]
                        else:
                            phi = pars[j + 2 * self.nKO]
                            ecc = pars[j + 3 * self.nKO]
                            w = pars[j + 4 * self.nKO]
                            wdot = pars[j + 5 * self.nKO]
                            cosi = pars[j + 6 * self.nKO]
                    else:
                        K = pars[j + 1 * self.nKO]
                        phi = pars[j + 2 * self.nKO]
                        ecc = pars[j + 3 * self.nKO]
                        w = pars[j + 4 * self.nKO]
                    if self.model not in ONE_D_MODELS:
                        if self.model is MODELS.BINARIESmodel:
                            if j==0:
                                Panom = period_correction(P,wdot)
                                v0,v1 = post_keplerian_sb2(t, Panom, K, q, ecc, w, wdot, phi, self.M0_epoch, cosi, self.star_radius, self.binary_radius, self.relativistic_correction, self.tidal_correction)
                                v[0] += v0
                                v[1] += v1
                            else:
                                v[0] += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)
                                v[1] += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)
                        else:
                            v[0] += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)
                    else:
                        if self.model is MODELS.BINARIESmodel and j==0:
                            Panom = period_correction(P,wdot)
                            v += post_keplerian(t, Panom, K, ecc, w, wdot, phi, self.M0_epoch, cosi, self.star_mass, self.binary_mass, self.star_radius, self.relativistic_correction, self.tidal_correction)
                        else:
                            v += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)

            # transiting planet ?
            if hasattr(self, 'TR') and self.TR and include_transiting_planet:
                pars = sample[self.indices['TRpars']].copy()
                for j in range(self.nTR):
                    pj += 1
                    if single_planet is not None:
                        if pj != -single_planet:
                            continue
                    if except_planet is not None:
                        if -pj in except_planet:
                            continue

                    P = pars[j + 0 * self.nTR]
                    K = pars[j + 1 * self.nTR]
                    Tc = pars[j + 2 * self.nTR]
                    ecc = pars[j + 3 * self.nTR]
                    w = pars[j + 4 * self.nTR]
                    
                    f = np.pi/2 - w # true anomaly at conjunction
                    E = 2.0 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc))) # eccentric anomaly at conjunction
                    M = E - ecc * np.sin(E) # mean anomaly at conjunction
                    if self.model not in ONE_D_MODELS:
                        v[0] += keplerian(t, P, K, ecc, w, M, Tc)
                    else:
                        v += keplerian(t, P, K, ecc, w, M, Tc)

            # get the planet parameters for this sample
            pars = sample[self.indices['planets']].copy()
            
            # how many planets in this sample?
            # nplanets = pars.size / self.n_dimensions
            nplanets = (pars[:self.max_components] != 0).sum()

            # add the Keplerians for each of the planets
            for j in range(int(nplanets)):

                if single_planet is not None:
                    if j + 1 != single_planet:
                        continue
                if except_planet is not None:
                    if j + 1 in except_planet:
                        continue

                P = pars[j + 0 * self.max_components]
                if P == 0.0:
                    continue
                if self.model is MODELS.RVGAIAmodel:
                    phi = pars[j + 1 * self.max_components]
                    ecc = pars[j + 2 * self.max_components]
                    a0 = pars[j + 3 * self.max_components]
                    w = pars[j + 4 * self.max_components]
                    cosi = pars[j + 5 * self.max_components]
                    K = Kfroma0(P,a0,ecc,cosi,plx)
                else:
                    K = pars[j + 1 * self.max_components]
                    phi = pars[j + 2 * self.max_components]
                    ecc = pars[j + 3 * self.max_components]
                    w = pars[j + 4 * self.max_components]
                # print(P, K, ecc, w, phi, self.M0_epoch)

                if self.model not in ONE_D_MODELS:
                    v[0] += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)
                    if self.model is MODELS.BINARIESmodel:
                        v[1] += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)
                else:
                    v += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)

        ni = self.n_instruments

        # systemic velocity (and C2) for this sample
        if self.model is MODELS.RVFWHMmodel:
            C = np.c_[sample[self.indices['vsys']], sample[self.indices['cfwhm']]]
            v += C.reshape(-1, 1)
        elif self.model is MODELS.RVFWHMRHKmodel:
            C = np.c_[sample[self.indices['vsys']], sample[self.indices['cfwhm']], sample[self.indices['crhk']]]
            v += C.reshape(-1, 1)
        elif self.model is MODELS.SPLEAFmodel:
            zp = sample[self.indices['zero_points']]
            C = np.r_[sample[self.indices['vsys']], zp[ni - 1::ni]]
            v += C.reshape(-1, 1)
        elif self.model is MODELS.BINARIESmodel:
            if self.double_lined:
                C = np.c_[sample[self.indices['vsys']], sample[self.indices['vsys_sec']]]
                v += C.reshape(-1, 1)
            else:
                v += sample[self.indices['vsys']]
        else:
            v += sample[self.indices['vsys']]

        # if evaluating at the same times as the data, add instrument offsets
        # otherwise, don't
        if self.multi and data_t:  # and len(self.data_file) > 1:
            offsets = sample[self.indices['inst_offsets']]
            ii = self.data.obs.astype(int) - 1

            if self.model in (MODELS.RVFWHMmodel, MODELS.RVFWHMRHKmodel):
                offsets = np.pad(offsets.reshape(-1, ni - 1), ((0, 0), (0, 1)))
                v += np.take(offsets, ii, axis=1)
            elif self.model is MODELS.SPLEAFmodel:
                # complicated code to get
                # [rv_offset1,   rv_offset2, ..., 0.0,   zero_point1, zero_point2, ...]
                # [inst1, inst1, ...,             instn, ai_inst1,    ai_inst2, ...]
                zero_points = sample[self.indices['zero_points']]
                offsets = np.r_[offsets, 0.0, zero_points]
                offsets = np.pad(-np.diff(offsets)[::ni].reshape(-1, 1), ((0, 0), (0, 1)))
                v += np.take(offsets.reshape(-1, ni), ii, axis=1)
            elif self.model is MODELS.BINARIESmodel:
                if self.double_lined:
                    offsets = np.pad(offsets.reshape(-1, ni - 1), ((0, 0), (0, 1)))
                    v += np.take(offsets, ii, axis=1)
                else:
                    offsets = np.pad(offsets, (0, 1))
                    v += np.take(offsets, ii)
            else:
                offsets = np.pad(offsets, (0, 1))
                v += np.take(offsets, ii)

        # add the trend, if present
        if include_trend:
            if self.trend:
                trend_par = sample[self.indices['trend']]
                # polyval wants coefficients in reverse order, and vsys was already
                # added so the last coefficient is 0
                trend_par = np.r_[trend_par[::-1], 0.0]
                if self.model in (MODELS.RVFWHMmodel, MODELS.RVFWHMRHKmodel):
                    v[0, :] += np.polyval(trend_par, t - self.data.tmiddle)
                else:
                    v += np.polyval(trend_par, t - self.data.tmiddle)
            
            if self.model in (MODELS.RVFWHMmodel, ) and self.trend_fwhm:
                trend_par = sample[self.indices['trend_fwhm']]
                trend_par = np.r_[trend_par[::-1], 0.0]
                v[1, :] += np.polyval(trend_par, t - self.data.tmiddle)

        # TODO: check if _extra_data is always read correctly
        if hasattr(self, 'indicator_correlations') and self.indicator_correlations and include_indicator_correlations:
            betas = sample[self.indices['betas']].copy()
            interp_u = np.zeros_like(t)
            for i, (c, ai) in enumerate(zip(betas, self.activity_indicators)):
                if ai != '':
                    interp_u += c * np.interp(t, self.data.t, self._extra_data[i])
            v += interp_u

        return v

    def planet_model(self, sample, t=None, 
                     include_known_object=True, include_transiting_planet=True,
                     single_planet=None, except_planet=None):
        """
        Evaluate the planet part of the model at one posterior `sample`.
        
        If `t` is None, use the observed times. 
        
        Note:
            This function does *not* evaluate the GP component of the model nor
            the systemic velocity and instrument offsets.

        Args:
            sample (array):
                One posterior sample, with shape (npar,)
            t (array):
                Times at which to evaluate the model, or None to use observed
                times
            include_known_object (bool):
                Whether to include the contribution from the known object
                planet(s)
            include_transiting_planet (bool):
                Whether to include the contribution from the transiting
                planet(s)
            single_planet (int):
                Index of a single planet to *include* in the model, starting at
                1. Use positive values (1, 2, ...) for the Np planets and
                negative values (-1, -2, ...) for the known object and
                transiting planets.
            except_planet (Union[int, List]):
                Index (or list of indices) of a single planet to *exclude* from
                the model, starting at 1. Use positive values (1, 2, ...) for
                the Np planets and negative values (-1, -2, ...) for the known
                object and transiting planets.

        Tip:
            To evaluate at all posterior samples, consider using

            ```python
            np.apply_along_axis(self.planet_model, 1, self.posterior_sample)
            ```
        
        Examples:
            To get the Keplerian contribution from the first planet in a
            posterior sample `p` use:

            ```python
            res.planet_model(p, single_planet=1)
            ```

            For, e.g., the second known object in the model, use:

            ```python
            res.planet_model(p, single_planet=-2)
            ```

            or to get the contributions from all planets _except_ that one

            ```python
            res.planet_model(p, except_planet=-2)
            ```

        """
        if sample.shape[0] != self.posterior_sample.shape[1]:
            n1 = sample.shape[0]
            n2 = self.posterior_sample.shape[1]
            msg = '`sample` has wrong dimensions, expected %d got %d' % (n2, n1)
            raise ValueError(msg)

        if t is None or t is self.data.t:
            t = self.data.t.copy()

        if self.model is MODELS.RVFWHMmodel:
            v = np.zeros((2, t.size))
        elif self.model is MODELS.RVFWHMRHKmodel:
            v = np.zeros((3, t.size))
        elif self.model is MODELS.BINARIESmodel:
            if self.double_lined:
                v = np.zeros((2,t.size))
            else:
                v = np.zeros_like(t)
        else:
            v = np.zeros_like(t)

        if single_planet and except_planet:
            raise ValueError("'single_planet' and 'except_planet' "
                             "cannot be used together")
        if single_planet == 0:
            raise ValueError("'single_planet' should not be 0")
        
        if self.model is MODELS.RVGAIAmodel:
            da,dd,mua,mud,plx = sample[self.indices['astrometric_solution']]

        # except_planet should be a list to exclude more than one planet
        if except_planet is not None:
            except_planet = np.atleast_1d(except_planet)

        pj = 0
        # known_object ?
        if self.KO and include_known_object:
            pars = sample[self.indices['KOpars']].copy()
            for j in range(self.nKO):
                pj += 1
                if single_planet is not None:
                    if pj != -single_planet:
                        continue
                if except_planet is not None:
                    if self.model is MODELS.BINARIESmodel and j==0:
                        pass
                    elif -pj in except_planet:
                        continue

                P = pars[j + 0 * self.nKO]
                # t0 = (P * phi) / (2. * np.pi) + self.M0_epoch
                if self.model is MODELS.BINARIESmodel:
                    K = pars[j + 1 * self.nKO]
                    if self.double_lined:
                        q = pars[j + 2 * self.nKO]
                        phi = pars[j + 3 * self.nKO]
                        ecc = pars[j + 4 * self.nKO]
                        w = pars[j + 5 * self.nKO]
                        wdot = pars[j + 6 * self.nKO]
                        cosi = pars[j + 7 * self.nKO]
                    else:
                        phi = pars[j + 2 * self.nKO]
                        ecc = pars[j + 3 * self.nKO]
                        w = pars[j + 4 * self.nKO]
                        wdot = pars[j + 5 * self.nKO]
                        cosi = pars[j + 6 * self.nKO]
                elif self.model is MODELS.RVGAIAmodel:
                    a0 = pars[j + 1 * self.nKO]
                    phi = pars[j + 2 * self.nKO]
                    ecc = pars[j + 3 * self.nKO]
                    w = pars[j + 4 * self.nKO]
                    cosi = pars[j + 5 * self.nKO]
                    K = Kfroma0(P,a0,ecc,cosi,plx)
                else:
                    K = pars[j + 1 * self.nKO]
                    phi = pars[j + 2 * self.nKO]
                    ecc = pars[j + 3 * self.nKO]
                    w = pars[j + 4 * self.nKO]
                if self.model is MODELS.BINARIESmodel:
                    if self.double_lined:
                        if j==0:
                            Panom = period_correction(P,wdot)
                            v0,v1 = post_keplerian_sb2(t, Panom, K, q, ecc, w, wdot, phi, self.M0_epoch, cosi, self.star_radius, self.binary_radius, self.relativistic_correction, self.tidal_correction)
                            v[0] += v0
                            v[1] += v1
                            if except_planet is not None:
                                if -pj in except_planet:
                                    v2,v3 = post_keplerian_sb2(t, P, K, q, ecc, w, 0, phi, self.M0_epoch, cosi, self.star_radius, self.binary_radius, False, False)
                                    v[0] -= v2
                                    v[1] -= v3

                        else:
                            v[0] += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)
                            v[1] += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)
                    else:
                        if j==0:
                            Panom = period_correction(P,wdot)
                            v += post_keplerian(t, Panom, K, ecc, w, wdot, phi, self.M0_epoch, cosi, self.star_mass, self.binary_mass, self.star_radius, self.relativistic_correction, self.tidal_correction)
                            if except_planet is not None:
                                if -pj in except_planet:
                                    v -= post_keplerian(t, P, K, ecc, w, 0, phi, self.M0_epoch, cosi, self.star_mass, self.binary_mass, self.star_radius, False, False)
                        else:
                            v += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)
                else:
                    v += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)

        # transiting planet ?
        if hasattr(self, 'TR') and self.TR and include_transiting_planet:
            pars = sample[self.indices['TRpars']].copy()
            for j in range(self.nTR):
                pj += 1
                if single_planet is not None:
                    if pj != -single_planet:
                        continue
                if except_planet is not None:
                    if -pj in except_planet:
                        continue

                P = pars[j + 0 * self.nTR]
                K = pars[j + 1 * self.nTR]
                Tc = pars[j + 2 * self.nTR]
                ecc = pars[j + 3 * self.nTR]
                w = pars[j + 4 * self.nTR]
                
                f = np.pi/2 - w # true anomaly at conjunction
                E = 2.0 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc))) # eccentric anomaly at conjunction
                M = E - ecc * np.sin(E) # mean anomaly at conjunction
                v += keplerian(t, P, K, ecc, w, M, Tc)

        # get the planet parameters for this sample
        pars = sample[self.indices['planets']].copy()

        # how many planets in this sample?
        # nplanets = pars.size / self.n_dimensions
        nplanets = (pars[:self.max_components] != 0).sum()

        # add the Keplerians for each of the planets
        for j in range(int(nplanets)):

            if single_planet is not None:
                if j + 1 != single_planet:
                    continue
            if except_planet is not None:
                if j + 1 in except_planet:
                    continue

            P = pars[j + 0 * self.max_components]
            if P == 0.0:
                continue
            if self.model is MODELS.RVGAIAmodel:
                phi = pars[j + 1 * self.max_components]
                # t0 = (P * phi) / (2. * np.pi) + self.M0_epoch
                ecc = pars[j + 2 * self.max_components]
                a0 = pars[j + 3 * self.max_components]
                w = pars[j + 4 * self.max_components]
                cosi = pars[j + 5 * self.max_components]
                K = Kfroma0(P,a0,ecc,cosi,plx)
            else:
                K = pars[j + 1 * self.max_components]
                phi = pars[j + 2 * self.max_components]
                ecc = pars[j + 3 * self.max_components]
                w = pars[j + 4 * self.max_components]
            if self.model in (MODELS.RVFWHMmodel, MODELS.RVFWHMRHKmodel):
                v[0, :] += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)
            else:
                v += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)

        return v

    def stochastic_model(self, sample, t=None, return_std=False, return_cov=False,
                         derivative=False, include_jitters=True, **kwargs):
        """
        Evaluate the stochastic part of the model (GP) at one posterior sample.
        This function returns the mean of the GP predictive distribution.
        
        If `t` is None, use the observed times.  

        Note:
            Instrument offsets are only added if `t` is None, but the systemic
            velocity is always added.

        Args:
            sample (array):
                One posterior sample, with shape (npar,)
            t (ndarray, optional):
                Times at which to evaluate the model, or `None` to use observed
                times
            return_std (bool, optional):
                Whether to return the standard deviation of the predictive.
                Default is False.
            return_cov (bool, optional):
                Whether to return the full covariance matrix of the predictive.
                Default is False
            derivative (bool, optional):
                Return the first time derivative of the GP prediction instead
            include_jitters (bool, optional):
                Whether to include the jitter values in `sample` in the
                prediction

        Tip:
            To evaluate at all posterior samples, consider using

            ```python
            np.apply_along_axis(res.stochastic_model, 1, res.posterior_sample)
            ```
        """
        from .. import GP

        if sample.shape[0] != self.posterior_sample.shape[1]:
            n1 = sample.shape[0]
            n2 = self.posterior_sample.shape[1]
            msg = '`sample` has wrong dimensions, should be %d got %d' % (n2, n1)
            raise ValueError(msg)

        if t is None or t is self.data.t:
            t = self.data.t.copy()

        if not self.has_gp:
            if return_std:
                return np.zeros_like(t), np.zeros_like(t)
            else:
                return np.zeros_like(t)

        if self.model is MODELS.RVFWHMmodel:
            D = np.vstack((self.data.y, self.data.y2))
            r = D - self.eval_model(sample)
            GPpars = sample[self.indices['GPpars']]

            η1RV, η1FWHM, η2RV, η2FWHM, η3RV, η3FWHM, η4RV, η4FWHM = GPpars[self._GP_par_indices]

            self.GP1.kernel.pars = np.array([η1RV, η2RV, η3RV, η4RV])
            if include_jitters:
                # get jitters per instrument in an array
                jRV = sample[self.indices['jitter']][:self.n_instruments]
                jRV = jRV[self.data.obs.astype(int) - 1]
                self.GP1.white_noise = jRV
            else:
                self.GP1.white_noise = 0.0

            self.GP2.kernel.pars = np.array([η1FWHM, η2FWHM, η3FWHM, η4FWHM])
            if include_jitters:
                # get jitters per instrument in an array
                jFW = sample[self.indices['jitter']][self.n_instruments:2*self.n_instruments]
                jFW = jFW[self.data.obs.astype(int) - 1]
                self.GP2.white_noise = jFW
            else:
                self.GP2.white_noise = 0.0

            if derivative:
                out0 = self.GP1.derivative(r[0], t, return_std=return_std)
                out1 = self.GP2.derivative(r[1], t, return_std=return_std)
            else:
                out0 = self.GP1.predict(r[0], t, return_std=return_std)
                out1 = self.GP2.predict(r[1], t, return_std=return_std)

            if return_std:
                return (
                    np.vstack([out0[0], out1[0]]),
                    np.vstack([out0[1], out1[1]])
                )
            else:
                return np.vstack([out0, out1])

        elif self.model is MODELS.RVFWHMRHKmodel:
            D = np.vstack((self.data.y, self.data.y2, self.data.y3))
            r = D - self.eval_model(sample)
            GPpars = sample[self.indices['GPpars']]
            pars = GPpars[self._GP_par_indices]
            η1RV, η1FWHM, η1RHK, *pars = pars
            η2RV, η2FWHM, η2RHK, *pars = pars
            η3RV, η3FWHM, η3RHK, *pars = pars
            η4RV, η4FWHM, η4RHK, *pars = pars
            if self.magnetic_cycle_kernel:
                η5RV, η5FWHM, η5RHK, η6, η7, *pars = pars

            parsRV = [η1RV, η2RV, η3RV, η4RV]
            if self.magnetic_cycle_kernel:
                parsRV += [η5RV, η6, η7]

            parsFWHM = [η1FWHM, η2FWHM, η3FWHM, η4FWHM]
            if self.magnetic_cycle_kernel:
                parsFWHM += [η5FWHM, η6, η7]

            parsRHK = [η1RHK, η2RHK, η3RHK, η4RHK]
            if self.magnetic_cycle_kernel:
                parsRHK += [η5RHK, η6, η7]

            self.GP1.kernel.pars = np.array(parsRV)
            if include_jitters:
                # get jitters per instrument in an array
                jRV = sample[self.indices['jitter']][:self.n_instruments]
                jRV = jRV[self.data.obs.astype(int) - 1]
                self.GP1.white_noise = jRV
            else:
                self.GP1.white_noise = 0.0

            self.GP2.kernel.pars = np.array(parsFWHM)
            if include_jitters:
                # get jitters per instrument in an array
                jFW = sample[self.indices['jitter']][self.n_instruments:2*self.n_instruments]
                jFW = jFW[self.data.obs.astype(int) - 1]
                self.GP2.white_noise = jFW
            else:
                self.GP2.white_noise = 0.0

            self.GP3.kernel.pars = np.array(parsRHK)
            if include_jitters:
                # get jitters per instrument in an array
                jRHK = sample[self.indices['jitter']][2*self.n_instruments:]
                jRHK = jRHK[self.data.obs.astype(int) - 1]
                self.GP3.white_noise = jRHK
            else:
                self.GP3.white_noise = 0.0

            if derivative:
                out1 = self.GP1.derivative(r[0], t, return_std=return_std)
                out2 = self.GP2.derivative(r[1], t, return_std=return_std)
                out3 = self.GP3.derivative(r[2], t, return_std=return_std)
            else:
                out1 = self.GP1.predict(r[0], t, return_std=return_std)
                out2 = self.GP2.predict(r[1], t, return_std=return_std)
                out3 = self.GP3.predict(r[2], t, return_std=return_std)

            if return_std:
                return (
                    np.vstack([out1[0], out2[0], out3[0]]),
                    np.vstack([out1[1], out1[1], out3[1]])
                )
            else:
                return np.vstack([out1, out2, out3])


        elif self.model is MODELS.SPLEAFmodel:
            if self.nseries > 1:
                D = np.r_[self.data.y.reshape(1,-1), self._extra_data[::2, :]]
                errors = np.r_[self.data.e.reshape(1,-1), self._extra_data[1::2, :]]
            else:
                D = self.data.y.copy().reshape(1,-1)
                errors = self.data.e.copy().reshape(1,-1)

            resid = D - self.eval_model(sample)

            #TODO: move to beginning of file
            from spleaf import cov, term
            tfull, resid_full, efull, obs_full, series_index = cov.merge_series(
                self.nseries * [self.data.t],
                resid,
                errors,
                self.nseries * [self.data.obs],
            )
            a, b = np.ones(self.nseries), np.ones(self.nseries)

            jit = {
                f'jit{i}{j}': term.InstrumentJitter(series_index[i][obs_full[series_index[i]] == j+1], 1.0)
                for i in range(self.nseries)
                for j in range(self.n_instruments)
            }

            k = {
                GP.KernelType.spleaf_matern32: term.Matern32Kernel(1.0, 1.0),
                GP.KernelType.spleaf_sho: term.SHOKernel(1.0, 1.0, 1.0),
                GP.KernelType.spleaf_mep: term.MEPKernel(1.0, 1.0, 1.0, 1.0),
                GP.KernelType.spleaf_esp: term.ESPKernel(1.0, 1.0, 1.0, 1.0, nharm=3),
            }[self.kernel]

            C = cov.Cov(tfull, 
                        **jit,
                        err=term.Error(efull),
                        gp=term.MultiSeriesKernel(k, series_index, a, b)
            )

            if self.kernel in (GP.KernelType.spleaf_mep, GP.KernelType.spleaf_esp):
                gp_indices = list(range(*self.indices['GPpars'].indices(sample.size)))
                gp_indices[1:3] = gp_indices[1:3][::-1]
            else:
                gp_indices = self.indices['GPpars']
            gp_pars = sample[gp_indices].copy()
            gp_pars[-1] *= 0.5
            # gp_pars = [1.0, 9.746, 17.95, 0.49]
            # print(gp_pars)

            # alphas and betas
            alpha_beta = np.r_[
                sample[self.indices['GP_alphas']].copy(),
                sample[self.indices['GP_betas']].copy(),
                # [6.5, 44.8],
                # [28.2, 0.0]
            ]
            # alpha_beta = np.insert(alpha_beta, np.arange(1, self.nseries + 1), 
            #                        sample[self.indices['GP_betas']])
            # print(alpha_beta)


            pars = np.r_[
                sample[self.indices['jitter']].copy(),
                gp_pars,
                alpha_beta,
            ]

            # for name, p in zip(C.param, pars):
            #     print(name, p)
            C.set_param(pars, C.param)
            # print(C.get_param())

            pred = np.zeros((self.nseries, t.size))
            std = np.zeros((self.nseries, t.size))

            for i in range(self.nseries):
                C.kernel['gp'].set_conditional_coef(series_id=i)
                if return_std:
                    pred[i], std[i] = C.conditional(resid_full, t, calc_cov='diag')
                else:
                    pred[i] = C.conditional(resid_full, t)

            if return_std:
                return pred, std#, C
            else:
                return pred#, C
            # return C.conditional(r, t)

        else:
            r = self.data.y - self.eval_model(sample)

            if include_jitters:
                # get jitters per instrument in an array
                jRV = sample[self.indices['jitter']]
                jRV = jRV[self.data.obs.astype(int) - 1]
                self.GP.white_noise = jRV
            else:
                self.GP.white_noise = 0.0

            # if self.model is MODELS.GPmodel_systematics:
            #     x = self._extra_data[:, 3]
            #     X = np.c_[t, interp1d(self.data.t, x, bounds_error=False)(t)]
            #     GPpars = sample[self.indices['GPpars']]
            #     mu = self.GP.predict(r, X, GPpars)
            #     # self.GP.kernel.pars = GPpars
            #     return mu
            # else:
            GPpars = sample[self.indices['GPpars']].copy()
            if self.kernel is GP.spleaf_esp:
                GPpars[-1] /= 2.0
            self.GP.kernel.pars = GPpars

            return self.GP.predict(r, t, return_std=return_std, return_cov=return_cov)

    def full_model(self, sample, t=None, **kwargs):
        """
        Evaluate the full model at one posterior sample, including the GP. If
        `t` is `None`, use the observed times. Instrument offsets are only added
        if `t` is `None`, but the systemic velocity is always added.
        
        To evaluate at all posterior samples, consider using
        
        ```python
        np.apply_along_axis(self.full_model, 1, self.posterior_sample)
        ```

        Arguments:
            sample (array): One posterior sample, with shape (npar,)
            t (ndarray, optional):
                Times at which to evaluate the model, or `None` to use observed
                times
            **kwargs: Keyword arguments passed directly to `eval_model`
        """
        deterministic = self.eval_model(sample, t, **kwargs)
        stochastic = self.stochastic_model(sample, t)
        return deterministic + stochastic

    def burst_model(self, sample, t=None, v=None):
        """
        For models with multiple instruments, this function "bursts" the
        computed RV into `n_instruments` individual arrays. This is mostly
        useful for plotting the RV model together with the original data.

        Args:
            sample (array): One posterior sample, with shape (npar,)
            t (array): Times at which to evaluate the model
            v (array): Pre-computed RVs. If `None`, calls `self.eval_model`
        """
        if v is None:
            v = self.eval_model(sample, t)
        if t is None:
            t = self.data.t.copy()

        if not self.multi:
            # print('not multi_instrument, burst_model adds nothing')
            return v

        ni = self.n_instruments
        offsets = sample[self.indices["inst_offsets"]]

        if self._time_overlaps[0]:
            v = np.tile(v, (self.n_instruments, 1))
            if self.model is MODELS.RVFWHMmodel:
                # offset is [off1_RV, off2_RV, ..., off1_FWHM, off2_FWHM, ...]
                # we want it as [off1_RV, off1_FWHM, off2_RV, off2_FWHM, ...]
                offsets = np.array(list(zip(
                    offsets[:self.n_instruments-1], 
                    offsets[self.n_instruments-1:]
                ))).flatten()
                # also add 0 offsets for the last instrument (RV and FWHM)
                offsets = np.r_[offsets, np.zeros(2)]

                v = (v.T + offsets).T
                # this constrains the RV to the times of each instrument
                for i in range(self.n_instruments):
                    obst = self.data.t[self.data.obs == i + 1]
                    # RV
                    v[2 * i, t < obst.min()] = np.nan
                    v[2 * i, t > obst.max()] = np.nan
                    # FWHM
                    v[2 * i + 1, t < obst.min()] = np.nan
                    v[2 * i + 1, t > obst.max()] = np.nan
            else:
                v = (v.T + np.r_[offsets, 0.0]).T
                # this constrains the RV to the times of each instrument
                for i in range(self.n_instruments):
                    obst = self.data.t[self.data.obs == i + 1]
                if i == 0:
                    v[i, t < obst.min()] = np.nan
                if i < self.n_instruments - 1:
                    v[i, t > obst.max()] = np.nan

        else:
            # the first time, plus the times that separate each instrument
            time_bins = np.sort(np.r_[t[0], self._offset_times])

            # which time "bin" each time belongs to
            ii = np.digitize(t, time_bins) - 1

            # reorder the indices so they are in the same order as the
            # instruments, as set by data.obs
            ii_copy = ii.copy()
            ind = np.unique(self.data.obs, return_index=True)[1]
            for a, b in zip(self.data.obs[ind] - 1, self.data.obs[np.sort(ind)] - 1):
                ii[ii_copy == a] = b

            if self.model in (MODELS.RVFWHMmodel, MODELS.RVFWHMRHKmodel):
                offsets = np.pad(offsets.reshape(-1, ni - 1), ((0, 0), (0, 1)))
                v += np.take(offsets, ii, axis=1)
            elif self.model is MODELS.SPLEAFmodel:
                zero_points = sample[self.indices['zero_points']]
                offsets = np.r_[offsets, 0.0, zero_points]
                offsets = np.pad(-np.diff(offsets)[::ni].reshape(-1, 1), ((0, 0), (0, 1)))
                v += np.take(offsets.reshape(-1, ni), ii, axis=1)
            else:
                offsets = np.pad(offsets, (0, 1))
                v += np.take(offsets, ii)

        return v

    def individual_logZ(self):
        """
        Calculates individual log-evidences for each Np value. When Np is fixed,
        simply return the full model log-evidence.
        """
        if self.fix:
            return self.evidence
        else:
            from .analysis import compute_values_from_ratios
            return np.log(compute_values_from_ratios(np.exp(self.evidence), self.ratios))

    def residuals(self, sample, full=False):
        if self.model is MODELS.RVFWHMmodel:
            D = np.vstack([self.data.y, self.data.y2])
        elif self.model is MODELS.RVFWHMRHKmodel:
            D = np.vstack([self.data.y, self.data.y2, self.data.y3])
        elif self.model is MODELS.BINARIESmodel:
            if self.double_lined:
                D = np.vstack([self.data.y, self.data.y2])
            else:
                D = self.data.y
        else:
            D = self.data.y

        if full:
            return D - self.full_model(sample)
        else:
            return D - self.eval_model(sample)

    def residual_std(self, sample, per_instrument=True, printit=True):
        sb2 = False
        r = self.residuals(sample, full=True)
        if self.model in (MODELS.RVFWHMmodel, MODELS.RVFWHMRHKmodel):
            r = r[0]
        if self.model is MODELS.BINARIESmodel:
            if self.double_lined:
                sb2 = True
                r2 = r[1]
                r = r[0]

        vals = []
        val = np.std(r)
        vals.append(val)

        vals2 = []
        if sb2:
            val2 = np.std(r2)
            vals2.append(val2)

        if printit:
            if sb2:
                print(f'full primary: {val:.3f} m/s')
                print(f'full secondary: {val2:.3f} m/s')
            else:
                print(f'full: {val:.3f} m/s')
        

        if per_instrument and self.multi:
            for inst, o in zip(self.instruments, np.unique(self.data.obs)):
                val = np.std(r[self.data.obs == o])
                vals.append(val)
                if sb2:
                    val2 = np.std(r2[self.data.obs == o])
                    vals2.append(val2)
                if printit:
                    if sb2:
                        print(f'{inst} primary: {val:.3f} m/s')
                        print(f'{inst} secondary: {val:.3f} m/s')
                    else:
                        print(f'{inst}: {val:.3f} m/s')
                
        ret = np.array(vals)
        if sb2:
            ret = (np.array(vals),np.array(vals2))

        return ret

    # residual_std = partialmethod(_residual_quantity, np.std)

    def residual_rms(self, sample, per_instrument=True, weighted=True, printit=True):
        sb2 = False
        r = self.residuals(sample, full=True)
        if self.model in (MODELS.RVFWHMmodel, MODELS.RVFWHMRHKmodel):
            r = r[0]
        if self.model is MODELS.BINARIESmodel:
            if self.double_lined:
                sb2 = True
                r2 = r[1]
                r = r[0]

        vals = []
        if weighted:
            val = wrms(r, weights=1 / self.data.e**2)
        else:
            val = rms(r)
        vals.append(val)

        vals2 = []
        if sb2:
            if weighted:
                val2 = wrms(r2, weights=1 / self.data.e2**2)
            else:
                val2 = rms(r2)
            vals2.append(val2)

        if printit:
            if sb2:
                print(f'full primary: {val:.3f} m/s')
                print(f'full secondary: {val2:.3f} m/s')
            else:
                print(f'full: {val:.3f} m/s')

        

        if per_instrument and self.multi:
            for inst, o in zip(self.instruments, np.unique(self.data.obs)):
                val = wrms(r[self.data.obs == o],
                           weights=1 / self.data.e[self.data.obs == o]**2)
                vals.append(val)
                if sb2:
                    val2 = wrms(r2[self.data.obs == o],
                           weights=1 / self.data.e2[self.data.obs == o]**2)
                    vals2.append(val2)
                if printit:
                    if sb2:
                        print(f'{inst} primary: {val:.3f} m/s')
                        print(f'{inst} secondary: {val:.3f} m/s')
                    else:
                        print(f'{inst}: {val:.3f} m/s')
                
        ret = np.array(vals)
        if sb2:
            ret = (np.array(vals),np.array(vals2))

        return ret

    def _planet_i_indices(self, i):
        start, stop, _ = self.indices['planets'].indices(self.posterior_sample.shape[1])
        return np.arange(start + i, stop, self.max_components)

    def from_prior(self, n=1):
        """ Generate `n` samples from the priors for all parameters. """
        prior_samples = []
        for _ in range(n):
            prior = []
            for par, pr in zip(self._parameters, self.parameter_priors):
                if par == 'ndim':
                    prior.append(self.n_dimensions)
                elif par == 'maxNp':
                    prior.append(self.max_components)
                elif par == 'staleness':
                    prior.append(0)
                else:
                    prior.append(distribution_rvs(pr)[0])

            prior = np.array(prior)
            # set to 0 the planet parameters > Np
            for j in range(int(prior[self.index_component]), self.max_components):
                prior[self._planet_i_indices(j)] = 0

            prior_samples.append(prior)
        return np.array(prior_samples).squeeze()

    def simulate_from_sample(self, sample, times, add_noise=True, errors=True,
                             append_to_file=False):
        y = self.full_model(sample, times)
        e = np.zeros_like(y)

        if add_noise:
            if self.model is MODELS.RVFWHMmodel:
                n1 = np.random.normal(0, self.e.mean(), times.size)
                n2 = np.random.normal(0, self.e2.mean(), times.size)
                y += np.c_[n1, n2].T
            elif self.model is MODELS.RVmodel:
                n = np.random.normal(0, self.e.mean(), times.size)
                y += n

        if errors:
            if self.model is MODELS.RVFWHMmodel:
                er1 = np.random.uniform(self.e.min(), self.e.max(), times.size)
                er2 = np.random.uniform(self.e2.min(), self.e2.max(),
                                        times.size)
                e += np.c_[er1, er2].T

            elif self.model is MODELS.RVmodel:
                er = np.random.uniform(self.e.min(), self.e.max(), times.size)
                e += er

        if append_to_file:
            last_file = self.data_file[-1]
            name, ext = os.path.splitext(last_file)
            n = times.size
            file = f'{name}_+{n}sim{ext}'
            print(file)

            with open(file, 'w') as out:
                out.writelines(open(last_file).readlines())
                if self.model is MODELS.RVFWHMmodel:
                    kw = dict(delimiter='\t', fmt=['%.5f'] + 4 * ['%.9f'])
                    np.savetxt(out, np.c_[times, y[0], e[0], y[1], e[1]], **kw)
                elif self.model is MODELS.RVmodel:
                    kw = dict(delimiter='\t', fmt=['%.5f'] + 2 * ['%.9f'])
                    np.savetxt(out, np.c_[times, y, e], **kw)

        if errors:
            return y, e
        else:
            return y

    @property
    def star(self):
        if self._star:
            return self._star
        try:
            if self.multi:
                self._star = get_star_name(self.data_file[0])
            else:
                self._star = get_star_name(self.data_file)
        except IndexError:
            pass
        return self._star

    @star.setter
    def star(self, star):
        self._star = str(star)

    @property
    def instruments(self):
        if not hasattr(self, '_instruments'):
            self.instruments = None
        return self._instruments
    
    @instruments.setter
    def instruments(self, instruments=None):
        if instruments is None:
            if not hasattr(self, 'data_file'):
                self._instruments = self.n_instruments * [''] if self.multi else ''

            if self.multi:
                self._instruments = list(map(get_instrument_name, self.data_file))
            else:
                self._instruments = get_instrument_name(self.data_file)
        else:
            self._instruments = instruments


    @property
    def Np(self):
        return self.posterior_sample[:, self.index_component]

    # @property
    # def ratios(self):
    #     bins = np.arange(self.max_components + 2)
    #     n, _ = np.histogram(self.Np, bins=bins)
    #     n = n.astype(np.float)
    #     n[n == 0] = np.nan
    #     r = n.flat[1:] / n.flat[:-1]
    #     r[np.isnan(r)] = np.inf
    #     return r
    @property
    def ratios(self):
        bins = np.arange(self.max_components + 2) - 0.5
        n, _ = np.histogram(self.Np, bins=bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = n[1:] / n[:-1]
            return r

    @property
    def _error_ratios(self):
        # self if a KimaResults instance
        from scipy.stats import multinomial
        bins = np.arange(self.max_components + 2)
        n, _ = np.histogram(self.Np, bins=bins)
        prob = n / self.ESS
        r = multinomial(self.ESS, prob).rvs(10000)
        r = r.astype(np.float)
        r[r == 0] = np.nan
        return (r[:, 1:] / r[:, :-1]).std(axis=0)

    @property
    def _time_overlaps(self):
        """
        This function checks for time overlaps between the observations from
        different instruments. It returns a two-tuple
            (True, indices of overlapping instruments) or
            (False, [])
        """
        from itertools import combinations
        from .utils import Interval

        intervals = [
            Interval.from_array(self.data.t[self.data.obs == i + 1])
            for i in range(self.n_instruments)
        ]

        overlaps = [
            (i + 1, j + 1)
            for i, j in combinations(range(self.n_instruments), 2)
            if self.data.t[self.data.obs == i + 1] in intervals[j]
        ]

        return len(overlaps) > 0, overlaps

        # # check for overlaps in the time from different instruments
        # if not self.multi:
        #     raise ValueError('Model is not multi_instrument')

        # def minmax(x):
        #     return x.min(), x.max()

        # # are the instrument identifiers all sorted?
        # # st = np.lexsort(np.vstack([self.t, self.data.obs]))
        # obs_is_sorted = np.all(np.diff(self.data.obs) >= 0)

        # # if not, which ones are not sorted?
        # if not obs_is_sorted:
        #     which_not_sorted = np.unique(self.data.obs[1:][np.diff(self.data.obs) < 0])

        # overlap = []
        # for i in range(1, self.n_instruments):
        #     t1min, t1max = minmax(self.data.t[self.data.obs == i])
        #     t2min, t2max = minmax(self.data.t[self.data.obs == i + 1])
        #     # if the instrument IDs are sorted or these two instruments
        #     # (i and i+1) are not the ones not-sorted
        #     print(i, i not in which_not_sorted, t1max, t2min)
        #     if obs_is_sorted or i not in which_not_sorted:
        #         if t2min < t1max:
        #             overlap.append((i, i + 1))
        #     # otherwise the check is different
        #     else:
        #         if t1min < t2max:
        #             overlap.append((i, i + 1))

        # return len(overlap) > 0, overlap

    @property
    def _offset_times(self):
        if not self.multi:
            raise ValueError('Model is not multi_instrument, no offset times')

        # check for overlaps
        has_overlaps, overlap = self._time_overlaps
        if has_overlaps:
            _o = []
            m = np.full_like(self.data.obs, True, dtype=bool)
            for ov in overlap:
                _o.append(self.data.t[self.data.obs == ov[0]].max())
                _o.append(self.data.t[self.data.obs == ov[1]].min())
                m &= self.data.obs != ov[0]

            _1 = self.data.t[m][np.ediff1d(self.data.obs[m], 0, None) != 0]
            _2 = self.data.t[m][np.ediff1d(self.data.obs[m], None, 0) != 0]
            return np.sort(np.r_[_o, np.mean((_1, _2), axis=0)])

        # otherwise it's much easier
        else:
            _1 = self.data.t[np.ediff1d(self.data.obs, 0, None) != 0]
            _2 = self.data.t[np.ediff1d(self.data.obs, None, 0) != 0]
            return np.mean((_1, _2), axis=0)

    def data_properties(self):
        t = self.data.t
        prop = {
            'time span': (np.ptp(t), 'days', True),
            'mean time gap': (np.ediff1d(t).mean(), 'days', True),
            'median time gap': (np.median(np.ediff1d(t)), 'days', True),
            'shortest time gap': (np.ediff1d(t).min(), 'days', True),
            'longest time gap': (np.ediff1d(t).max(), 'days', True),
        }
        width = max(list(map(len, prop.keys()))) + 2
        for k, v in prop.items():
            print(f'{k:{width}s}: {v[0]:10.6f}  {v[1]}', end=' ')
            if v[2]:
                print(f'({1/v[0]:10.6f} {v[1]}⁻¹)')

    # @property
    # def eta1(self):
    #     if self.has_gp:
    #         return self.posterior_sample[:, self.indices['GPpars']][:, 0]
    #     return None

    # @property
    # def eta2(self):
    #     if self.has_gp:
    #         i = 2 if self.model is MODELS.RVFWHMmodel else 1
    #         return self.posterior_sample[:, self.indices['GPpars']][:, i]
    #     return None

    # @property
    # def eta3(self):
    #     if self.has_gp:
    #         i = 3 if self.model is MODELS.RVFWHMmodel else 2
    #         return self.posterior_sample[:, self.indices['GPpars']][:, i]
    #     return None

    # @property
    # def eta4(self):
    #     if self.has_gp:
    #         i = 4 if self.model is MODELS.RVFWHMmodel else 3
    #         return self.posterior_sample[:, self.indices['GPpars']][:, i]
    #     return None

    # most of the following methods just dispatch to display
    make_plots = display.make_plots

    #
    phase_plot = display.phase_plot

    #
    plot1 = display.plot_posterior_np
    plot_posterior_np = display.plot_posterior_np

    #
    plot2 = display.plot_posterior_period
    plot_posterior_periods = display.plot_posterior_period

    #
    plot3 = display.plot_PKE
    plot_posterior_PKE = display.plot_PKE

    #
    plot4 = display.plot_gp
    plot_gp = display.plot_gp
    plot_posterior_hyperpars = display.plot_gp

    #
    plot5 = display.plot_gp_corner
    plot_gp_corner = display.plot_gp_corner
    
    #
    corner_planet_parameters = display.corner_planet_parameters
    corner_known_object = display.corner_known_object


    def get_sorted_planet_samples(self, full=True):
        # all posterior samples for the planet parameters
        # this array is nsamples x (n_dimensions*max_components)
        # that is, nsamples x 5, nsamples x 10, for 1 and 2 planets for example
        if full:
            samples = self.posterior_sample.copy()
        else:
            samples = self.posterior_sample[:, self.indices['planets']].copy()

        if self.max_components == 0:
            return samples

        # here we sort the samples array by the orbital period
        # this is a bit difficult because the organization of the array is
        # P1 P2 K1 K2 ....
        sorted_samples = samples.copy()
        n = self.max_components * self.n_dimensions
        mc = self.max_components
        p = samples[:, self.indices['planets.P']]
        ind_sort_P = np.arange(p.shape[0])[:, np.newaxis], np.argsort(p)

        for par in ('P', 'K', 'φ', 'e', 'w'):
            sorted_samples[:, self.indices[f'planets.{par}']] = \
                samples[:, self.indices[f'planets.{par}']][ind_sort_P]

        return sorted_samples

    def _apply_cuts_period(self, pmin=None, pmax=None, return_mask=False):
        """ apply cuts in orbital period """
        if pmin is None and pmax is None:
            if return_mask:
                return np.ones(self.ESS, dtype=bool)
            else:
                return self.posterior_sample

        periods = self.posterior_sample[:, self.indices['planets.P']]

        if pmin is None:
            mask_min = np.ones(self.ESS, dtype=bool)
        else:
            mask_min = np.logical_and.reduce((periods > pmin).T)
        if pmax is None:
            mask_max = np.ones(self.ESS, dtype=bool)
        else:
            mask_max = np.logical_and.reduce((periods < pmax).T)

        if return_mask:
            return mask_min & mask_max
        else:
            return self.posterior_sample[mask_min & mask_max]

        np.logical_and(*(res.posterior_sample[:, res.indices['planets.P']] > 10).T)
        too_low_periods = np.zeros_like(samples[:, 0], dtype=bool)
        too_high_periods = np.zeros_like(samples[:, 0], dtype=bool)

        if pmin is not None:
            too_low_periods = samples[:, 0] < pmin
            samples = samples[~too_low_periods, :]

        if pmax is not None:
            too_high_periods = samples[:, 1] > pmax
            samples = samples[~too_high_periods, :]

        if return_mask:
            mask = ~too_low_periods & ~too_high_periods
            return samples, mask
        else:
            return samples

    def print_results(self, show_prior=False):
        """
        Print a summary of the results, showing the posterior estimates for each parameter.

        Args:
            show_prior (bool, optional):
                Whether to show the prior distribution.
        
        Note:
            This function is heavily inspired by a similar implementation in
            [UltraNest](https://johannesbuchner.github.io/UltraNest/index.html).
        """
        # part of this code from UltraNest:
        # https://github.com/JohannesBuchner/UltraNest/
        #
        def print_header():
            print("    ", end="")
            print("%-15s" % "parameter", end=": ")
            print("%-20s" % "median ± std", end=" ")
            print("Ͱ", end=" ")
            if show_prior:
                print('%-24s' % 'prior', end=' ')
            print(' distribution '.center(60), end=' ')
            print()

        def print_line(p, v, prior=None, show_prior=False):
            if isinstance(prior, distributions.Fixed):
                print(f'    {p:<15s}: {v[0]:<20.3f} Ͱ', end=' ')
                if show_prior:
                    print(f"{str(prior):<24s}")
                else:
                    print("%10s" % "Fixed")
                return

            sigma, med = np.std(v), np.median(v)
            j = 3 if sigma == 0 else max(0, int(-np.floor(np.log10(sigma))) + 1)
            fmt = '%%.%df' % j

            med_sigma = '%s ± %s' % (fmt % med, fmt % sigma)

            H, edges = np.histogram(v, bins=40)
            lo, hi = edges[0], edges[-1]
            step = edges[1] - lo
            if prior is not None and show_prior:
                lower = prior.ppf(1e-6)
                if np.isfinite(lower):
                    lo = max(lower, lo - 2 * step)
                upper = prior.ppf(1.0 - 1e-6)
                if np.isfinite(upper):
                    hi = min(upper, hi + 2 * step)
            H, edges = np.histogram(v, bins=np.linspace(lo, hi, 40))
            lo, hi = edges[0], edges[-1]
            dist = ''.join([' ▁▂▃▄▅▆▇██'[i] for i in np.ceil(H * 7 / H.max()).astype(int)])

            range_dist = '%10s│%s│%-10s' % (fmt % lo, dist, fmt % hi)

            if show_prior:
                prior_short = str(prior)
                prior_short = prior_short.replace('ModifiedLogUniform', 'MLU')
                prior_short = prior_short.replace('LogUniform', 'LU')
                prior_short = prior_short.replace('Uniform', 'U')
                prior_short = prior_short.replace('Kumaraswamy', 'Kuma')
                prior_short = prior_short.replace('Gaussian', 'G')
                print('    %-15s: %-20s Ͱ %-24s %60s' % (p, med_sigma, prior_short, range_dist))
            else:
                print('    %-15s: %-20s Ͱ %60s' % (p, med_sigma, range_dist))

        ########

        print(f'logL max: {self.sample_info[:,1].max():.2f}')
        print(f'logZ: {self.evidence:.2f}', end='\n\n')
        print_header()

        # if self.posteriors.jitter.ndim == 1:
        #     J = self.posteriors.jitter.reshape(-1, 1)
        #     number = False
        # else:
        #     J = self.posteriors.jitter
        #     number = True

        jitter = self.posteriors.jitter.copy()
        if jitter.ndim == 1:
            jitter = jitter.reshape(-1, 1)

        start_jitter = 0
        if self.model is MODELS.RVmodel and self.multi:
            v = jitter[:, 0]
            print_line('stellar_jitter', v, self.priors['stellar_jitter_prior'], show_prior)
            start_jitter = 1

        series_k = 0
        has_instruments = len(self.instruments) == self.n_jitters - start_jitter
        for i in range(start_jitter, self.n_jitters):
            v = jitter[:, i]

            if has_instruments:
                jitter_name = f'jit {self.instruments[i - start_jitter]}'
            else:
                jitter_name = self.parameters[i]

            if self.model is MODELS.SPLEAFmodel and 'series_j' in self.parameters[i]:
                prior = self.priors[f'series_jitters_prior_{series_k+1}']
                series_k += 1
            elif self.model is MODELS.RVFWHMmodel:
                if i < self.n_instruments:
                    prior = self.priors['Jprior']
                else:
                    prior = self.priors['Jfwhm_prior']    
            else:
                prior = self.priors['Jprior']
            print_line(jitter_name, v, prior, show_prior)

        # k = 0
        # for i, v in enumerate(J.T):
        #         print_line('stellar_jitter', v, self.priors['stellar_jitter_prior'], show_prior)
        #     else:
        #         print_line(f'jitter{k+1}' if number else 'jitter', v, self.priors['Jprior'], show_prior)
        #         k += 1

        if self.studentt:
            print_line("nu", self.posteriors.nu, self.priors["nu_prior"], show_prior)

        if self.model is MODELS.BINARIESmodel and self.double_lined:
            print_line('vsys_sec', self.posteriors.vsys_sec, self.priors['Cprior'], show_prior)
        print_line('vsys', self.posteriors.vsys, self.priors['Cprior'], show_prior)
        

        if self.model is MODELS.RVFWHMmodel:
            print_line('cfwhm', self.posteriors.cfwhm, self.priors['Cfwhm_prior'], show_prior)

        if self.trend:
            if self.trend_degree >= 1:
                print_line('slope', self.posteriors.slope, self.priors['slope_prior'], show_prior)
            if self.trend_degree >= 2:
                print_line('quadr', self.posteriors.quadr, self.priors['quadr_prior'], show_prior)
            if self.trend_degree >= 3:
                print_line('cubic', self.posteriors.cubic, self.priors['cubic_prior'], show_prior)

        if self.model is MODELS.RVFWHMmodel and self.trend_fwhm:
            if self.trend_fwhm_degree >= 1:
                print_line('slope_fwhm', self.posteriors.slope_fwhm, self.priors['slope_fwhm_prior'], show_prior)
            if self.trend_fwhm_degree >= 2:
                print_line('quadr_fwhm', self.posteriors.quadr_fwhm, self.priors['quadr_fwhm_prior'], show_prior)
            if self.trend_fwhm_degree >= 3:
                print_line('cubic_fwhm', self.posteriors.cubic_fwhm, self.priors['cubic_fwhm_prior'], show_prior)

        if self.model is MODELS.SPLEAFmodel:
            for i in range(self.n_instruments * (self.nseries - 1)):
                v = self.posterior_sample[:, self.indices['zero_points']][:, i]
                print_line(f'zero-point {i+1}: ', v, self.priors[f'zero_points_prior_{i+1}'], show_prior)

        if self.multi:
            if self.model is MODELS.RVFWHMmodel:
                rv_offsets = self.posteriors.offset[:, :self.n_instruments - 1]
                for i, v in enumerate(rv_offsets.T):
                    print_line(f'offset{i+1}', v, self.priors[f'individual_offset_prior[{i}]'], show_prior)
                fw_offsets = self.posteriors.offset[:, self.n_instruments - 1:]
                for i, v in enumerate(fw_offsets.T):
                    print_line(f'offset{i+1}_fwhm', v, self.priors[f'individual_offset_fwhm_prior[{i}]'], show_prior)
            else:
                for i, v in enumerate(self.posteriors.offset.T):
                    print_line(f'offset{i+1}', v, self.priors[f'individual_offset_prior[{i}]'], show_prior)

        if self.max_components > 0:
            print("    - planets")
            if not self.fix:
                print_line('Np', self.Np, self.priors["np_prior"], show_prior)

            for i in range(self.max_components):
                Np_mask = self.Np > i
                if not Np_mask.any():
                    continue
                print_line(f'{i+1}: P', self.posteriors.P[Np_mask, i], self.priors['Pprior'], show_prior)
                print_line(f'{i+1}: K', self.posteriors.K[Np_mask, i], self.priors['Kprior'], show_prior)
                print_line(f'{i+1}: M0', self.posteriors.φ[Np_mask, i], self.priors['phiprior'], show_prior)
                print_line(f'{i+1}: e', self.posteriors.e[Np_mask, i], self.priors['eprior'], show_prior)
                print_line(f'{i+1}: w', self.posteriors.w[Np_mask, i], self.priors['wprior'], show_prior)
                print('    --')
        
        if self.has_gp:
            print('    - GP')
            gp_parameter_names = np.array(self._parameters)[self.indices['GPpars']]
            for i in range(self.n_hyperparameters):
                name = gp_parameter_names[i]
                print_line(name, getattr(self.posteriors, f'η{i+1}'), self.priors[f'{name}_prior'], show_prior)

            if self.model is MODELS.SPLEAFmodel:
                series_names = np.r_[['RV'], self._extra_data_names[::2]]
                for i in range(self.nseries):
                    print_line(f'α{i+1} ({series_names[i]})', self.alphas[:, i], self.priors[f'alpha{i+1}_prior'], show_prior)
                    print_line(f'β{i+1} ({series_names[i]})', self.betas[:, i], self.priors[f'beta{i+1}_prior'], show_prior)

        if self.KO:
            print('    - KO')
            for i in range(self.nKO):
                print_line(f'{i+1}: P', self.posteriors.KO.P[:, i], self.priors[f'KO_Pprior_{i}'], show_prior)
                print_line(f'{i+1}: K', self.posteriors.KO.K[:, i], self.priors[f'KO_Kprior_{i}'], show_prior)
                print_line(f'{i+1}: M0', self.posteriors.KO.φ[:, i], self.priors[f'KO_phiprior_{i}'], show_prior)
                print_line(f'{i+1}: e', self.posteriors.KO.e[:, i], self.priors[f'KO_eprior_{i}'], show_prior)
                print_line(f'{i+1}: w', self.posteriors.KO.w[:, i], self.priors[f'KO_wprior_{i}'], show_prior)

        if self.TR:
            print('    - TR')
            for i in range(self.nTR):
                print_line(f'{i+1}: P', self.posteriors.TR.P[:, i], self.priors[f'TR_Pprior_{i}'], show_prior)
                print_line(f'{i+1}: K', self.posteriors.TR.K[:, i], self.priors[f'TR_Kprior_{i}'], show_prior)
                print_line(f'{i+1}: Tc', self.posteriors.TR.Tc[:, i], self.priors[f'TR_Tcprior_{i}'], show_prior)
                print_line(f'{i+1}: e', self.posteriors.TR.e[:, i], self.priors[f'TR_eprior_{i}'], show_prior)
                print_line(f'{i+1}: w', self.posteriors.TR.w[:, i], self.priors[f'TR_wprior_{i}'], show_prior)


    def _set_plots(self):
        from functools import partial
        if self.model in (MODELS.RVFWHMmodel, MODELS.RVFWHMRHKmodel):
            self.plot_random_samples = self.plot6 = partial(display.plot_random_samples_multiseries, res=self)
        elif self.model in (MODELS.SPLEAFmodel,):
            self.plot_random_samples = self.plot6 = partial(display.plot_random_samples_spleaf, res=self)
        else:
            self.plot_random_samples = self.plot6 = partial(display.plot_random_samples, res=self)
        self.plot_random_samples.__doc__ = display.plot_random_samples.__doc__
        self.plot6.__doc__ = display.plot_random_samples.__doc__

        if self.model is MODELS.RVHGPMmodel:
            self.plot_hgpm = partial(display.plot_hgpm,
                                     res=self, pm_data=self.pm_data)
            self.hist_bary = partial(display.hist_bary, res=self)

    #
    hist_vsys = display.hist_vsys
    hist_jitter = display.hist_jitter
    hist_correlations = display.hist_correlations
    hist_trend = display.hist_trend
    hist_MA = display.hist_MA
    hist_nu = display.hist_nu
