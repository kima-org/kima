"""
This module defines the `KimaResults` class to hold results from a run.
"""

from copy import deepcopy
import os
import sys
import pickle
from typing import List, Union
import zipfile
import time
import tempfile
from string import ascii_lowercase
from dataclasses import dataclass, field
from io import StringIO
from contextlib import redirect_stdout

from .. import __models__
from kima.kepler import keplerian as kepleriancpp
from kima import distributions
from .classic import postprocess
from .GP import (GP, RBFkernel, QPkernel, QPCkernel, PERkernel, QPpCkernel,
                 QPpMAGCYCLEkernel, mixtureGP)

from .analysis import get_planet_mass_and_semimajor_axis
from .utils import (read_datafile, read_datafile_rvfwhm, read_datafile_rvfwhmrhk, read_model_setup,
                    get_star_name, mjup2mearth, get_prior, get_instrument_name,
                    _show_kima_setup, read_big_file, rms, wrms, chdir)

from .import display

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import (norm, gaussian_kde, randint as discrete_uniform)
try:  # only available in scipy 1.1.0
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None

pathjoin = os.path.join
colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]


def keplerian(*args, **kwargs):
    return np.array(kepleriancpp(*args, **kwargs))



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

        priors += list(setup['priors.known_object'].values())
        prior_names += ['KO_' + k for k in setup['priors.known_object'].keys()]
    except KeyError:
        pass

    prior_dists = []
    for p in priors:
        p = p.replace(';', ',')
        if 'UniformAngle' in p:
            p = 'UniformAngle()'
        prior_dists.append(eval('distributions.' + p))

    priors = {n: v for n, v in zip(prior_names, prior_dists)}
    # priors = {
    #     n: v
    #     for n, v in zip(prior_names, [get_prior(p) for p in priors])
    # }

    return priors


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
    instrument: str = 'GAIA'

    def __repr__(self):
        return f'data_holder(N={self.N}, t, w, sigw, psi, pf)'


@dataclass
class posterior_holder:
    """ A simple class to hold the posterior samples

    Attributes:
        P (ndarray): Orbital period(s)
        K (ndarray): Semi-amplitude(s)
        e (ndarray): Orbital eccentricities(s)
        ω (ndarray): Argument(s) of pericenter
        φ (ndarray): Mean anomaly(ies) at the epoch
        jitter (ndarray): Per-instrument jitter(s)
        stellar_jitter (ndarray): Global jitter
        offset (ndarray): Between-instrument offset(s)
        vsys (ndarray): Systemic velocity

    """
    Np: np.ndarray = field(init=False)
    P: np.ndarray = field(init=False)
    K: np.ndarray = field(init=False)
    e: np.ndarray = field(init=False)
    ω: np.ndarray = field(init=False)
    φ: np.ndarray = field(init=False)
    # 
    jitter: np.ndarray = field(init=False)
    stellar_jitter: np.ndarray = field(init=False)
    offset: np.ndarray = field(init=False)
    vsys: np.ndarray = field(init=False)
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

    def __repr__(self):
        fields = list(self.__dataclass_fields__.keys())
        fields = [f for f in fields if hasattr(self, f) and getattr(self, f).size > 0]
        fields = ', '.join(fields)
        return f'posterior_holder({fields})'


def load_results(model_or_file, data=None, diagnostic=False, verbose=True,
                 moreSamples=1):
    # load from a pickle or zip file
    if isinstance(model_or_file, str):
        if not os.path.exists(model_or_file):
            raise FileNotFoundError(model_or_file)
        res = KimaResults.load(model_or_file)

    elif isinstance(model_or_file, __models__):
        res = KimaResults(model_or_file, data, diagnostic=diagnostic, verbose=verbose)

    return res

    # # load from current directory (latest results)
    # elif model_or_file is None:
    #     # cannot do it if there is no data information
    #     setup = read_model_setup()
    #     if setup['data']['file'] == '' and setup['data']['files'] == '':
    #         msg = 'no data information saved in kima_model_setup.txt '
    #         msg += '(RVData was probably created with list/array arguments) \n'
    #         msg += 'provide the `model` to load_results()'
    #         raise ValueError(msg)

        
    #     res = KimaResults()
    
    # # load from a model object
    # elif isinstance(model_or_file, __models__):
    #     return KimaResults.from_model(model_or_file, diagnostic, verbose)
    # else:
    #     raise NotImplementedError
    # return res


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

    _debug = False

    # def __init__(self, save_plots=False, return_figs=True, verbose=False, _dummy=False):
    #     self.save_plots = save_plots
    #     self.return_figs = return_figs
    #     self.verbose = verbose


    #     self.setup = setup = read_model_setup()

    #     if _dummy:
    #         return

    #     try:
    #         self.model = setup['kima']['model']
    #     except KeyError:
    #         self.model = 'RVmodel'

    #     if self._debug:
    #         print('model:', self.model)

    #     try:
    #         self.fix = setup['kima']['fix'] == 'true'
    #         self.npmax = int(setup['kima']['npmax'])
    #     except KeyError:
    #         self.fix = True
    #         self.npmax = 0

    #     # read the priors
    #     self.priors = _read_priors(self, setup)
    #     if self._debug:
    #         print('finished reading priors')

    #     # and the data
    #     self._read_data()
    #     if self._debug:
    #         print('finished reading data')

    #     # read the posterior samples
    #     self.posterior_sample = np.atleast_2d(read_big_file('posterior_sample.txt'))

    #     # try reading posterior sample info to get log-likelihoods
    #     try:
    #         self.posterior_lnlike = np.atleast_2d(read_big_file('posterior_sample_info.txt'))
    #         self._lnlike_available = True
    #     except IOError:
    #         self._lnlike_available = False
    #         print('Could not find file "posterior_sample_info.txt", '
    #               'log-likelihoods will not be available.')

    #     # read original samples
    #     try:
    #         t1 = time.time()
    #         self.sample = np.atleast_2d(read_big_file('sample.txt'))
    #         t2 = time.time()
    #         self.sample_info = np.atleast_2d(read_big_file('sample_info.txt'))
    #         with open('sample.txt', 'r') as fs:
    #             header = fs.readline()
    #             header = header.replace('#', '').replace('  ', ' ').strip()
    #             self.parameters = [p for p in header.split(' ') if p != '']
    #             self.parameters.pop(self.parameters.index('ndim'))
    #             self.parameters.pop(self.parameters.index('maxNp'))
    #             self.parameters.pop(self.parameters.index('staleness'))

    #         # different sizes can happen when running the model and sample_info
    #         # was updated while reading sample.txt
    #         if self.sample.shape[0] != self.sample_info.shape[0]:
    #             minimum = min(self.sample.shape[0], self.sample_info.shape[0])
    #             self.sample = self.sample[:minimum]
    #             self.sample_info = self.sample_info[:minimum]

    #     except IOError:
    #         self.sample = None
    #         self.sample_info = None
    #         self.parameters = []

    #     if self._debug:
    #         print('finished reading sample file', end=' ')
    #         print(f'(took {t2 - t1:.1f} seconds)')

    #     self.indices = {}
    #     self.total_parameters = 0

    #     self._current_column = 0
        
    #     # read jitters
    #     self._read_jitters()

    #     # read astrometric solution
    #     if self.model == 'GAIAmodel':
    #         self._read_astrometric_solution()

    #     # read limb-darkening coefficients
    #     if self.model == 'TRANSITmodel':
    #         self._read_limb_dark()
        
    #     # find trend in the compiled model and read it
    #     try:
    #         self.trend = self.setup['kima']['trend'] == 'true'
    #         self.trend_degree = int(self.setup['kima']['degree'])
    #         self._read_trend()
    #     except KeyError:
    #         self.trend, self.trend_degree = False, 0
    #     # multiple instruments? read offsets
    #     self._read_multiple_instruments()
    #     # activity indicator correlations?
    #     self._read_actind_correlations()
    #     # find GP in the compiled model
    #     self._read_GP()
    #     # find MA in the compiled model
    #     self._read_MA()

    #     # find KO in the compiled model
    #     try:
    #         self.KO = self.setup['kima']['known_object'] == 'true'
    #         self.nKO = int(self.setup['kima']['n_known_object'])
    #     except KeyError:
    #         self.KO = False
    #         self.nKO = 0
    #     self._read_KO()

    #     # find transiting planet in the compiled model
    #     try:
    #         self.TR = self.setup['kima']['transiting_planet'] == 'true'
    #         self.nTR = int(self.setup['kima']['n_transiting_planet'])
    #     except KeyError:
    #         self.TR = False
    #         self.nTR = 0
    #     self._read_TR()


    #     if self.model == 'OutlierRVmodel':
    #         self._read_outlier()

    #     self._read_components()

    #     # staleness (ignored)
    #     self._current_column += 1

    #     # student-t likelihood?
    #     try:
    #         self.studentt = self.setup['kima']['studentt'] == 'true'
    #     except KeyError:
    #         self.studentt = False
    #     self._read_studentt()

    #     if self.model == 'RVFWHMRHKmodel':
    #         self.C3 = self.posterior_sample[:, self._current_column]
    #         self.indices['C3'] = self._current_column
    #         self._current_column += 1

    #     if self.model in ('RVFWHMmodel', 'RVFWHMRHKmodel'):
    #         self.C2 = self.posterior_sample[:, self._current_column]
    #         self.indices['C2'] = self._current_column
    #         self._current_column += 1

    #     if self.model != 'GAIAmodel':
    #         self.vsys = self.posterior_sample[:, -1]
    #         self.indices['vsys'] = -1

    #     # build the marginal posteriors for planet parameters
    #     self.get_marginals()

    #     if self.fix:
    #         self.parameters.pop(self.parameters.index('Np'))

    #     self._set_plots()
    #     # # make the plots, if requested
    #     # self.make_plots(options, self.save_plots)


    # @classmethod
    # def from_model(cls, model, diagnostic=False, verbose=True):
    def __init__(self, model, data=None, diagnostic=False, 
                 save_plots=False, return_figs=True, verbose=False):
        self.save_plots = save_plots
        self.return_figs = return_figs
        self.verbose = verbose

        self.setup = setup = read_model_setup()

        hidden = StringIO()
        stdout = sys.stdout if verbose else hidden

        with redirect_stdout(stdout):
            try:
                evidence, H, logx_samples = postprocess(plot=diagnostic, numResampleLogX=1, moreSamples=1)
            except FileNotFoundError as e:
                if e.filename == 'levels.txt':
                    msg = f'No levels.txt file found in {os.getcwd()}. Did you run the model?'
                    raise FileNotFoundError(msg)
                raise e

        self.model = model.__class__.__name__
        self.fix = model.fix
        self.npmax = model.npmax
        self.evidence = evidence
        self.information = H

        self.posterior_sample = np.atleast_2d(read_big_file('posterior_sample.txt'))
        self._ESS = self.posterior_sample.shape[0]

        #self.priors = {}
        self.priors = _read_priors(self)

        # arbitrary units?
        if 'arb' in model.data.units:
            self.arbitrary_units = True
        else:
            self.arbitrary_units = False

        if data is None:
            data = model.data

        self.data = data_holder()
        self.data.t = np.array(np.copy(data.t))
        self.data.y = np.array(np.copy(data.y))
        self.data.e = np.array(np.copy(data.sig))
        self.data.obs = np.array(np.copy(data.obsi))
        self.data.N = data.N

        if self.model == 'RVFWHMmodel':
            self.data.y2, self.data.e2, *_ = np.array(data.actind)

        self._extra_data = np.array(np.copy(data.actind))

        self.M0_epoch = data.M0_epoch
        self.n_instruments = np.unique(data.obsi).size
        self.multi = data.multi

        if self.multi:
            self.data_file = data.datafiles
        else:
            self.data_file = data.datafile

        self.data.instrument = data.instrument
        if self.multi and len(data.instruments) > 0:
            self.instruments = data.instruments

        try:
            self.posterior_lnlike = np.atleast_2d(read_big_file('posterior_sample_info.txt'))
            self._lnlike_available = True
        except IOError:
            self._lnlike_available = False
            print('Could not find file "posterior_sample_info.txt", '
                  'log-likelihoods will not be available.')

        try:
            t1 = time.time()
            self.sample = np.atleast_2d(read_big_file('sample.txt'))
            t2 = time.time()
            self.sample_info = np.atleast_2d(read_big_file('sample_info.txt'))
            with open('sample.txt', 'r') as fs:
                header = fs.readline()
                header = header.replace('#', '').replace('  ', ' ').strip()
                self.parameters = [p for p in header.split(' ') if p != '']
                self.parameters.pop(self.parameters.index('ndim'))
                self.parameters.pop(self.parameters.index('maxNp'))
                self.parameters.pop(self.parameters.index('staleness'))

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
        self.total_parameters = 0

        self._current_column = 0

        # read jitters
        if self.multi and self.model in ('RVmodel'):
            self.n_jitters = 1  # stellar jitter
        else:
            self.n_jitters = 0

        self.n_jitters += self.n_instruments

        if self.model == 'RVFWHMmodel':
            self.n_jitters *= 2
        if self.model == 'RVFWHMRHKmodel':
            self.n_jitters *= 3
        self._read_jitters()

        # read limb-darkening coefficients
        if self.model == 'TRANSITmodel':
            self._read_limb_dark()

        # read trend
        self.trend = model.trend
        self.trend_degree = model.degree
        self._read_trend()

        # multiple instruments? read offsets
        self._read_multiple_instruments()

        # activity indicator correlations?
        self._read_actind_correlations()

        # find GP in the compiled model
        self._read_GP()

        # # find MA in the compiled model
        # self._read_MA()

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

        if self.model == 'OutlierRVmodel':
            self._read_outlier()

        self._read_components()

        # staleness (ignored)
        self._current_column += 1

        try:
            self.studentt = model.studentt
            self._read_studentt()
        except AttributeError:
            self.studentt = False

        if self.model == 'RVFWHMmodel':
            self.C2 = self.posterior_sample[:, self._current_column]
            self.indices['C2'] = self._current_column
            self._current_column += 1

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

    # def _read_data(self):
    #     setup = self.setup
    #     section = 'data' if 'data' in setup else 'kima'

    #     try:
    #         self.multi = setup[section]['multi'] == 'true'
    #     except KeyError:
    #         self.multi = False

    #     if self.model == 'HierarchicalRVmodel':
    #         self.multi = True

    #     if self.multi:
    #         if setup[section]['files'] == '':
    #             # multi is true but in only one file
    #             data_file = setup[section]['file']
    #             self.multi_onefile = True
    #         else:
    #             data_file = setup[section]['files'].split(',')[:-1]
    #             self.multi_onefile = False
    #             # raise NotImplementedError('TO DO')
    #     else:
    #         data_file = setup[section]['file']

    #     if self.verbose:
    #         print('Loading data file %s' % data_file)

    #     if data_file == '':
    #         raise ValueError('no data information in kima_model_setup.txt')

    #     self.data_file = data_file

    #     self.data_skip = int(setup[section]['skip'])
    #     self.units = setup[section]['units']
    #     self.M0_epoch = float(setup[section]['M0_epoch'])

    #     if self.multi:
    #         if self.model == 'RVFWHMmodel':
    #             data, obs = read_datafile_rvfwhm(self.data_file, self.data_skip)
    #         if self.model == 'RVFWHMRHKmodel':
    #             data, obs = read_datafile_rvfwhmrhk(self.data_file, self.data_skip)
    #         else:
    #             data, obs = read_datafile(self.data_file, self.data_skip)
            
    #         if self.multi_onefile:
    #             self.instruments = list(np.unique(obs).astype(str))

    #         if obs.min() == 0:
    #             obs += 1

    #         # make sure the times are sorted when coming from multiple
    #         # instruments
    #         ind = data[:, 0].argsort()
    #         data = data[ind]
    #         obs = obs[ind]
    #         self.n_instruments = np.unique(obs).size
    #         if self.model == 'RVFWHMmodel':
    #             self.n_jitters = 2 * self.n_instruments
    #         elif self.model == 'RVFWHMRHKmodel':
    #             self.n_jitters = 3 * self.n_instruments
    #         elif self.model == 'HierarchicalRVmodel':
    #             self.n_instruments = 1
    #             self.n_jitters = 1
    #         else:
    #             self.n_jitters = self.n_instruments

    #         if self.model == 'RVmodel':
    #             # for stellar jitter
    #             self.n_jitters += 1

    #     else:
    #         if self.model == 'RVFWHMmodel':
    #             cols = range(5)
    #             self.n_jitters = 2
    #         elif self.model == 'RVFWHMRHKmodel':
    #             cols = range(7)
    #             self.n_jitters = 3
    #         else:
    #             cols = range(3)
    #             self.n_jitters = 1
    #         self.n_instruments = 1

    #         data = np.loadtxt(self.data_file, skiprows=self.data_skip, usecols=cols)
    #         obs = np.ones_like(data[:, 0], dtype=int)
    #         self._extra_data = np.loadtxt(self.data_file, skiprows=self.data_skip)

    #     # to m/s
    #     if self.units == 'kms':
    #         data[:, 1] *= 1e3
    #         data[:, 2] *= 1e3
    #         if self.model == 'RVFWHMmodel':
    #             data[:, 3] *= 1e3
    #             data[:, 4] *= 1e3

    #     # arbitrary units?
    #     if 'arb' in self.units:
    #         self.arbitrary_units = True
    #     else:
    #         self.arbitrary_units = False

    #     self.data = data_holder()
    #     self.data.t = data[:, 0].copy()
    #     self.data.y = data[:, 1].copy()
    #     self.data.e = data[:, 2].copy()
    #     self.data.obs = obs.copy()
    #     self.data.N = self.data.t.size

    #     if self.model in ('RVFWHMmodel', 'RVFWHMRHKmodel'):
    #         self.data.y2 = data[:, 3].copy()
    #         self.data.e2 = data[:, 4].copy()
    #     if self.model in ('RVFWHMRHKmodel'):
    #         self.data.y3 = data[:, 5].copy()
    #         self.data.e3 = data[:, 6].copy()

    #     if self.model == 'GAIAmodel':
    #         self.astrometric_data = astrometric_data_holder()
    #         data = np.genfromtxt(self.data_file, names=True, comments='--')
    #         self.astrometric_data.t = data['mjd']
    #         self.astrometric_data.w = data['w']
    #         self.astrometric_data.sigw = data['sigw']
    #         self.astrometric_data.psi = data['psi']
    #         self.astrometric_data.pf = data['pf']
    #         self.astrometric_data.N = data['mjd'].size
            

    #     self.tmiddle = self.data.t.min() + 0.5 * self.data.t.ptp()

    def _read_jitters(self):
        i1, i2 = self._current_column, self._current_column + self.n_jitters
        self.jitter = self.posterior_sample[:, i1:i2]
        self._current_column += self.n_jitters
        self.indices['jitter_start'] = i1
        self.indices['jitter_end'] = i2
        self.indices['jitter'] = slice(i1, i2)
        if self._debug:
            print('finished reading jitters')

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
        else:
            n_trend = 0
        self.total_parameters += n_trend

        if self._debug:
            print('finished reading trend, trend =', self.trend)

    def _read_multiple_instruments(self):
        if self.multi:
            # there are n instruments and n-1 offsets per output
            if self.model == 'RVFWHMmodel':
                n_inst_offsets = 2 * (self.n_instruments - 1)
            elif self.model == 'RVFWHMRHKmodel':
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
        self.total_parameters += n_inst_offsets

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
        self.total_parameters += n_act_ind

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

        self.indices['np'] = self.index_component
        self._current_column += 1

        # indices of the planet parameters
        n_planet_pars = self.max_components * self.n_dimensions
        istart = self._current_column
        iend = istart + n_planet_pars
        self._current_column += n_planet_pars
        self.indices['planets'] = slice(istart, iend)
        
        if self.model == 'GAIAmodel':
            for j, p in zip(range(self.n_dimensions), ('P', 'φ', 'e', 'a', 'ω', 'cosi', 'W')):
                iend = istart + self.max_components
                self.indices[f'planets.{p}'] = slice(istart, iend)
                istart += self.max_components
        else:
            for j, p in zip(range(self.n_dimensions), ('P', 'K', 'φ', 'e', 'ω')):
                iend = istart + self.max_components
                self.indices[f'planets.{p}'] = slice(istart, iend)
                istart += self.max_components

    def _read_studentt(self):
        if self.studentt:
            self.nu = self.posterior_sample[:, self._current_column]
            self.indices['nu'] = self._current_column
            self._current_column += 1
            self.total_parameters += 1

    @property
    def _GP_par_indices(self):
        """
        indices for specific GP hyperparameters:
        eta1_RV, eta1_FWHM, eta2_RV, eta2_FWHM, eta3_RV, eta3_FWHM, eta4_RV, eta4_FWHM
        """
        if self.model == 'RVFWHMmodel':
            i = [0, 1]  # eta1_rv, eta1_fwhm
            i += [2, 2] if self.share_eta2 else [2]
            i += [3, 3] if self.share_eta3 else [3]
            i += [4, 4] if self.share_eta4 else [4]
        elif self.model == 'RVFWHMRHKmodel':
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
        if self.model not in ('GPmodel', 'RVFWHMmodel', 'RVFWHMRHKmodel', 'SPLEAFmodel'):
            self.has_gp = False
            self.n_hyperparameters = 0
            return
        
        self.has_gp = True
        self.GPkernel = 'standard'

        try:
            self.magnetic_cycle_kernel = self.setup['kima']['magnetic_cycle_kernel'] == 'true'
            if self.magnetic_cycle_kernel:
                self.GPkernel = 'standard+magcycle'
        except KeyError:
            pass

        if self.model == 'GPmodel':
            try:
                n_hyperparameters = {
                    'standard': 4,
                    'standard+magcycle': 7,
                }
                n_hyperparameters = n_hyperparameters[self.GPkernel]
            except KeyError:
                raise ValueError(
                    f'GP kernel = {self.GPkernel} not recognized')

            self.n_hyperparameters = n_hyperparameters

        elif self.model == 'RVFWHMmodel':
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

        elif self.model == 'RVFWHMRHKmodel':
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

        elif self.model == 'SPLEAFmodel':
            self.multi_series = self.setup['kima']['multi_series'] == 'true'
            self.nseries = int(self.setup['kima']['nseries'])
            self.n_hyperparameters = 2

        istart = self._current_column
        iend = istart + self.n_hyperparameters
        self.etas = self.posterior_sample[:, istart:iend]

        self._current_column += self.n_hyperparameters
        self.indices['GPpars_start'] = istart
        self.indices['GPpars_end'] = iend
        self.indices['GPpars'] = slice(istart, iend)

        t, e = self.data.t, self.data.e
        kernels = {
            'standard': QPkernel(1, 1, 1, 1),
            'standard+magcycle': QPpMAGCYCLEkernel(1, 1, 1, 1, 1, 1, 1),
            #'periodic': PERkernel(1, 1, 1),
            #'qpc': QPCkernel(1, 1, 1, 1, 1),
            #'RBF': RBFkernel(1, 1),
            #'qp_plus_cos': QPpCkernel(1, 1, 1, 1, 1, 1),
        }

        if self.model == 'RVFWHMmodel':
            self.GP1 = GP(deepcopy(kernels[self.GPkernel]), t, e, white_noise=0.0)
            self.GP2 = GP(deepcopy(kernels[self.GPkernel]), t, self.data.e2, white_noise=0.0)

        if self.model == 'RVFWHMRHKmodel':
            self.GP1 = GP(deepcopy(kernels[self.GPkernel]), t, e, white_noise=0.0)
            self.GP2 = GP(deepcopy(kernels[self.GPkernel]), t, self.data.e2, white_noise=0.0)
            self.GP3 = GP(deepcopy(kernels[self.GPkernel]), t, self.data.e3, white_noise=0.0)

        elif self.model == 'GPmodel_systematics':
            X = np.c_[self.data.t, self._extra_data[:, 3]]
            self.GP = mixtureGP([], X, None, e)

        elif self.model == 'SPLEAFmodel':
            pass
        else:
            self.GP = GP(kernels[self.GPkernel], t, e, white_noise=0.0)

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
        self.total_parameters += n_MAparameters

    def _read_KO(self):
        if self.KO:
            if self.model == 'TRANSITmodel':
                n_KOparameters = 6 * self.nKO
            else:
                n_KOparameters = 5 * self.nKO
            start = self._current_column
            koinds = slice(start, start + n_KOparameters)
            self.KOpars = self.posterior_sample[:, koinds]
            self._current_column += n_KOparameters
            self.indices['KOpars'] = koinds
        else:
            n_KOparameters = 0
        self.total_parameters += n_KOparameters

    def _read_TR(self):
        if self.TR:
            if self.model == 'TRANSITmodel':
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
        self.total_parameters += n_TRparameters

    def _read_outlier(self):
        n_outlier_parameters = 3
        start = self._current_column
        outlier_inds = slice(start, start + n_outlier_parameters)
        self.outlier_pars = self.posterior_sample[:, outlier_inds]
        self._current_column += n_outlier_parameters
        self.indices['outlier'] = outlier_inds
        self.total_parameters += n_outlier_parameters

    @property
    def _mc(self):
        """ Maximum number of Keplerians in the model """
        return self.max_components

    @property
    def _nd(self):
        """ Number of parameters per Keplerian """
        return self.n_dimensions

    @property
    def parameter_priors(self):
        """ A list of priors which can be indexed using self.indices """
        priors = np.full(self.posterior_sample.shape[1], None)

        if self.model == 'RVFWHMmodel':
            for i in range(self.n_instruments):
                priors[i] = self.priors['Jprior']
            for i in range(self.n_instruments, 2 * self.n_instruments):
                priors[i] = self.priors['J2prior']
        else:
            priors[self.indices['jitter']] = self.priors['Jprior']
            if self.model == 'RVmodel':
                priors[0] = self.priors['stellar_jitter_prior']

        if self.trend:
            names = ('slope_prior', 'quadr_prior', 'cubic_prior')
            trend_priors = [self.priors[n] for n in names if n in self.priors]
            priors[self.indices['trend']] = trend_priors

        if self.multi:
            no = self.n_instruments - 1
            if self.model == 'RVFWHMmodel':
                prior1 = self.priors['offsets_prior']
                prior2 = self.priors['offsets2_prior']
                offset_priors = no * [prior1] + no * [prior2]
                priors[self.indices['inst_offsets']] = np.array(offset_priors)
            else:
                prior = self.priors['offsets_prior']
                priors[self.indices['inst_offsets']] = np.array(no * [prior])

        if self.has_gp:
            if self.model == 'GPmodel':
                priors[self.indices['GPpars']] = [
                    self.priors[f'eta{i}_prior'] for i in range(1, 5)
                ]
            elif self.model == 'RVFWHMmodel':
                i = self.indices['GPpars_start']
                priors[i] = self.priors['eta1_1_prior']
                i += 1
                priors[i] = self.priors['eta1_2_prior']
                i += 1
                if self.share_eta2:
                    priors[i] = self.priors['eta2_1_prior']
                    i += 1
                else:
                    priors[i] = self.priors['eta2_1_prior']
                    priors[i + 1] = self.priors['eta2_1_prior']
                    i += 2
                #
                if self.share_eta3:
                    priors[i] = self.priors['eta3_1_prior']
                    i += 1
                else:
                    priors[i] = self.priors['eta3_1_prior']
                    priors[i + 1] = self.priors['eta3_1_prior']
                    i += 2
                #
                if self.share_eta4:
                    priors[i] = self.priors['eta4_1_prior']
                    i += 1
                else:
                    priors[i] = self.priors['eta4_1_prior']
                    priors[i + 1] = self.priors['eta4_1_prior']
                    i += 2

        if self.KO:
            KO_priors = []
            KO_priors += [self.priors[f'KO_Pprior_{i}'] for i in range(self.nKO)]
            KO_priors += [self.priors[f'KO_Kprior_{i}'] for i in range(self.nKO)]
            KO_priors += [self.priors[f'KO_phiprior_{i}'] for i in range(self.nKO)]
            KO_priors += [self.priors[f'KO_eprior_{i}'] for i in range(self.nKO)]
            KO_priors += [self.priors[f'KO_wprior_{i}'] for i in range(self.nKO)]
            priors[self.indices['KOpars']] = KO_priors

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

        try:
            priors[self.indices['vsys']] = self.priors['Cprior']
        except KeyError:
            priors[self.indices['vsys']] = self.priors['Vprior']
            priors[self.indices['C2']] = self.priors['C2prior']

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
            from showresults import showresults
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
            else:
                try:
                    with open(filename, 'rb') as f:
                        res = pickle.load(f)
                except UnicodeDecodeError:
                    with open(filename, 'rb') as f:
                        res = pickle.load(f, encoding='latin1')

            if hasattr(res, 'studentT'):
                res.studentt = res.studentT
                del res.studentT


        except Exception:
            # print('Unable to load data from ', filename, ':', e)
            raise

        res._set_plots()
        return res


    def show_kima_setup(self):
        return _show_kima_setup()

    def save_pickle(self, filename: str, verbose=True):
        """ Pickle this KimaResults object into a file.

        Args:
            filename (str): The name of the file where to save the model
            verbose (bool, optional): Print a message. Defaults to True.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=2)
        if verbose:
            print('Wrote to file "%s"' % f.name)

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

        # jitter(s)
        self.posteriors.jitter = self.posterior_sample[:, self.indices['jitter']]
        if self.n_jitters == 1:
            self.posteriors.jitter = self.posteriors.jitter.ravel()

        if self.has_gp:
            for i in range(self.n_hyperparameters):
                setattr(self.posteriors, f'η{i+1}', self.etas[:, i])
                setattr(self.posteriors, f'_eta{i+1}', self.etas[:, i])

        if self.model == 'GAIAmodel':
            da, dd, mua, mud, plx = self.posterior_sample[:, self.indices['astrometric_solution']].T
            self.posteriors.da = da
            self.posteriors.dd = dd
            self.posteriors.mua = mua
            self.posteriors.mud = mud
            self.posteriors.plx = plx

        # instrument offsets
        if self.multi:
            self.posteriors.offset = self.posterior_sample[:, self.indices['inst_offsets']]
        
        if self.model != 'GAIAmodel':
            # systemic velocity
            self.posteriors.vsys = self.posterior_sample[:, self.indices['vsys']]

        # parameters of the outlier model
        if self.model == 'OutlierRVmodel':
            self.posteriors.outlier_mean, self.posteriors.outlier_sigma, self.posteriors.outlier_Q = \
                self.posterior_sample[:, self.indices['outlier']].T

        max_components = self.max_components
        index_component = self.index_component

        # periods
        i1 = 0 * max_components + index_component + 1
        i2 = 0 * max_components + index_component + max_components + 1
        s = np.s_[i1:i2]
        self.posteriors.P = self.posterior_sample[:, s]

        # amplitudes
        i1 = 1 * max_components + index_component + 1
        i2 = 1 * max_components + index_component + max_components + 1
        s = np.s_[i1:i2]
        self.posteriors.K = self.posterior_sample[:, s]

        # phases
        i1 = 2 * max_components + index_component + 1
        i2 = 2 * max_components + index_component + max_components + 1
        s = np.s_[i1:i2]
        self.posteriors.φ = self.posteriors._phi = self.posterior_sample[:, s]

        # eccentricities
        i1 = 3 * max_components + index_component + 1
        i2 = 3 * max_components + index_component + max_components + 1
        s = np.s_[i1:i2]
        self.posteriors.e = self.posterior_sample[:, s]

        # omegas
        i1 = 4 * max_components + index_component + 1
        i2 = 4 * max_components + index_component + max_components + 1
        s = np.s_[i1:i2]
        self.posteriors.ω = self.posterior_sample[:, s]

        # times of periastron
        self.posteriors.Tp = (self.posteriors.P * self.posteriors.φ) / (2 * np.pi) + self.M0_epoch

        # # times of inferior conjunction (transit, if the planet transits)
        # f = np.pi / 2 - self.Omega
        # ee = 2 * np.arctan(
        #     np.tan(f / 2) * np.sqrt((1 - self.E) / (1 + self.E)))
        # Tc = self.Tp + self.T / (2 * np.pi) * (ee - self.E * np.sin(ee))
        # self.posteriors.Tc = Tc

    def get_medians(self):
        """ return the median values of all the parameters """
        if self.posterior_sample.shape[0] % 2 == 0:
            print(
                'Median is not a solution because number of samples is even!!')

        self.medians = np.median(self.posterior_sample, axis=0)
        self.means = np.mean(self.posterior_sample, axis=0)
        return self.medians, self.means

    def _select_posterior_samples(self, Np=None, mask=None):
        if mask is None:
            mask = np.ones(self.ESS, dtype=bool)

        if Np is None:
            return self.posterior_sample[mask].copy()
        else:
            mask_Np = self.posterior_sample[:, self.index_component] == Np
            return self.posterior_sample[mask & mask_Np].copy()

    def log_prior(self, sample):
        """ Calculate the log prior for a given sample

        Args:
            sample (array): sample for which to calculate the log prior
        
        To evaluate at all posterior samples, consider using
            np.apply_along_axis(self.log_prior, 1, self.posterior_sample)
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

        logp = [p.logpdf(v) if p else 0.0 for p, v in zip(self.parameter_priors, sample)]

        # _np = int(sample[self.indices['np']])
        # st = self.indices['planets'].start
        # k = 0
        # for j in range(self._nd):
        #     for i in range(_np, self._mc):
        #         logp.pop(st + i + 3 * j - k)
        #         k += 1
        # return logp
        return np.sum(logp)

    def log_posterior(self, sample, separate=False):
        logp = self.log_prior(sample)
        index = (self.posterior_sample == sample).sum(axis=1).argmax()
        logl = self.posterior_lnlike[index, 1]
        if separate:
            return logp + logl, logl, logp
        return logp + logl

    def map_sample(self, Np=None, mask=None, printit=True, cache=True):
        from tqdm import tqdm

        if cache and hasattr(self, '_map_sample'):
            map_sample = self._map_sample
        else:
            samples = self._select_posterior_samples(Np, mask)
            logpost = []
            for sample in tqdm(samples):
                logpost.append(self.log_posterior(sample))
            logpost = np.array(logpost)
            ind = logpost.argmax()
            self._map_sample = map_sample = samples[ind]

        logpost, loglike, logprior = self.log_posterior(
            map_sample, separate=True)

        if printit:
            print('Sample with the highest posterior value')
            print(f'(logLike = {loglike:.2f}, logPrior = {logprior:.2f},',
                  end=' ')
            print(f'logPost = {logpost:.2f})')

            if Np is not None:
                print(f'from samples with {Np} Keplerians only')

            msg = '-> might not be representative '\
                  'of the full posterior distribution\n'
            print(msg)

            self.print_sample(map_sample)

        return map_sample

    def maximum_likelihood_sample(self, from_posterior=False, Np=None,
                                  printit=True, mask=None):
        """
        Get the maximum likelihood sample. By default, this is the highest
        likelihood sample found by DNest4. If `from_posterior` is True, this
        returns instead the highest likelihood sample *from those that represent
        the posterior*. The latter may change, due to random choices, between
        different calls to "showresults". If `Np` is given, select only samples
        with that number of planets.
        """
        if self.sample_info is None and not self._lnlike_available:
            print('log-likelihoods are not available! '
                  'maximum_likelihood_sample() doing nothing...')
            return

        if from_posterior:
            if mask is None:
                mask = np.ones(self.ESS, dtype=bool)

            if Np is None:
                ind = np.argmax(self.posterior_lnlike[:, 1])
                maxlike = self.posterior_lnlike[ind, 1]
                pars = self.posterior_sample[ind]
            else:
                mask = self.posterior_sample[:, self.index_component] == Np
                if not mask.any():
                    raise ValueError(f'No samples with {Np} Keplerians')

                ind = np.argmax(self.posterior_lnlike[mask, 1])
                maxlike = self.posterior_lnlike[mask][ind, 1]
                pars = self.posterior_sample[mask][ind]
        else:
            if mask is None:
                mask = np.ones(self.sample.shape[0], dtype=bool)

            if Np is None:
                ind = np.argmax(self.sample_info[mask, 1])
                maxlike = self.sample_info[mask][ind, 1]
                pars = self.sample[mask][ind]
            else:
                mask = self.sample[:, self.index_component] == Np
                if not mask.any():
                    raise ValueError(f'No samples with {Np} Keplerians')

                ind = np.argmax(self.sample_info[mask, 1])
                maxlike = self.sample_info[mask][ind, 1]
                pars = self.sample[mask][ind]

        if printit:
            if from_posterior:
                print('Posterior sample with the highest likelihood value',
                      end=' ')
            else:
                print('Sample with the highest likelihood value', end=' ')

            print('(logL = {:.2f})'.format(maxlike))

            if Np is not None:
                print('from samples with %d Keplerians only' % Np)

            msg = '-> might not be representative '\
                  'of the full posterior distribution\n'
            print(msg)

            self.print_sample(pars)

        return pars

    def median_sample(self, Np=None, printit=True):
        """
        Get the median posterior sample. If `Np` is given, select only from
        posterior samples with that number of planets.
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
            if self.model == 'RVFWHMmodel':
                inst = instruments + instruments
                data = self.n_instruments * ['RV'] + self.n_instruments * ['FWHM']
            else:
                inst = instruments
                data = self.n_instruments * ['']

            for i, jit in enumerate(p[self.indices['jitter']]):
                print(f'  {data[i]:5s} ({inst[i]}): {jit:.2f} m/s')
        else:
            if self.model in ('RVFWHMmodel', 'RVFWHMRHKmodel'):
                print(f'{"RV":>10s}', end=': ')
                print(p[self.indices['jitter']][:self.n_instruments])
                print(f'{"FWHM":>10s}', end=': ')
                print(p[self.indices['jitter']][self.n_instruments:2*self.n_instruments])

                if self.model == 'RVFWHMRHKmodel':
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

            if self.model == 'GAIAmodel':
                pars = ['P', 'phi', 'ecc', 'a', 'w', 'cosi', 'W']
            else:
                pars = ['P', 'K', 'M0', 'e', 'ω']

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
                            P, K, M0, ecc, ω = planet_pars
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

            pars = ('P', 'K', 'M0', 'e', 'ω')
            print((self.n_dimensions * ' {:>10s} ').format(*pars))

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

            pars = ('P', 'K', 'Tc', 'e', 'ω')
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
            if self.model == 'GPmodel':
                pars = ('η1', 'η2', 'η3', 'η4')
            elif self.model == 'RVFWHMmodel':
                pars = ('η1 RV', 'η1 FWHM', 'η2', 'η3', 'η4')
            elif self.model == 'RVFWHMRHKmodel':
                pars = ['η1 RV', 'η1 FWHM', 'η1 RHK']
                pars += ['η2'] if self.share_eta2 else ['η2 RV', 'η2 FWHM', 'η2 RHK']
                pars += ['η3'] if self.share_eta3 else ['η3 RV', 'η3 FWHM', 'η3 RHK']
                pars += ['η4'] if self.share_eta4 else ['η4 RV', 'η4 FWHM', 'η4 RHK']

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

            # , *p[self.indices['GPpars']])
            # if self.GPkernel in (0, 2, 3):
            #     eta1, eta2, eta3, eta4 = pars[self.indices['GPpars']]
            #     print('GP parameters: ', eta1, eta2, eta3, eta4)
            # elif self.GPkernel == 1:
            #     eta1, eta2, eta3 = pars[self.indices['GPpars']]
            #     print('GP parameters: ', eta1, eta2, eta3)

        if self.trend:
            names = ('slope', 'quadr', 'cubic')
            units = ['m/s/yr', 'm/s/yr²', 'm/s/yr³']
            trend = p[self.indices['trend']].copy()
            # transfrom from /day to /yr
            trend *= 365.25**np.arange(1, self.trend_degree + 1)

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

        if self.model != 'GAIAmodel':
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
        start = self.data.t.min() - over * self.data.t.ptp()
        end = self.data.t.max() + over * self.data.t.ptp()
        return np.linspace(start, end, N)

    def _get_ttGP(self, N=1000, over=0.1):
        """ Create array of times for GP prediction plots. """
        kde = gaussian_kde(self.data.t)
        ttGP = kde.resample(N - self.data.N).reshape(-1)
        # constrain ttGP within observed times, to not waste
        ttGP = (ttGP + self.data.t[0]) % self.data.t.ptp() + self.data.t[0]
        # add the observed times as well
        ttGP = np.r_[ttGP, self.data.t]
        ttGP.sort()  # in-place
        return ttGP

    def eval_model(self, sample, t=None,
                   include_planets=True, include_known_object=True,
                   include_indicator_correlations=True,
                   include_trend=True, single_planet: int = None,
                   except_planet: Union[int, List] = None):
        """
        Evaluate the deterministic part of the model at one posterior `sample`.
        If `t` is None, use the observed times. Instrument offsets are only
        added if `t` is None, but the systemic velocity is always added.
        To evaluate at all posterior samples, consider using
            np.apply_along_axis(self.eval_model, 1, self.posterior_sample)

        Note: this function does *not* evaluate the GP component of the model.

        Arguments:
            sample (array): One posterior sample, with shape (npar,)
            t (array):
                Times at which to evaluate the model, or None to use observed
                times
            include_planets (bool):
                Whether to include the contribution from the planets
            include_known_object (bool):
                Whether to include the contribution from the known object
                planets
            include_indicator_correlations (bool):
                Whether to include the indicator correlation model
            include_trend (bool):
                Whether to include the contribution from the trend
            single_planet (int):
                Index of a single planet to *include* in the model, starting at
                1. Use positive values (1, 2, ...) for the Np planets and
                negative values (-1, -2, ...) for the known object planets.
            except_planet (Union[int, List]):
                Index (or list of indices) of a single planet to *exclude* from
                the model, starting at 1. Use positive values (1, 2, ...) for
                the Np planets and negative values (-1, -2, ...) for the known
                object planets.
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

        if self.model == 'RVFWHMmodel':
            v = np.zeros((2, t.size))
        elif self.model == 'RVFWHMRHKmodel':
            v = np.zeros((3, t.size))
        else:
            v = np.zeros_like(t)

        if include_planets:
            if single_planet and except_planet:
                raise ValueError("'single_planet' and 'except_planet' "
                                 "cannot be used together")

            # except_planet should be a list to exclude more than one planet
            if except_planet is not None:
                if isinstance(except_planet, int):
                    except_planet = [except_planet]

            # known_object ?
            if self.KO and include_known_object:
                pars = sample[self.indices['KOpars']].copy()
                for j in range(self.nKO):
                    if single_planet is not None:
                        if j + 1 != -single_planet:
                            continue
                    if except_planet is not None:
                        if j + 1 in except_planet:
                            continue

                    P = pars[j + 0 * self.nKO]
                    K = pars[j + 1 * self.nKO]
                    phi = pars[j + 2 * self.nKO]
                    # t0 = (P * phi) / (2. * np.pi) + self.M0_epoch
                    ecc = pars[j + 3 * self.nKO]
                    w = pars[j + 4 * self.nKO]
                    if self.model not in ('RVmodel', 'GPmodel'):
                        v[0] += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)
                    else:
                        v += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)
            
            # transiting planet ?
            if hasattr(self, 'TR') and self.TR:
                pars = sample[self.indices['TRpars']].copy()
                for j in range(self.nTR):
                    if single_planet is not None:
                        if j + 1 != -single_planet:
                            continue
                    if except_planet is not None:
                        if j + 1 in except_planet:
                            continue

                    P = pars[j + 0 * self.nTR]
                    K = pars[j + 1 * self.nTR]
                    Tc = pars[j + 2 * self.nTR]
                    ecc = pars[j + 3 * self.nTR]
                    w = pars[j + 4 * self.nTR]
                    
                    f = np.pi/2 - w # true anomaly at conjunction
                    E = 2.0 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc))) # eccentric anomaly at conjunction
                    M = E - ecc * np.sin(E) # mean anomaly at conjunction
                    if self.model != 'RVmodel':
                        v[0] += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)
                    else:
                        v += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)

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
                K = pars[j + 1 * self.max_components]
                phi = pars[j + 2 * self.max_components]
                # t0 = (P * phi) / (2. * np.pi) + self.M0_epoch
                ecc = pars[j + 3 * self.max_components]
                w = pars[j + 4 * self.max_components]
                # print(P, K, ecc, w, phi, self.M0_epoch)
                if self.model in ('RVFWHMmodel', 'RVFWHMRHKmodel'):
                    v[0, :] += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)
                else:
                    v += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)

        # systemic velocity (and C2) for this sample
        if self.model == 'RVFWHMmodel':
            C = np.c_[sample[self.indices['vsys']], sample[self.indices['C2']]]
            v += C.reshape(-1, 1)
        elif self.model == 'RVFWHMRHKmodel':
            C = np.c_[sample[self.indices['vsys']], sample[self.indices['C2']], sample[self.indices['C3']]]
            v += C.reshape(-1, 1)
        else:
            v += sample[self.indices['vsys']]

        # if evaluating at the same times as the data, add instrument offsets
        # otherwise, don't
        if self.multi and data_t:  # and len(self.data_file) > 1:
            offsets = sample[self.indices['inst_offsets']]
            ii = self.data.obs.astype(int) - 1

            if self.model in ('RVFWHMmodel', 'RVFWHMRHKmodel'):
                ni = self.n_instruments
                offsets = np.pad(offsets.reshape(-1, ni - 1), ((0, 0), (0, 1)))
                v += np.take(offsets, ii, axis=1)
            else:
                offsets = np.pad(offsets, (0, 1))
                v += np.take(offsets, ii)

        # add the trend, if present
        if include_trend and self.trend:
            trend_par = sample[self.indices['trend']]
            # polyval wants coefficients in reverse order, and vsys was already
            # added so the last coefficient is 0
            trend_par = np.r_[trend_par[::-1], 0.0]
            if self.model in ('RVFWHMmodel', 'RVFWHMRHKmodel'):
                v[0, :] += np.polyval(trend_par, t - self.tmiddle)
            else:
                v += np.polyval(trend_par, t - self.tmiddle)

        # TODO: check if _extra_data is always read correctly
        if hasattr(self, 'indicator_correlations') and self.indicator_correlations and include_indicator_correlations:
            betas = sample[self.indices['betas']].copy()
            interp_u = np.zeros_like(t)
            for i, (c, ai) in enumerate(zip(betas, self.activity_indicators)):
                if ai != '':
                    interp_u += c * np.interp(t, self.data.t, self._extra_data[i])
            v += interp_u

        return v

    def planet_model(self, sample, t=None, include_known_object=True,
                     single_planet: int = None,
                     except_planet: Union[int, List, np.ndarray] = None):
        """
        Evaluate the planet part of the model at one posterior `sample`. If `t`
        is None, use the observed times. To evaluate at all posterior samples,
        consider using

            np.apply_along_axis(self.planet_model, 1, self.posterior_sample)

        Note:
            this function does *not* evaluate the GP component of the model
            nor the systemic velocity and instrument offsets.

        Arguments:
            sample (array):
                One posterior sample, with shape (npar,)
            t (array):
                Times at which to evaluate the model, or None to use observed
                times
            include_known_object (bool):
                Whether to include the contribution from the known object
                planets
            single_planet (int):
                Index of a single planet to *include* in the model, starting at
                1. Use positive values (1, 2, ...) for the Np planets and
                negative values (-1, -2, ...) for the known object planets.
            except_planet (Union[int, List]):
                Index (or list of indices) of a single planet to *exclude* from
                the model, starting at 1. Use positive values (1, 2, ...) for
                the Np planets and negative values (-1, -2, ...) for the known
                object planets.
        """
        if sample.shape[0] != self.posterior_sample.shape[1]:
            n1 = sample.shape[0]
            n2 = self.posterior_sample.shape[1]
            msg = '`sample` has wrong dimensions, expected %d got %d' % (n2,
                                                                         n1)
            raise ValueError(msg)

        if t is None or t is self.data.t:
            t = self.data.t.copy()

        if self.model == 'RVFWHMmodel':
            v = np.zeros((2, t.size))
        elif self.model == 'RVFWHMRHKmodel':
            v = np.zeros((3, t.size))
        else:
            v = np.zeros_like(t)

        if single_planet and except_planet:
            raise ValueError("'single_planet' and 'except_planet' "
                             "cannot be used together")

        # except_planet should be a list to exclude more than one planet
        if except_planet is not None:
            except_planet = np.atleast_1d(except_planet)

        # known_object ?
        if self.KO and include_known_object:
            pars = sample[self.indices['KOpars']].copy()
            for j in range(self.nKO):
                if single_planet is not None:
                    if j + 1 != -single_planet:
                        continue
                if except_planet is not None:
                    if j + 1 in -except_planet:
                        continue

                P = pars[j + 0 * self.nKO]
                K = pars[j + 1 * self.nKO]
                phi = pars[j + 2 * self.nKO]
                # t0 = (P * phi) / (2. * np.pi) + self.M0_epoch
                ecc = pars[j + 3 * self.nKO]
                w = pars[j + 4 * self.nKO]
                v += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)

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
            K = pars[j + 1 * self.max_components]
            phi = pars[j + 2 * self.max_components]
            # t0 = (P * phi) / (2. * np.pi) + self.M0_epoch
            ecc = pars[j + 3 * self.max_components]
            w = pars[j + 4 * self.max_components]
            if self.model in ('RVFWHMmodel', 'RVFWHMRHKmodel'):
                v[0, :] += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)
            else:
                v += keplerian(t, P, K, ecc, w, phi, self.M0_epoch)

        return v

    def stochastic_model(self, sample, t=None, return_std=False, derivative=False,
                         include_jitters=True, **kwargs):
        """
        Evaluate the stochastic part of the model (GP) at one posterior sample.
        If `t` is None, use the observed times. Instrument offsets are only
        added if `t` is None, but the systemic velocity is always added.
        To evaluate at all posterior samples, consider using

            np.apply_along_axis(self.stochastic_model, 1, self.posterior_sample)

        Arguments:
            sample (array):
                One posterior sample, with shape (npar,)
            t (ndarray, optional):
                Times at which to evaluate the model, or None to use observed
                times
            return_std (bool, optional):
                Whether to return the standard deviation of the predictive.
                Default is False.
            derivative (bool, optional):
                Return the first time derivative of the GP prediction instead
            include_jitters (bool, optional):
                Whether to include the jitter values in `sample` in the prediction
        """

        if sample.shape[0] != self.posterior_sample.shape[1]:
            n1 = sample.shape[0]
            n2 = self.posterior_sample.shape[1]
            msg = '`sample` has wrong dimensions, should be %d got %d' % (n2, n1)
            raise ValueError(msg)

        if t is None or t is self.data.t:
            t = self.data.t.copy()

        if not self.has_gp:
            return np.zeros_like(t)

        if self.model == 'RVFWHMmodel':
            D = np.vstack((self.data.y, self.data.y2))
            r = D - self.eval_model(sample)
            GPpars = sample[self.indices['GPpars']]

            η1RV, η1FWHM, η2RV, η2FWHM, η3RV, η3FWHM, η4RV, η4FWHM = GPpars[self._GP_par_indices]
            self.GP1.kernel.pars = np.array([η1RV, η2RV, η3RV, η4RV])
            self.GP2.kernel.pars = np.array([η1FWHM, η2FWHM, η3FWHM, η4FWHM])

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

        elif self.model == 'RVFWHMRHKmodel':
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


        elif self.model == 'SPLEAFmodel':
            r = self.data.y - self.eval_model(sample)
            if self.multi_series:
                raise NotImplementedError()
            else:
                #TODO: move to beginning of file
                from spleaf import cov, term
                C = cov.Cov(
                    self.data.t,
                    err=term.Error(self.data.e),
                    gp=term.Matern52Kernel(1.0, 1.0)
                )
                GPpars = sample[self.indices['GPpars']]
                C.set_param(GPpars, C.param)
                return C.conditional(r, t)

        else:
            r = self.data.y - self.eval_model(sample)
            if self.model == 'GPmodel_systematics':
                x = self._extra_data[:, 3]
                X = np.c_[t, interp1d(self.data.t, x, bounds_error=False)(t)]
                GPpars = sample[self.indices['GPpars']]
                mu = self.GP.predict(r, X, GPpars)
                # print(GPpars)
                # self.GP.kernel.pars = GPpars
                return mu
            else:
                GPpars = sample[self.indices['GPpars']]
                self.GP.kernel.pars = GPpars
                return self.GP.predict(r, t, return_std=return_std)

    def full_model(self, sample, t=None, **kwargs):
        """
        Evaluate the full model at one posterior sample, including the GP. If
        `t` is `None`, use the observed times. Instrument offsets are only added
        if `t` is `None`, but the systemic velocity is always added. To evaluate
        at all posterior samples, consider using
        
            np.apply_along_axis(self.full_model, 1, self.posterior_sample)

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

        Arguments:
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
        offsets = sample[self.indices['inst_offsets']]
        if self.model in ('RVFWHMmodel', 'RVFWHMRHKmodel'):
            offsets = np.pad(offsets.reshape(-1, ni - 1), ((0, 0), (0, 1)))
        else:
            offsets = np.pad(offsets, (0, 1))

        if self._time_overlaps[0]:
            v = np.tile(v, (self.n_instruments, 1))
            if self.model == 'RVFWHMmodel':
                offsets = np.insert(offsets[0],
                                    np.arange(1, offsets.shape[1] + 1),
                                    offsets[1])
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
                v = (v.T + offsets).T
                # this constrains the RV to the times of each instrument
                for i in range(self.n_instruments):
                    obst = self.data.t[self.data.obs == i + 1]
                if i == 0:
                    v[i, t < obst.min()] = np.nan
                if i < self.n_instruments - 1:
                    v[i, t > obst.max()] = np.nan

        else:
            time_bins = np.sort(np.r_[t[0], self._offset_times])
            ii = np.digitize(t, time_bins) - 1

            #! HACK!
            obs_is_sorted = np.all(np.diff(self.data.obs) >= 0)
            if not obs_is_sorted:
                ii = -ii.max() * (ii - ii.max())
            #! end HACK!

            if self.model in ('RVFWHMmodel', 'RVFWHMRHKmodel'):
                v += np.take(offsets, ii, axis=1)
            else:
                v += np.take(offsets, ii)

        return v

    def residuals(self, sample, full=False):
        if self.model == 'RVFWHMmodel':
            D = np.vstack([self.data.y, self.data.y2])
        elif self.model == 'RVFWHMRHKmodel':
            D = np.vstack([self.data.y, self.data.y2, self.data.y3])
        else:
            D = self.data.y

        if full:
            return D - self.full_model(sample)
        else:
            return D - self.eval_model(sample)

    def residual_rms(self, sample, weighted=True, printit=True):
        r = self.residuals(sample, full=True)
        if self.model in ('RVFWHMmodel', 'RVFWHMRHKmodel'):
            r = r[0]

        vals = []
        if weighted:
            val = wrms(r, weights=1 / self.data.e**2)
        else:
            val = rms(r)

        if printit:
            print(f'full: {val:.3f} m/s')

        vals.append(val)

        if self.multi:
            for inst, o in zip(self.instruments, np.unique(self.data.obs)):
                val = wrms(r[self.data.obs == o],
                           weights=1 / self.data.e[self.data.obs == o]**2)
                if printit:
                    print(f'{inst}: {val:.3f} m/s')
                vals.append(val)

        return np.array(vals)

    def from_prior(self, n=1):
        """ Generate `n` samples from the priors for all parameters. """
        prior_samples = []
        for i in range(n):
            prior = []
            for p in self.parameter_priors:
                if p is None:
                    prior.append(None)
                else:
                    prior.append(p.rvs())
            prior_samples.append(np.array(prior))
        return np.array(prior_samples)

    def simulate_from_sample(self, sample, times, add_noise=True, errors=True,
                             append_to_file=False):
        y = self.full_model(sample, times)
        e = np.zeros_like(y)

        if add_noise:
            if self.model == 'RVFWHMmodel':
                n1 = np.random.normal(0, self.e.mean(), times.size)
                n2 = np.random.normal(0, self.e2.mean(), times.size)
                y += np.c_[n1, n2].T
            elif self.model == 'RVmodel':
                n = np.random.normal(0, self.e.mean(), times.size)
                y += n

        if errors:
            if self.model == 'RVFWHMmodel':
                er1 = np.random.uniform(self.e.min(), self.e.max(), times.size)
                er2 = np.random.uniform(self.e2.min(), self.e2.max(),
                                        times.size)
                e += np.c_[er1, er2].T

            elif self.model == 'RVmodel':
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
                if self.model == 'RVFWHMmodel':
                    kw = dict(delimiter='\t', fmt=['%.5f'] + 4 * ['%.9f'])
                    np.savetxt(out, np.c_[times, y[0], e[0], y[1], e[1]], **kw)
                elif self.model == 'RVmodel':
                    kw = dict(delimiter='\t', fmt=['%.5f'] + 2 * ['%.9f'])
                    np.savetxt(out, np.c_[times, y, e], **kw)

        if errors:
            return y, e
        else:
            return y

    @property
    def star(self):
        if self.multi:
            return get_star_name(self.data_file[0])
        else:
            return get_star_name(self.data_file)

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
        bins = np.arange(self.max_components + 2)
        n, _ = np.histogram(self.Np, bins=bins)
        a = np.pad(n[:-1] / n[1:], (0, 1))
        b = np.pad(n[1:] / n[:-1], (1, 0))
        return n[1:] / n[:-1]
        # resNp = np.where(n == np.max(n[(a > 150) | (b > 150)]))[0][0]


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
        # check for overlaps in the time from different instruments
        if not self.multi:
            raise ValueError('Model is not multi_instrument')

        def minmax(x):
            return x.min(), x.max()

        # are the instrument identifiers all sorted?
        # st = np.lexsort(np.vstack([self.t, self.data.obs]))
        obs_is_sorted = np.all(np.diff(self.data.obs) >= 0)

        # if not, which ones are not sorted?
        if not obs_is_sorted:
            which_not_sorted = np.unique(
                self.data.obs[1:][np.diff(self.data.obs) < 0])

        overlap = []
        for i in range(1, self.n_instruments):
            t1min, t1max = minmax(self.data.t[self.data.obs == i])
            t2min, t2max = minmax(self.data.t[self.data.obs == i + 1])
            # if the instrument IDs are sorted or these two instruments
            # (i and i+1) are not the ones not-sorted
            if obs_is_sorted or i not in which_not_sorted:
                if t2min < t1max:
                    overlap.append((i, i + 1))
            # otherwise the check is different
            else:
                if t1min < t2max:
                    overlap.append((i, i + 1))

        return len(overlap) > 0, overlap

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

    @property
    def data_properties(self):
        t = self.data.t
        prop = {
            'time span': (t.ptp(), 'days', True),
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
    #         i = 2 if self.model == 'RVFWHMmodel' else 1
    #         return self.posterior_sample[:, self.indices['GPpars']][:, i]
    #     return None

    # @property
    # def eta3(self):
    #     if self.has_gp:
    #         i = 3 if self.model == 'RVFWHMmodel' else 2
    #         return self.posterior_sample[:, self.indices['GPpars']][:, i]
    #     return None

    # @property
    # def eta4(self):
    #     if self.has_gp:
    #         i = 4 if self.model == 'RVFWHMmodel' else 3
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

        for par in ('P', 'K', 'φ', 'e', 'ω'):
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


    #
    def _set_plots(self):
        from functools import partial
        if self.model in ('RVFWHMmodel', 'RVFWHMRHKmodel'):
            self.plot_random_samples = self.plot6 = partial(display.plot_random_samples_multiseries, res=self)
        else:
            self.plot_random_samples = self.plot6 = partial(display.plot_random_samples, res=self)

    #
    hist_vsys = display.hist_vsys
    hist_jitter = display.hist_jitter
    hist_correlations = display.hist_correlations
    hist_trend = display.hist_trend
    hist_MA = display.hist_MA
    hist_nu = display.hist_nu
