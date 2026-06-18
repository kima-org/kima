# flake8: noqa
# ruff: noqa

__all__ = [
    'RVData', 'PHOTdata', 'GAIAdata', 'ETVData', 'HGPMdata',
    'RVmodel', 'GPmodel', 'RVFWHMmodel', 'TRANSITmodel', 'OutlierRVmodel', 
    'BINARIESmodel', 'GAIAmodel', 'RVGAIAmodel',
    'MODELS',
    'keplerian', 'post_keplerian', 'distributions',
    'run', 'load_results', 'chdir',
]

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = None

import sys
from enum import Enum

from .pykima.utils import chdir

from .Data import RVData, PHOTdata, GAIAdata, ETVData
from .Data import HGPMdata as HGPMdata_original

from .RVmodel import RVmodel
from .GPmodel import GPmodel

from .RVFWHMmodel import RVFWHMmodel
from .RVFWHMRHKmodel import RVFWHMRHKmodel
from .SPLEAFmodel import SPLEAFmodel

from .TRANSITmodel import TRANSITmodel

from .OutlierRVmodel import OutlierRVmodel
from .BINARIESmodel import BINARIESmodel

from .GAIAmodel import GAIAmodel
from .RVGAIAmodel import RVGAIAmodel
from .ETVmodel import ETVmodel
from .ApodizedRVmodel import ApodizedRVmodel
from .RVHGPMmodel import RVHGPMmodel

__models__ = (
    RVmodel,
    GPmodel,
    RVFWHMmodel,
    RVFWHMRHKmodel,
    SPLEAFmodel,
    TRANSITmodel,
    OutlierRVmodel,
    BINARIESmodel,
    GAIAmodel,
    RVGAIAmodel,
    ETVmodel,
    ApodizedRVmodel,
    RVHGPMmodel,
)
MODELS = Enum('MODELS', {m.__name__: m.__name__ for m in __models__})


# add plot method to data classes
from .pykima.display_hooks import plot_RVData, plot_HGPMdata
RVData.plot = plot_RVData

class HGPMdata(HGPMdata_original):
    __doc__ = HGPMdata_original.__doc__
    def __init__(self, *args, **kwargs):
        import pooch
        file_path = pooch.retrieve(
            url="https://cdsarc.cds.unistra.fr/ftp/J/ApJS/254/42/HGCA_vEDR3.fits",
            known_hash='23684d583baaa236775108b360c650e79770a695e16914b1201f290c1826065c',
            path=self._temp_path,
            fname='HGCA_vEDR3.fits',
            progressbar=True
        )
        return super().__init__(*args, **kwargs)

    def plot(self, *args, **kwargs):
        return plot_HGPMdata(self, *args, **kwargs)

HGPMdata_original.plot = HGPMdata.plot


this = sys.modules[__name__]
this._SOUNDS_ = False
from .pykima.utils import sounds, maybe_success_sound, maybe_error_sound

# kima.run, and wrapper with sounds
from .Sampler import run as _run_really

def run(*args, **kwargs):
    try:
        _ = _run_really(*args, **kwargs)
    except Exception as e:
        maybe_error_sound()
        print()
        raise e from None
    else:
        maybe_success_sound()

run.__doc__ = _run_really.__doc__


# kima.load_results
from .pykima.results import load_results, KimaResults
# kima.cleanup
from .pykima.cli import cli_clean as cleanup

# sub-packages
from .kepler import keplerian
from .postkepler import post_keplerian

# couldn't find a better way to do this...
def _keplerian_wrapper_for_docs():
    """
    Calculate the Keplerian curve of one planet at times `t`

    Args:
        t (array):
            Times at which to calculate the Keplerian function
        P (float):
            Orbital period [days]
        K (float):
            Semi-amplitude
        ecc (float):
            Orbital eccentricity
        w (float):
            Argument of periastron [rad]
        M0 (float):
            Mean anomaly at the epoch [rad]
        M0_epoch (float):
            Reference epoch for the mean anomaly (M=0 at this time) [days]

    Returns:
        v (array):
            Keplerian function evaluated at input times `t`
    """
    pass

def _keplerian_gaia_wrapper_for_docs():
    """
    Calculate the Keplerian curve of one planet at times `t`

    Args:
        t (array):
            Times at which to calculate the Keplerian function
        psi (array):
            Scan angle of the Gaia satellite at time t [rad]
        A (float):
            Thiele-innes parameter A [mas]
        B (float):
            Thiele-innes parameter B [mas]
        F (float):
            Thiele-innes parameter F [mas]
        G (float):
            Thiele-innes parameter G [mas]
        ecc (float):
            Orbital eccentricity
        P (float):
            Orbital period P [days]
        M0 (float):
            Mean anomaly at the epoch [rad]
        M0_epoch (float):
            Reference epoch for the mean anomaly (M=0 at this time) [days]

    Returns:
        wk (array):
            Gaia along-scan 'abscissa' function for a keplerian orbit evaluated
            at input times `t`
    """

def _post_keplerian_wrapper_for_docs():
    """
    Calculate the Keplerian curve of the orbit of a dark companion around a star
    at times `t` with post-Keplerian additions. Suited to the orbit of a close
    binary star.

    Args:
        t (array):
            Times at which to calculate the Keplerian function
        P (float):
            Orbital period [days]
        K (float):
            Semi-amplitude
        ecc (float):
            Orbital eccentricity
        w (float):
            Argument of pericentre [rad]
        wdot (float):
            Pericentre precession rate [arcsecs/year]
        M0 (float):
            Mean anomaly at the epoch [rad]
        M0_epoch (float):
            Reference epoch for the mean anomaly (M=0 at this time) [days]
        cosi (float):
            Cosine of the inclination angle of the orbit (=0 for an edge-on orbit)
        M1 (float):
            Mass of primary star [Msun]
        M2 (float):
            Mass of secondary star [Msun]
        R1 (float):
            Radius of primary star [Rsun] If not specfied and tidal correction
            included the relation R = M^0.8 will be used.
        GR (bool):
            Whether to include the radial velocity corrections from General
            Relativity (Transverse Doppler, Light Travel-Time, and
            Graviatational Redshift)
        Tid (bool):
            Whether to include the radial velocity correction from Tides 
            (only suitable for circular orbits)
        Kprec (float):
            What precision in m/s to calculate K2 to for the relativistic
            correction (defaults to 50 m/s) 

    Returns:
        v (array):
            Keplerian function with potential corrections evaluated at input
            times `t`
    """
    pass


#from . import spleaf
from . import distributions
from . import kmath
from . import GP


# examples
from . import examples
