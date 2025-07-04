# flake8: noqa
# ruff: noqa

__all__ = [
    'RVData', 'PHOTdata', 'GAIAdata', 'ETVData', 'HGPMdata',
    'RVmodel', 'GPmodel', 'RVFWHMmodel', 'TRANSITmodel', 'OutlierRVmodel', 'BINARIESmodel',
    'GAIAmodel', 'RVGAIAmodel',
    'MODELS',
    'keplerian', 'distributions',
    'run', 'load_results',
]

import sys
from enum import Enum

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
    RVHGPMmodel,
)
MODELS = Enum('MODELS', {m.__name__: m.__name__ for m in __models__})


# add plot method to data classes
from .pykima.display import plot_RVData, plot_HGPMdata
RVData.plot = plot_RVData

class HGPMdata(HGPMdata_original):
    __doc__ = HGPMdata_original.__doc__
    def __init__(self, *args, **kwargs):
        import pooch
        file_path = pooch.retrieve(
            url="https://cdsarc.cds.unistra.fr/ftp/J/ApJS/254/42/HGCA_vEDR3.fits",
            known_hash='23684d583baaa236775108b360c650e79770a695e16914b1201f290c1826065c',
            path=self._temp_path,
            fname='HGCA_vEDR3.fits'
        )
        return super().__init__(*args, **kwargs)

    def plot(self, *args, **kwargs):
        return plot_HGPMdata(self, *args, **kwargs)




# kima.run
from .Sampler import run
# kima.load_results
from .pykima.results import load_results, KimaResults
# kima.cleanup
from .pykima.cli import cli_clean as cleanup

# sub-packages
from .kepler import keplerian
#from . import spleaf
from . import distributions
from . import kmath
from . import GP


# examples
from . import examples
