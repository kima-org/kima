# flake8: noqa
# ruff: noqa

__all__ = [
    'RVData', 'PHOTdata', 'GAIAdata', 'ETVData',
    'RVmodel', 'GPmodel', 'RVFWHMmodel', 'TRANSITmodel', 'OutlierRVmodel', 'BINARIESmodel',
    'GAIAmodel', 'RVGAIAmodel',
    '__models__',
    'keplerian', 'distributions',
    'run', 'load_results',
]

import sys
from enum import Enum

from .Data import RVData, PHOTdata, GAIAdata, ETVData

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
)
MODELS = Enum('MODELS', {m.__name__: m.__name__ for m in __models__})


# add plot method to data classes
from .pykima.display import plot_RVData
RVData.plot = plot_RVData


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
from . import GP


# examples
from . import examples
