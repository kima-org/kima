from typing import Any, Optional, overload, Typing, Sequence
from enum import Enum
import kima.Data

class GAIAdata:
    """
    docs
    """

    def __init__(self, filename: str, units: str = 'ms', skip: int = 0, max_rows: int = 0, delimiter: str = ' ') -> None:
        """
        Load astrometric data from a file
        """
        ...
    
    @property
    def pf(self) -> list[float]:
        """
        the parallax factors
        """
        ...
    
    @property
    def psi(self) -> list[float]:
        """
        The Gaia scan angles
        """
        ...
    
    @property
    def t(self) -> list[float]:
        """
        The times of observations
        """
        ...
    
    @property
    def w(self) -> list[float]:
        """
        The observed centroid positions
        """
        ...
    
    @property
    def wsig(self) -> list[float]:
        """
        The observed centroid position uncertainties
        """
        ...
    
class PHOTdata:
    """
    docs
    """

    @property
    def N(self) -> int:
        """
        Total number of observations
        """
        ...
    
    def __init__(self, filename: str, units: str = 'ms', skip: int = 0, delimiter: str = ' ') -> None:
        """
        Load photometric data from a file
        """
        ...
    
    @property
    def sig(self) -> list[float]:
        """
        The observed flux uncertainties
        """
        ...
    
    @property
    def t(self) -> list[float]:
        """
        The times of observations
        """
        ...
    
    @property
    def y(self) -> list[float]:
        """
        The observed flux
        """
        ...
    
class RVData:
    """
    Load and store RV data
    """

    @property
    def M0_epoch(self) -> float:
        """
        reference epoch for the mean anomaly
        """
        ...
    @M0_epoch.setter
    def M0_epoch(self, arg: float, /) -> None:
        """
        reference epoch for the mean anomaly
        """
        ...
    
    @property
    def N(self) -> int:
        """
        Total number of observations
        """
        ...
    
    def __init__(self, t: list[list[float]], y: list[list[float]], sig: list[list[float]], units: str = 'ms', instruments: list[str] = []) -> None:
        """
        Load RV data from arrays, for multiple instruments
        """
        ...
    
    @overload
    def __init__(self, filenames: list[str], units: str = 'ms', skip: int = 0, max_rows: int = 0, delimiter: str = ' ', indicators: list[str] = []) -> None:
        """
        Load RV data from a list of files
        """
        ...
    
    @overload
    def __init__(self, filename: str, units: str = 'ms', skip: int = 0, max_rows: int = 0, delimiter: str = ' ', indicators: list[str] = []) -> None:
        """
        Load RV data from a file
        """
        ...
    
    @overload
    def __init__(self, t: list[float], y: list[float], sig: list[float], units: str = 'ms', instrument: str = '') -> None:
        """
        Load RV data from arrays
        """
        ...
    
    @property
    def actind(self) -> list[list[float]]:
        """
        Activity indicators
        """
        ...
    
    def get_RV_span(self) -> float:
        ...
    
    def get_timespan(self) -> float:
        ...
    
    @property
    def instrument(self) -> str:
        """
        instrument name
        """
        ...
    @instrument.setter
    def instrument(self, arg: str, /) -> None:
        """
        instrument name
        """
        ...
    
    def load(self, filename: str, units: str, skip: int, max_rows: int, delimiter: str, indicators: list[str]) -> None:
        """
        Load RV data from a tab/space separated file with columns
        ```
        time  vrad  error  quant  error
        ...   ...   ...    ...    ...
        ```
        Args:
        filename (str): the name of the file
        untis (str): units of the RVs and errors, either "kms" or "ms"
        skip (int): number of lines to skip in the beginning of the file (default = 2)
        indicators (list[str]): nodoc
        """
        ...
    
    @property
    def multi(self) -> bool:
        """
        Data comes from multiple instruments
        """
        ...
    
    @property
    def obsi(self) -> list[int]:
        """
        The instrument identifier
        """
        ...
    
    def plot(*args, **kwargs):
        """
        Simple plot of RV data
        """
        ...
    
    @property
    def sig(self) -> list[float]:
        """
        The observed RV uncertainties
        """
        ...
    
    @property
    def skip(self) -> int:
        """
        Lines skipped when reading data
        """
        ...
    
    @property
    def t(self) -> list[float]:
        """
        The times of observations
        """
        ...
    
    def topslope(self) -> float:
        ...
    
    @property
    def units(self) -> str:
        """
        Units of the RVs and uncertainties
        """
        ...
    
    @property
    def y(self) -> list[float]:
        """
        The observed radial velocities
        """
        ...
    
class loadtxt:
    """
    None
    """

    def __init__(self, arg: str, /) -> None:
        ...
    
    def comments(self, arg: str, /) -> kima.Data.loadtxt:
        ...
    
    def delimiter(self, arg: str, /) -> kima.Data.loadtxt:
        ...
    
    def max_rows(self, arg: int, /) -> kima.Data.loadtxt:
        ...
    
    def skiprows(self, arg: int, /) -> kima.Data.loadtxt:
        ...
    
    def usecols(self, arg: list[int], /) -> kima.Data.loadtxt:
        ...
    
class multiRVData:
    """
    docs
    """

    @property
    def N(self) -> int:
        """
        Total number of observations
        """
        ...
    
    def __init__(self, filenames: list[list[str]], units: str = 'ms', skip: int = 0, max_rows: int = 0, delimiter: str = ' ') -> None:
        ...
    
    @property
    def full_sig(self) -> list[float]:
        """
        The observed RV uncertainties
        """
        ...
    
    @property
    def full_t(self) -> list[float]:
        """
        The times of observations
        """
        ...
    
    @property
    def full_y(self) -> list[float]:
        """
        The observed radial velocities
        """
        ...
    
    @property
    def sig(self) -> list[list[float]]:
        """
        The observed RV uncertainties
        """
        ...
    
    @property
    def t(self) -> list[list[float]]:
        """
        The times of observations
        """
        ...
    
    @property
    def y(self) -> list[list[float]]:
        """
        The observed radial velocities
        """
        ...
    
