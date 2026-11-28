from collections.abc import Sequence
from typing import overload

from kima import plot as plot


class loadtxt:
    def __init__(self, arg: str, /) -> None: ...

    def skiprows(self, arg: int, /) -> loadtxt: ...

    def comments(self, arg: str, /) -> loadtxt: ...

    def delimiters(self, arg: str, /) -> loadtxt: ...

    def usecols(self, arg: Sequence[int], /) -> loadtxt: ...

    def max_rows(self, arg: int, /) -> loadtxt: ...

    def max_errors(self, arg: int, /) -> loadtxt: ...

    def __call__(self) -> list[list[float]]: ...

class RVdata:
    """Load and store RV data"""

    @overload
    def __init__(self, filenames: Sequence[str], units: str = 'ms', skip: int = 0, max_rows: int = 0, delimiter: str = ' \t,', indicators: Sequence[str] = [], double_lined: bool = False) -> None:
        """
        Load RV data from a list of files.

        Args:
            filenames (list):
                List of filenames to load
            units (str):
                Units of the data ('ms' or 'kms')
            skip (int):
                Number of lines to skip at the top of the file
            max_rows (int):
                Maximum number of rows to read
            delimiter (str):
                Delimiter between columns
            indicators (list):
                List of names for the indicator columns (should be present in all files)
            double_lined (bool):
                Whether the data is for a double-lined binary
        """

    @overload
    def __init__(self, filename: str, units: str = 'ms', skip: int = 0, max_rows: int = 0, multi: bool = False, delimiter: str = ' \t,', indicators: Sequence[str] = [], double_lined: bool = False) -> None:
        """
        Load RV data from a file.

        Args:
            filename (str):
                Name of the file to read
            units (str):
                Units of the data ('ms' or 'kms')
            skip (int):
                Number of lines to skip at the top of the file
            max_rows (int):
                Maximum number of rows to read
            delimiter (str):
                Delimiter between columns
            indicators (list):
                List of names for the indicator columns
            double_lined (bool):
                Whether the data is for a double-lined binary
        """

    @overload
    def __init__(self, t: Sequence[float], y: Sequence[float], sig: Sequence[float], indicators: Sequence[Sequence[float]] = [], units: str = 'ms', instrument: str = '') -> None:
        """
        Load RV data from arrays for a single instrument.

        Args:
            t (list, array):
                List of observation times
            y (list, array):
                List of RV values
            sig (list, array):
                List of RV uncertainties
            indicators (list[list]):
                Lists of indicator values
            units (str):
                Units of the radial velocity data ('ms' or 'kms')
            instrument (str):
                Name of the instrument
        """

    @overload
    def __init__(self, t: Sequence[Sequence[float]], y: Sequence[Sequence[float]], sig: Sequence[Sequence[float]], indicators: Sequence[Sequence[Sequence[float]]] = [], units: str = 'ms', instruments: Sequence[str] = []) -> None:
        """
        Load RV data from arrays for multiple instruments.

        Args:
            t (list[list], array[array]):
                Lists of observation times
            y (list[list], array[array]):
                Lists of RV values
            sig (list[list], array[array]):
                Lists of RV uncertainties
            indicators (list[list]):
                Lists of indicator values (3D array)
            units (str):
                Units of the radial velocity data ('ms' or 'kms')
            instrument (str):
                Name of the instrument
        """

    @property
    def datafile(self) -> str:
        """The file name"""

    @property
    def datafiles(self) -> list[str]:
        """The list of file names"""

    @property
    def t(self) -> list[float]:
        """The times of observations"""

    @property
    def y(self) -> list[float]:
        """The observed radial velocities"""

    @property
    def sig(self) -> list[float]:
        """The observed RV uncertainties"""

    @property
    def y2(self) -> list[float]:
        """The observed secondary radial velocities (double-lined binary)"""

    @property
    def sig2(self) -> list[float]:
        """The observed secondary RV uncertainties (double-lined binary)"""

    @property
    def obsi(self) -> list[int]:
        """The instrument identifier"""

    @property
    def N(self) -> int:
        """Total number of observations"""

    @property
    def actind(self) -> list[list[float]]:
        """Activity indicators"""

    @property
    def normalized_actind(self) -> list[list[float]]:
        """Activity indicators normalized to [0,1]"""

    @property
    def units(self) -> str:
        """Units of the RVs and uncertainties"""

    @property
    def multi(self) -> bool:
        """Data comes from multiple instruments"""

    @property
    def skip(self) -> int:
        """Lines skipped when reading data"""

    @property
    def instrument(self) -> str:
        """instrument name"""

    @instrument.setter
    def instrument(self, arg: str, /) -> None: ...

    @property
    def instruments(self) -> list[str]:
        """instrument names"""

    @instruments.setter
    def instruments(self, arg: Sequence[str], /) -> None: ...

    @property
    def indicator_names(self) -> list[str]:
        """names of activity indicators"""

    @indicator_names.setter
    def indicator_names(self, arg: Sequence[str], /) -> None: ...

    @property
    def M0_epoch(self) -> float:
        """reference epoch for the mean anomaly"""

    @M0_epoch.setter
    def M0_epoch(self, arg: float, /) -> None: ...

    @property
    def trend_epoch(self) -> float:
        """reference epoch for the trend"""

    @trend_epoch.setter
    def trend_epoch(self, arg: float, /) -> None: ...

    @property
    def double_lined(self) -> bool:
        """if the data is for a double-lined binary"""

    @double_lined.setter
    def double_lined(self, arg: bool, /) -> None: ...

    def __getstate__(self) -> tuple[list[float], list[float], list[float], list[float], list[float], list[int], list[list[float]], list[str], list[str], str, str, bool, bool, int, float, float, list[float], str, list[str], int]: ...

    def __setstate__(self, arg: tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float], Sequence[float], Sequence[int], Sequence[Sequence[float]], Sequence[str], Sequence[str], str, str, bool, bool, int, float, float, Sequence[float], str, Sequence[str], int], /) -> None: ...

    def get_t_min(self) -> float:
        """Get the minimum time"""

    def get_t_max(self) -> float:
        """Get the maximum time"""

    def get_timespan(self) -> float:
        """Get the timespan"""

    def get_t_middle(self) -> float:
        """Get the middle time"""

    def get_RV_min(self) -> float:
        """Get the minimum RV"""

    def get_RV_max(self) -> float:
        """Get the maximum RV"""

    def get_RV_span(self) -> float:
        """Get the RV span"""

    def get_max_RV_span(self) -> float:
        """Get the maximum RV span of individual instruments"""

    def get_RV_mean(self) -> float:
        """Get the mean RV"""

    def get_RV_var(self) -> float:
        """Get the variance of RVs"""

    def get_RV_std(self) -> float:
        """Get the standard deviation of RVs"""

    def get_actind_min(self, arg: int, /) -> float:
        """Get the minimum value of activity indicator i"""

    def get_actind_max(self, arg: int, /) -> float:
        """Get the maximum value of activity indicator i"""

    def get_actind_span(self, arg: int, /) -> float:
        """Get the span of activity indicator i"""

    def get_actind_mean(self, arg: int, /) -> float:
        """Get the mean of activity indicator i"""

    def get_actind_var(self, arg: int, /) -> float:
        """Get the variance of activity indicator i"""

    def get_actind_std(self, arg: int, /) -> float:
        """Get the standard deviation of activity indicator i"""

    def topslope(self) -> float:
        """Get the maximum slope allowed by the data"""

    def get_trend_magnitude(self, degree: int, i: int = -1) -> int:
        """Order of magnitude for trend coefficient (of `degree`) given the data"""

    def get_unique_t(self) -> list[float]:
        """Get the unique times"""

class RVData(RVdata):
    """Deprecated alias of RVdata"""

class PHOTdata:
    """docs"""

    def __init__(self, filename: str, skip: int = 0, delimiter: str = ' \t,') -> None:
        """Load photometric data from a file"""

    @property
    def sig(self) -> list[float]:
        """The observed flux uncertainties"""

    @property
    def obsi(self) -> list[int]:
        """The instrument identifier"""

    @property
    def N(self) -> int:
        """Total number of observations"""

    @property
    def datafile(self) -> str:
        """The file name"""

    @property
    def datafiles(self) -> list[str]:
        """The list of file names"""

    @property
    def t(self) -> list[float]:
        """The times of observations"""

    @property
    def y(self) -> list[float]:
        """The observed flux"""

    def __getstate__(self) -> tuple[int, list[float], list[float], list[float], list[int], str, float, str, list[str], int]: ...

    def __setstate__(self, arg: tuple[int, Sequence[float], Sequence[float], Sequence[float], Sequence[int], str, float, str, Sequence[str], int], /) -> None: ...

class GAIAdata:
    """Load and store Gaia astrometric data"""

    def __init__(self, filename: str, units: str = 'mas', skip: int = 0, max_rows: int = 0, delimiter: str = ' \t,') -> None:
        """
        Load astrometric data from a file

        Args:
            filename (str):
                Name of the file to read
            units (str):
                Units of the data (must be 'mas')
            skip (int):
                Number of lines to skip at the top of the file
            max_rows (int):
                Maximum number of rows to read
            delimiter (str):
                Delimiter between columns
        """

    @property
    def datafile(self) -> str:
        """The file name"""

    @property
    def t(self) -> list[float]:
        """The times of observations"""

    @property
    def w(self) -> list[float]:
        """The observed centroid positions"""

    @property
    def wsig(self) -> list[float]:
        """The observed centroid position uncertainties"""

    @property
    def psi(self) -> list[float]:
        """The Gaia scan angles"""

    @property
    def pf(self) -> list[float]:
        """The parallax factors"""

    @property
    def N(self) -> int:
        """Total number of observations"""

    @property
    def M0_epoch(self) -> float:
        """reference epoch for the mean anomaly"""

    @M0_epoch.setter
    def M0_epoch(self, arg: float, /) -> None: ...

    @property
    def units(self) -> str:
        """Units of the Gaia data and uncertainties"""

    def __getstate__(self) -> tuple[int, list[float], list[float], list[float], list[float], list[float], str, float, str, int]: ...

    def __setstate__(self, arg: tuple[int, Sequence[float], Sequence[float], Sequence[float], Sequence[float], Sequence[float], str, float, str, int], /) -> None: ...

class HGPMdata:
    """Load and store Hipparcos-Gaia proper motion data"""

    def __init__(self, gaia_id: int) -> None:
        """Load the Hipparcos-Gaia Catalog of Accelerations"""

    @property
    def gaia_id(self) -> int:
        """Gaia DR3 ID"""

    @property
    def parallax_gaia(self) -> float:
        """Gaia DR3 parallax"""

    @property
    def parallax_gaia_error(self) -> float:
        """Gaia DR3 parallax error"""

    @property
    def epoch_ra_hip(self) -> float:
        """Central epoch of Hipparcos RA measurement"""

    @property
    def epoch_dec_hip(self) -> float:
        """Central epoch of Hipparcos DEC measurement"""

    @property
    def epoch_ra_gaia(self) -> float:
        """Central epoch of Gaia RA measurement"""

    @property
    def epoch_dec_gaia(self) -> float:
        """Central epoch of Gaia DEC measurement"""

    @property
    def pm_ra_hip(self) -> float:
        """Calibrated proper motion in RA from the composite Hipparcos catalog"""

    @property
    def pm_dec_hip(self) -> float:
        """Calibrated proper motion in DEC from the composite Hipparcos catalog"""

    @property
    def pm_ra_gaia(self) -> float:
        """Gaia EDR3 proper motion in RA"""

    @property
    def pm_dec_gaia(self) -> float:
        """Gaia EDR3 proper motion in DEC"""

    @property
    def pm_ra_hg(self) -> float:
        """
        Calibrated proper motion in RA from the Hipparcos-Gaia positional difference
        """

    @property
    def pm_dec_hg(self) -> float:
        """
        Calibrated proper motion in DEC from the Hipparcos-Gaia positional difference
        """

    @property
    def sig_hip_ra(self) -> float: ...

    @property
    def sig_hip_dec(self) -> float: ...

    @property
    def rho_hip(self) -> float: ...

    @property
    def sig_gaia_ra(self) -> float: ...

    @property
    def sig_gaia_dec(self) -> float: ...

    @property
    def rho_gaia(self) -> float: ...

    @property
    def sig_hg_ra(self) -> float: ...

    @property
    def sig_hg_dec(self) -> float: ...

    @property
    def rho_hg(self) -> float: ...

    @property
    def chisq(self) -> float: ...

    def __getstate__(self) -> tuple[int]: ...

    def __setstate__(self, arg: tuple[int], /) -> None: ...

class ETVData:
    """docs"""

    def __init__(self, filename: str, units: str = 'days', skip: int = 0, max_rows: int = 0, delimiter: str = ' \t,') -> None:
        """Load Eclipse timing data from a file"""

    @property
    def datafile(self) -> str:
        """The file name"""

    @property
    def epochs(self) -> list[float]:
        """The epoch (Nth eclipse since number 0)"""

    @property
    def et(self) -> list[float]:
        """The observed mid-eclipse times"""

    @property
    def etsig(self) -> list[float]:
        """The uncertainties in the eclipse times"""

    @property
    def N(self) -> int:
        """Total number of observations"""

    @property
    def obsi(self) -> list[int]:
        """The instrument identifier"""

    @property
    def M0_epoch(self) -> float:
        """
        reference epoch for the mean anomaly, not used in ETVmodel as reference time is a free parameter
        """

    @M0_epoch.setter
    def M0_epoch(self, arg: float, /) -> None: ...

    @property
    def units(self) -> str:
        """Units of the ETV data and uncertainties"""
