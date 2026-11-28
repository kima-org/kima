from collections.abc import Sequence
from typing import overload

import numpy
from numpy.typing import NDArray


@overload
def murison_solver(M: float, ecc: float) -> float: ...

@overload
def murison_solver(M: NDArray[numpy.float64], ecc: float) -> list[float]: ...

def murison_keplerian(arg0: Sequence[float], arg1: float, arg2: float, arg3: float, arg4: float, arg5: float, arg6: float, /) -> list[float]: ...

@overload
def nijenhuis_solver(M: float, ecc: float) -> float: ...

@overload
def nijenhuis_solver(M: NDArray[numpy.float64], ecc: float) -> list[float]: ...

@overload
def brandt_solver(M: float, ecc: float) -> float: ...

@overload
def brandt_solver(M: NDArray[numpy.float64], ecc: float) -> list[float]: ...

def contour_solver(M: float, ecc: float) -> float: ...

def keplerian(t: Sequence[float], P: float, K: float, ecc: float, w: float, M0: float, M0_epoch: float) -> NDArray[numpy.float64]:
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

def keplerian_rvpm(t_rv: Sequence[float], t_pm: tuple[Sequence[float], Sequence[float], Sequence[float]], parallax: float, P: float, K: float, ecc: float, w: float, M0: float, M0_epoch: float, inc: float, Omega: float) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64], NDArray[numpy.float64], NDArray[numpy.float64]]:
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

def keplerian2(t: Sequence[float], P: float, K: float, ecc: float, w: float, M0: float, M0_epoch: float) -> list[float]: ...

def keplerian_etv(epochs: Sequence[float], P: float, K: float, ecc: float, w: float, M0: float, ephem1: float) -> NDArray[numpy.float64]:
    """
    Calculate the LTTE Keplerian curve of one planet at each epoch

    Args: 
        epochs (array):
            epochs at which to calculate the Keplerian function, i.e. the numbering of the eclipses
        P (float):
            Orbital period of planet[days]
        K (float):
            Semi-amplitude [s]
        ecc (float):
            Orbital eccentricity
        w (float):
            Argument of periastron [rad]
        M0 (float):
            Mean anomaly at the epoch [rad]
        ephem1 (float):
            Linear ephemeris term, i.e. orbital period of the binary at reference epoch = 0 [days]

    Returns:
        tau (array):
            Keplerian function evaluated at input epochs [s]
    """

def keplerian_gaia(t: Sequence[float], psi: Sequence[float], A: float, B: float, F: float, G: float, ecc: float, P: float, M0: float, M0_epoch: float) -> NDArray[numpy.float64]:
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
            Gaia along-scan 'abscissa' function for a keplerian orbit evaluated at input times `t`
    """
