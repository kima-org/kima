from typing import Any, Optional, overload, Typing, Sequence
from enum import Enum
import kima.kepler

def brandt_solver(M: numpy.typing.NDArray, ecc: float) -> list[float]:
    """
    brandt_solver(M: ndarray[dtype=float64], ecc: float) -> list[float]
    """
    ...

@overload
def brandt_solver(M: float, ecc: float) -> float:
    """
    brandt_solver(M: float, ecc: float) -> float
    """
    ...

def contour_solver(M: float, ecc: float) -> float:
    ...

def keplerian(t: list[float], P: float, K: float, ecc: float, w: float, M0: float, M0_epoch: float) -> list[float]:
    ...

def keplerian2(t: list[float], P: float, K: float, ecc: float, w: float, M0: float, M0_epoch: float) -> list[float]:
    ...

def murison_solver(M: numpy.typing.NDArray, ecc: float) -> list[float]:
    """
    murison_solver(M: ndarray[dtype=float64], ecc: float) -> list[float]
    """
    ...

@overload
def murison_solver(M: float, ecc: float) -> float:
    """
    murison_solver(M: float, ecc: float) -> float
    """
    ...

def nijenhuis_solver(M: numpy.typing.NDArray, ecc: float) -> list[float]:
    """
    nijenhuis_solver(M: ndarray[dtype=float64], ecc: float) -> list[float]
    """
    ...

@overload
def nijenhuis_solver(M: float, ecc: float) -> float:
    """
    nijenhuis_solver(M: float, ecc: float) -> float
    """
    ...

