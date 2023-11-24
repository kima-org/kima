from typing import Any, Optional, overload, Typing, Sequence
from enum import Enum
import kima.SPLEAFmodel

class SPLEAFmodel:
    """
    None
    """

    @property
    def Cprior(self) -> kima.distributions.Distribution:
        """
        Prior for the systemic velocity
        """
        ...
    @Cprior.setter
    def Cprior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the systemic velocity
        """
        ...
    
    @property
    def Jprior(self) -> kima.distributions.Distribution:
        """
        Prior for the extra white noise (jitter)
        """
        ...
    @Jprior.setter
    def Jprior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the extra white noise (jitter)
        """
        ...
    
    def __init__(self, fix: bool, npmax: int, data: kima.Data.RVData, multi_series: bool) -> None:
        ...
    
    @property
    def conditional(self) -> kima.RVmodel.RVConditionalPrior:
        ...
    @conditional.setter
    def conditional(self, arg: kima.RVmodel.RVConditionalPrior, /) -> None:
        ...
    
    @property
    def cubic_prior(self) -> kima.distributions.Distribution:
        """
        Prior for the cubic coefficient of the trend
        """
        ...
    @cubic_prior.setter
    def cubic_prior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the cubic coefficient of the trend
        """
        ...
    
    @property
    def degree(self) -> float:
        ...
    @degree.setter
    def degree(self, arg: float, /) -> None:
        ...
    
    @property
    def kernel(self) -> Term:
        ...
    @kernel.setter
    def kernel(self, arg: Term, /) -> None:
        ...
    
    @property
    def quadr_prior(self) -> kima.distributions.Distribution:
        """
        Prior for the quadratic coefficient of the trend
        """
        ...
    @quadr_prior.setter
    def quadr_prior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the quadratic coefficient of the trend
        """
        ...
    
    @property
    def slope_prior(self) -> kima.distributions.Distribution:
        """
        Prior for the slope
        """
        ...
    @slope_prior.setter
    def slope_prior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the slope
        """
        ...
    
    @property
    def trend(self) -> bool:
        ...
    @trend.setter
    def trend(self, arg: bool, /) -> None:
        ...
    
