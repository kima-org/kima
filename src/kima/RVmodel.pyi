from typing import Any, Optional, overload, Typing, Sequence
from enum import Enum
import kima.RVmodel

class RVConditionalPrior:
    """
    None
    """

    @property
    def Kprior(self) -> kima.distributions.Distribution:
        """
        Prior for the semi-amplitude(s)
        """
        ...
    @Kprior.setter
    def Kprior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the semi-amplitude(s)
        """
        ...
    
    @property
    def Pprior(self) -> kima.distributions.Distribution:
        """
        Prior for the orbital period(s)
        """
        ...
    @Pprior.setter
    def Pprior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the orbital period(s)
        """
        ...
    
    def __init__(self) -> None:
        ...
    
    @property
    def eprior(self) -> kima.distributions.Distribution:
        """
        Prior for the orbital eccentricity(ies)
        """
        ...
    @eprior.setter
    def eprior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the orbital eccentricity(ies)
        """
        ...
    
    @property
    def phiprior(self) -> kima.distributions.Distribution:
        """
        Prior for the mean anomaly(ies)
        """
        ...
    @phiprior.setter
    def phiprior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the mean anomaly(ies)
        """
        ...
    
    @property
    def wprior(self) -> kima.distributions.Distribution:
        """
        Prior for the argument(s) of periastron
        """
        ...
    @wprior.setter
    def wprior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the argument(s) of periastron
        """
        ...
    
class RVmodel:

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
    
    @property
    def KO_Kprior(self) -> list[kima.distributions.Distribution]:
        """
        Prior for KO semi-amplitude
        """
        ...
    @KO_Kprior.setter
    def KO_Kprior(self, arg: list[kima.distributions.Distribution], /) -> None:
        """
        Prior for KO semi-amplitude
        """
        ...
    
    @property
    def KO_Pprior(self) -> list[kima.distributions.Distribution]:
        """
        Prior for KO orbital period
        """
        ...
    @KO_Pprior.setter
    def KO_Pprior(self, arg: list[kima.distributions.Distribution], /) -> None:
        """
        Prior for KO orbital period
        """
        ...
    
    @property
    def KO_eprior(self) -> list[kima.distributions.Distribution]:
        """
        Prior for KO eccentricity
        """
        ...
    @KO_eprior.setter
    def KO_eprior(self, arg: list[kima.distributions.Distribution], /) -> None:
        """
        Prior for KO eccentricity
        """
        ...
    
    @property
    def KO_phiprior(self) -> list[kima.distributions.Distribution]:
        """
        Prior for KO mean anomaly(ies)
        """
        ...
    @KO_phiprior.setter
    def KO_phiprior(self, arg: list[kima.distributions.Distribution], /) -> None:
        """
        Prior for KO mean anomaly(ies)
        """
        ...
    
    @property
    def KO_wprior(self) -> list[kima.distributions.Distribution]:
        """
        Prior for KO argument of periastron
        """
        ...
    @KO_wprior.setter
    def KO_wprior(self, arg: list[kima.distributions.Distribution], /) -> None:
        """
        Prior for KO argument of periastron
        """
        ...
    
    @property
    def TR_Kprior(self) -> list[kima.distributions.Distribution]:
        """
        Prior for TR semi-amplitude
        """
        ...
    @TR_Kprior.setter
    def TR_Kprior(self, arg: list[kima.distributions.Distribution], /) -> None:
        """
        Prior for TR semi-amplitude
        """
        ...
    
    @property
    def TR_Pprior(self) -> list[kima.distributions.Distribution]:
        """
        Prior for TR orbital period
        """
        ...
    @TR_Pprior.setter
    def TR_Pprior(self, arg: list[kima.distributions.Distribution], /) -> None:
        """
        Prior for TR orbital period
        """
        ...
    
    @property
    def TR_Tcprior(self) -> list[kima.distributions.Distribution]:
        """
        Prior for TR mean anomaly(ies)
        """
        ...
    @TR_Tcprior.setter
    def TR_Tcprior(self, arg: list[kima.distributions.Distribution], /) -> None:
        """
        Prior for TR mean anomaly(ies)
        """
        ...
    
    @property
    def TR_eprior(self) -> list[kima.distributions.Distribution]:
        """
        Prior for TR eccentricity
        """
        ...
    @TR_eprior.setter
    def TR_eprior(self, arg: list[kima.distributions.Distribution], /) -> None:
        """
        Prior for TR eccentricity
        """
        ...
    
    @property
    def TR_wprior(self) -> list[kima.distributions.Distribution]:
        """
        Prior for TR argument of periastron
        """
        ...
    @TR_wprior.setter
    def TR_wprior(self, arg: list[kima.distributions.Distribution], /) -> None:
        """
        Prior for TR argument of periastron
        """
        ...
    
    def __init__(self, fix: bool, npmax: int, data: kima.Data.RVData) -> None:
        """
        Implements a sum-of-Keplerians model where the number of Keplerians can be free.
        
        Args:
        fix (bool):
        whether the number of Keplerians should be fixed
        npmax (int):
        maximum number of Keplerians
        data (RVData):
        the RV data
        """
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
    def data(self) -> kima.Data.RVData:
        """
        the data
        """
        ...
    
    @property
    def degree(self) -> int:
        """
        degree of the polynomial trend
        """
        ...
    @degree.setter
    def degree(self, arg: int, /) -> None:
        """
        degree of the polynomial trend
        """
        ...
    
    @property
    def enforce_stability(self) -> bool:
        """
        whether to enforce AMD-stability
        """
        ...
    @enforce_stability.setter
    def enforce_stability(self, arg: bool, /) -> None:
        """
        whether to enforce AMD-stability
        """
        ...
    
    @property
    def fix(self) -> bool:
        """
        whether the number of Keplerians is fixed
        """
        ...
    @fix.setter
    def fix(self, arg: bool, /) -> None:
        """
        whether the number of Keplerians is fixed
        """
        ...
    
    @property
    def indicator_correlations(self) -> bool:
        """
        include in the model linear correlations with indicators
        """
        ...
    @indicator_correlations.setter
    def indicator_correlations(self, arg: bool, /) -> None:
        """
        include in the model linear correlations with indicators
        """
        ...
    
    @property
    def individual_offset_prior(self) -> list[kima.distributions.Distribution]:
        """
        Common prior for the between-instrument offsets
        """
        ...
    @individual_offset_prior.setter
    def individual_offset_prior(self, arg: list[kima.distributions.Distribution], /) -> None:
        """
        Common prior for the between-instrument offsets
        """
        ...
    
    @property
    def known_object(self) -> bool:
        """
        whether the model includes (better) known extra Keplerian curve(s)
        """
        ...
    
    @property
    def n_known_object(self) -> int:
        """
        how many known objects
        """
        ...
    
    @property
    def n_transiting_planet(self) -> int:
        """
        how many transiting planets
        """
        ...
    
    @property
    def npmax(self) -> int:
        """
        maximum number of Keplerians
        """
        ...
    @npmax.setter
    def npmax(self, arg: int, /) -> None:
        """
        maximum number of Keplerians
        """
        ...
    
    @property
    def nu_prior(self) -> kima.distributions.Distribution:
        """
        Prior for the degrees of freedom of the Student-t likelihood
        """
        ...
    @nu_prior.setter
    def nu_prior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the degrees of freedom of the Student-t likelihood
        """
        ...
    
    @property
    def offsets_prior(self) -> kima.distributions.Distribution:
        """
        Common prior for the between-instrument offsets
        """
        ...
    @offsets_prior.setter
    def offsets_prior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Common prior for the between-instrument offsets
        """
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
    
    def set_known_object(self, arg: int, /) -> None:
        ...
    
    def set_transiting_planet(self, arg: int, /) -> None:
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
    def star_mass(self) -> float:
        """
        stellar mass [Msun]
        """
        ...
    @star_mass.setter
    def star_mass(self, arg: float, /) -> None:
        """
        stellar mass [Msun]
        """
        ...
    
    @property
    def stellar_jitter_prior(self) -> kima.distributions.Distribution:
        """
        Prior for the stellar jitter (common to all instruments)
        """
        ...
    @stellar_jitter_prior.setter
    def stellar_jitter_prior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the stellar jitter (common to all instruments)
        """
        ...
    
    @property
    def studentt(self) -> bool:
        """
        use a Student-t distribution for the likelihood (instead of Gaussian)
        """
        ...
    @studentt.setter
    def studentt(self, arg: bool, /) -> None:
        """
        use a Student-t distribution for the likelihood (instead of Gaussian)
        """
        ...
    
    @property
    def transiting_planet(self) -> bool:
        """
        whether the model includes transiting planet(s)
        """
        ...
    
    @property
    def trend(self) -> bool:
        """
        whether the model includes a polynomial trend
        """
        ...
    @trend.setter
    def trend(self, arg: bool, /) -> None:
        """
        whether the model includes a polynomial trend
        """
        ...
    
class TRANSITConditionalPrior:
    """
    None
    """

    @property
    def Pprior(self) -> kima.distributions.Distribution:
        """
        Prior for the orbital period(s)
        """
        ...
    @Pprior.setter
    def Pprior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the orbital period(s)
        """
        ...
    
    @property
    def RPprior(self) -> kima.distributions.Distribution:
        """
        Prior for the planet(s) radius
        """
        ...
    @RPprior.setter
    def RPprior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the planet(s) radius
        """
        ...
    
    def __init__(self) -> None:
        ...
    
    @property
    def aprior(self) -> kima.distributions.Distribution:
        """
        Prior for the planet(s) semi-major axis
        """
        ...
    @aprior.setter
    def aprior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the planet(s) semi-major axis
        """
        ...
    
    @property
    def eprior(self) -> kima.distributions.Distribution:
        """
        Prior for the orbital eccentricity(ies)
        """
        ...
    @eprior.setter
    def eprior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the orbital eccentricity(ies)
        """
        ...
    
    @property
    def incprior(self) -> kima.distributions.Distribution:
        """
        Prior for the inclinations(s)
        """
        ...
    @incprior.setter
    def incprior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the inclinations(s)
        """
        ...
    
    @property
    def t0prior(self) -> kima.distributions.Distribution:
        """
        Prior for the time(s) of inferior conjunction
        """
        ...
    @t0prior.setter
    def t0prior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the time(s) of inferior conjunction
        """
        ...
    
    @property
    def wprior(self) -> kima.distributions.Distribution:
        """
        Prior for the argument(s) of periastron
        """
        ...
    @wprior.setter
    def wprior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the argument(s) of periastron
        """
        ...
    
