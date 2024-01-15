from typing import Any, Optional, overload, Typing, Sequence
from enum import Enum
import kima.RVFWHMmodel

class RVFWHMmodel:
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
    
    def __init__(self, fix: bool, npmax: int, data: kima.Data.RVData) -> None:
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
    def eta1_fwhm_prior(self) -> kima.distributions.Distribution:
        """
        Prior for the GP 'amplitude' on the FWHM
        """
        ...
    @eta1_fwhm_prior.setter
    def eta1_fwhm_prior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the GP 'amplitude' on the FWHM
        """
        ...
    
    @property
    def eta1_prior(self) -> kima.distributions.Distribution:
        """
        Prior for the GP 'amplitude' on the RVs
        """
        ...
    @eta1_prior.setter
    def eta1_prior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for the GP 'amplitude' on the RVs
        """
        ...
    
    @property
    def eta2_fwhm_prior(self) -> kima.distributions.Distribution:
        """
        Prior for η2, the GP correlation timescale, on the FWHM
        """
        ...
    @eta2_fwhm_prior.setter
    def eta2_fwhm_prior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for η2, the GP correlation timescale, on the FWHM
        """
        ...
    
    @property
    def eta2_prior(self) -> kima.distributions.Distribution:
        """
        Prior for η2, the GP correlation timescale, on the RVs
        """
        ...
    @eta2_prior.setter
    def eta2_prior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for η2, the GP correlation timescale, on the RVs
        """
        ...
    
    @property
    def eta3_fwhm_prior(self) -> kima.distributions.Distribution:
        """
        Prior for η3, the GP period, on the FWHM
        """
        ...
    @eta3_fwhm_prior.setter
    def eta3_fwhm_prior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for η3, the GP period, on the FWHM
        """
        ...
    
    @property
    def eta3_prior(self) -> kima.distributions.Distribution:
        """
        Prior for η3, the GP period, on the RVs
        """
        ...
    @eta3_prior.setter
    def eta3_prior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for η3, the GP period, on the RVs
        """
        ...
    
    @property
    def eta4_fwhm_prior(self) -> kima.distributions.Distribution:
        """
        Prior for η4, the recurrence timescale or (inverse) harmonic complexity, on the FWHM
        """
        ...
    @eta4_fwhm_prior.setter
    def eta4_fwhm_prior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for η4, the recurrence timescale or (inverse) harmonic complexity, on the FWHM
        """
        ...
    
    @property
    def eta4_prior(self) -> kima.distributions.Distribution:
        """
        Prior for η4, the recurrence timescale or (inverse) harmonic complexity, on the RVs
        """
        ...
    @eta4_prior.setter
    def eta4_prior(self, arg: kima.distributions.Distribution, /) -> None:
        """
        Prior for η4, the recurrence timescale or (inverse) harmonic complexity, on the RVs
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
    
    @property
    def share_eta2(self) -> bool:
        """
        whether the η2 parameter is shared between RVs and FWHM
        """
        ...
    @share_eta2.setter
    def share_eta2(self, arg: bool, /) -> None:
        """
        whether the η2 parameter is shared between RVs and FWHM
        """
        ...
    
    @property
    def share_eta3(self) -> bool:
        """
        whether the η3 parameter is shared between RVs and FWHM
        """
        ...
    @share_eta3.setter
    def share_eta3(self, arg: bool, /) -> None:
        """
        whether the η3 parameter is shared between RVs and FWHM
        """
        ...
    
    @property
    def share_eta4(self) -> bool:
        """
        whether the η4 parameter is shared between RVs and FWHM
        """
        ...
    @share_eta4.setter
    def share_eta4(self, arg: bool, /) -> None:
        """
        whether the η4 parameter is shared between RVs and FWHM
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
    
