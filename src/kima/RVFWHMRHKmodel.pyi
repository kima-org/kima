from collections.abc import Sequence

import kima.Data
import kima.RVmodel
import kima.distributions


class RVFWHMRHKmodel:
    def __init__(self, fix: bool, npmax: int, data: kima.Data.RVdata) -> None:
        """
        Implements a joint model for RVs, FWHM, and R'HK with a GP component for activity signals.

        Args:
            fix (bool):
                whether the number of Keplerians should be fixed
            npmax (int):
                maximum number of Keplerians
            data (RVdata):
                the RV data
        """

    @property
    def directory(self) -> str:
        """directory where the model ran"""

    @directory.setter
    def directory(self, arg: str, /) -> None: ...

    @property
    def fix(self) -> bool:
        """whether the number of Keplerians is fixed"""

    @fix.setter
    def fix(self, arg: bool, /) -> None: ...

    @property
    def npmax(self) -> int:
        """maximum number of Keplerians"""

    @npmax.setter
    def npmax(self, arg: int, /) -> None: ...

    @property
    def data(self) -> kima.Data.RVdata:
        """the data"""

    @property
    def trend(self) -> bool:
        """whether the model includes a polynomial trend"""

    @trend.setter
    def trend(self, arg: bool, /) -> None: ...

    @property
    def degree(self) -> int:
        """degree of the polynomial trend"""

    @degree.setter
    def degree(self, arg: int, /) -> None: ...

    def set_known_object(self, arg: int, /) -> None: ...

    @property
    def known_object(self) -> bool:
        """whether the model includes (better) known extra Keplerian curve(s)"""

    @property
    def n_known_object(self) -> int:
        """how many known objects"""

    def set_transiting_planet(self, arg: int, /) -> None: ...

    @property
    def transiting_planet(self) -> bool:
        """whether the model includes transiting planet(s)"""

    @property
    def n_transiting_planet(self) -> int:
        """how many transiting planets"""

    @property
    def star_mass(self) -> float:
        """stellar mass [Msun]"""

    @star_mass.setter
    def star_mass(self, arg: float, /) -> None: ...

    @property
    def enforce_stability(self) -> bool:
        """whether to enforce AMD-stability"""

    @enforce_stability.setter
    def enforce_stability(self, arg: bool, /) -> None: ...

    @property
    def share_eta2(self) -> bool:
        """whether the η2 parameter is shared between RVs and FWHM"""

    @share_eta2.setter
    def share_eta2(self, arg: bool, /) -> None: ...

    @property
    def share_eta3(self) -> bool:
        """whether the η3 parameter is shared between RVs and FWHM"""

    @share_eta3.setter
    def share_eta3(self, arg: bool, /) -> None: ...

    @property
    def share_eta4(self) -> bool:
        """whether the η4 parameter is shared between RVs and FWHM"""

    @share_eta4.setter
    def share_eta4(self, arg: bool, /) -> None: ...

    @property
    def magnetic_cycle_kernel(self) -> bool:
        """whether to consider a (periodic) GP kernel for a magnetic cycle"""

    @magnetic_cycle_kernel.setter
    def magnetic_cycle_kernel(self, arg: bool, /) -> None: ...

    @property
    def Cprior(self) -> kima.distributions.Distribution:
        """Prior for the systemic velocity"""

    @Cprior.setter
    def Cprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def Cfwhm_prior(self) -> kima.distributions.Distribution:
        """Prior for the 'systemic' FWHM"""

    @Cfwhm_prior.setter
    def Cfwhm_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def Crhk_prior(self) -> kima.distributions.Distribution:
        """Prior for the 'systemic' R'hk"""

    @Crhk_prior.setter
    def Crhk_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def Jprior(self) -> kima.distributions.Distribution:
        """Prior for the extra white noise (jitter)"""

    @Jprior.setter
    def Jprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def Jfwhm_prior(self) -> kima.distributions.Distribution:
        """Prior for the extra white noise (jitter) in the FWHM"""

    @Jfwhm_prior.setter
    def Jfwhm_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def Jrhk_prior(self) -> kima.distributions.Distribution:
        """Prior for the extra white noise (jitter) in the R'hk"""

    @Jrhk_prior.setter
    def Jrhk_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def slope_prior(self) -> kima.distributions.Distribution:
        """Prior for the slope"""

    @slope_prior.setter
    def slope_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def quadr_prior(self) -> kima.distributions.Distribution:
        """Prior for the quadratic coefficient of the trend"""

    @quadr_prior.setter
    def quadr_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def cubic_prior(self) -> kima.distributions.Distribution:
        """Prior for the cubic coefficient of the trend"""

    @cubic_prior.setter
    def cubic_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def eta1_prior(self) -> kima.distributions.Distribution:
        """Prior for the GP 'amplitude' on the RVs"""

    @eta1_prior.setter
    def eta1_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def eta1_fwhm_prior(self) -> kima.distributions.Distribution:
        """Prior for the GP 'amplitude' on the FWHM"""

    @eta1_fwhm_prior.setter
    def eta1_fwhm_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def eta1_rhk_prior(self) -> kima.distributions.Distribution:
        """Prior for the GP 'amplitude' on the R'HK"""

    @eta1_rhk_prior.setter
    def eta1_rhk_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def eta2_prior(self) -> kima.distributions.Distribution:
        """Prior for η2, the GP correlation timescale, on the RVs"""

    @eta2_prior.setter
    def eta2_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def eta2_fwhm_prior(self) -> kima.distributions.Distribution:
        """Prior for η2, the GP correlation timescale, on the FWHM"""

    @eta2_fwhm_prior.setter
    def eta2_fwhm_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def eta2_rhk_prior(self) -> kima.distributions.Distribution:
        """Prior for η2, the GP correlation timescale, on the R'HK"""

    @eta2_rhk_prior.setter
    def eta2_rhk_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def eta3_prior(self) -> kima.distributions.Distribution:
        """Prior for η3, the GP period, on the RVs"""

    @eta3_prior.setter
    def eta3_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def eta3_fwhm_prior(self) -> kima.distributions.Distribution:
        """Prior for η3, the GP period, on the FWHM"""

    @eta3_fwhm_prior.setter
    def eta3_fwhm_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def eta3_rhk_prior(self) -> kima.distributions.Distribution:
        """Prior for η3, the GP period, on the R'HK"""

    @eta3_rhk_prior.setter
    def eta3_rhk_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def eta4_prior(self) -> kima.distributions.Distribution:
        """
        Prior for η4, the recurrence timescale or (inverse) harmonic complexity, on the RVs
        """

    @eta4_prior.setter
    def eta4_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def eta4_fwhm_prior(self) -> kima.distributions.Distribution:
        """
        Prior for η4, the recurrence timescale or (inverse) harmonic complexity, on the FWHM
        """

    @eta4_fwhm_prior.setter
    def eta4_fwhm_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def eta4_rhk_prior(self) -> kima.distributions.Distribution:
        """
        Prior for η4, the recurrence timescale or (inverse) harmonic complexity, on the R'HK
        """

    @eta4_rhk_prior.setter
    def eta4_rhk_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def KO_Pprior(self) -> list[kima.distributions.Distribution]:
        """Prior for KO orbital period"""

    @KO_Pprior.setter
    def KO_Pprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def KO_Kprior(self) -> list[kima.distributions.Distribution]:
        """Prior for KO semi-amplitude"""

    @KO_Kprior.setter
    def KO_Kprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def KO_eprior(self) -> list[kima.distributions.Distribution]:
        """Prior for KO eccentricity"""

    @KO_eprior.setter
    def KO_eprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def KO_wprior(self) -> list[kima.distributions.Distribution]:
        """Prior for KO argument of periastron"""

    @KO_wprior.setter
    def KO_wprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def KO_phiprior(self) -> list[kima.distributions.Distribution]:
        """Prior for KO mean anomaly(ies)"""

    @KO_phiprior.setter
    def KO_phiprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def TR_Pprior(self) -> list[kima.distributions.Distribution]:
        """Prior for TR orbital period"""

    @TR_Pprior.setter
    def TR_Pprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def TR_Kprior(self) -> list[kima.distributions.Distribution]:
        """Prior for TR semi-amplitude"""

    @TR_Kprior.setter
    def TR_Kprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def TR_eprior(self) -> list[kima.distributions.Distribution]:
        """Prior for TR eccentricity"""

    @TR_eprior.setter
    def TR_eprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def TR_wprior(self) -> list[kima.distributions.Distribution]:
        """Prior for TR argument of periastron"""

    @TR_wprior.setter
    def TR_wprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def TR_Tcprior(self) -> list[kima.distributions.Distribution]:
        """Prior for TR mean anomaly(ies)"""

    @TR_Tcprior.setter
    def TR_Tcprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def conditional(self) -> kima.RVmodel.KeplerianConditionalPrior: ...

    @conditional.setter
    def conditional(self, arg: kima.RVmodel.KeplerianConditionalPrior, /) -> None: ...
