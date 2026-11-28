from collections.abc import Sequence

import kima.Data
import kima.distributions


class KeplerianConditionalPrior:
    def __init__(self) -> None: ...

    @property
    def Pprior(self) -> kima.distributions.Distribution:
        """Prior for the orbital period(s)"""

    @Pprior.setter
    def Pprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def Kprior(self) -> kima.distributions.Distribution:
        """Prior for the semi-amplitude(s)"""

    @Kprior.setter
    def Kprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def eprior(self) -> kima.distributions.Distribution:
        """Prior for the orbital eccentricity(ies)"""

    @eprior.setter
    def eprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def wprior(self) -> kima.distributions.Distribution:
        """Prior for the argument(s) of periastron"""

    @wprior.setter
    def wprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def phiprior(self) -> kima.distributions.Distribution:
        """Prior for the mean anomaly(ies)"""

    @phiprior.setter
    def phiprior(self, arg: kima.distributions.Distribution, /) -> None: ...

class ApodizedKeplerianConditionalPrior:
    def __init__(self) -> None: ...

    @property
    def Pprior(self) -> kima.distributions.Distribution:
        """Prior for the orbital period(s)"""

    @Pprior.setter
    def Pprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def Kprior(self) -> kima.distributions.Distribution:
        """Prior for the semi-amplitude(s)"""

    @Kprior.setter
    def Kprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def eprior(self) -> kima.distributions.Distribution:
        """Prior for the orbital eccentricity(ies)"""

    @eprior.setter
    def eprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def wprior(self) -> kima.distributions.Distribution:
        """Prior for the argument(s) of periastron"""

    @wprior.setter
    def wprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def phiprior(self) -> kima.distributions.Distribution:
        """Prior for the mean anomaly(ies)"""

    @phiprior.setter
    def phiprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def tauprior(self) -> kima.distributions.Distribution:
        """Prior for the apodization width(s) τ"""

    @tauprior.setter
    def tauprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def t0prior(self) -> kima.distributions.Distribution:
        """Prior for the center of the apodizing window(s) t0"""

    @t0prior.setter
    def t0prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def sprior(self) -> kima.distributions.Distribution:
        """Prior for the shape(s) of the apodizing function (if plateau)"""

    @sprior.setter
    def sprior(self, arg: kima.distributions.Distribution, /) -> None: ...

class RVmodel:
    def __init__(self, fix: bool, npmax: int, data: kima.Data.RVdata) -> None:
        """
        Implements a sum-of-Keplerians model where the number of Keplerians can be free.

        Args:
            fix (bool):
                whether the number of Keplerians should be fixed
            npmax (int):
                maximum number of Keplerians
            data (RVdata):
                the RV data
        """

    def get_log_likelihood(self, arg: Sequence[float], /) -> float: ...

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
        """the RV dataset"""

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

    @property
    def studentt(self) -> bool:
        """use a Student-t distribution for the likelihood (instead of Gaussian)"""

    @studentt.setter
    def studentt(self, arg: bool, /) -> None: ...

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

    def set_apodized_keplerians(self, arg: int, /) -> None: ...

    @property
    def apodized_keplerians(self) -> bool:
        """whether the model includes apodized Keplerian(s)"""

    @property
    def n_apodized_keplerians(self) -> int:
        """how many apodized Keplerians"""

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
    def indicator_correlations(self) -> bool:
        """include in the model linear correlations with indicators"""

    @indicator_correlations.setter
    def indicator_correlations(self, arg: bool, /) -> None: ...

    @property
    def jitter_propto_indicator(self) -> bool:
        """docs"""

    @jitter_propto_indicator.setter
    def jitter_propto_indicator(self, arg: bool, /) -> None: ...

    @property
    def jitter_propto_indicator_index(self) -> int:
        """docs"""

    @jitter_propto_indicator_index.setter
    def jitter_propto_indicator_index(self, arg: int, /) -> None: ...

    @property
    def optimize_equal_times(self) -> bool:
        """docs"""

    @optimize_equal_times.setter
    def optimize_equal_times(self, arg: bool, /) -> None: ...

    @property
    def marginalise_C(self) -> bool:
        """Analytically marginalise the systemic velocity"""

    @marginalise_C.setter
    def marginalise_C(self, arg: bool, /) -> None: ...

    def __getstate__(self) -> tuple[bool, int, kima.Data.RVdata]: ...

    def __setstate__(self, arg: tuple[bool, int, kima.Data.RVdata], /) -> None: ...

    @property
    def Cprior(self) -> kima.distributions.Distribution:
        """Prior for the systemic velocity"""

    @Cprior.setter
    def Cprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def Jprior(self) -> kima.distributions.Distribution:
        """Prior for the extra white noise (jitter)"""

    @Jprior.setter
    def Jprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def stellar_jitter_prior(self) -> kima.distributions.Distribution:
        """Prior for the stellar jitter (common to all instruments)"""

    @stellar_jitter_prior.setter
    def stellar_jitter_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

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
    def offsets_prior(self) -> kima.distributions.Distribution:
        """Common prior for the between-instrument offsets"""

    @offsets_prior.setter
    def offsets_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def individual_offset_prior(self) -> list[kima.distributions.Distribution]:
        """Common prior for the between-instrument offsets"""

    @individual_offset_prior.setter
    def individual_offset_prior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def beta_prior(self) -> kima.distributions.Distribution:
        """(Common) prior for the activity indicator coefficients"""

    @beta_prior.setter
    def beta_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def nu_prior(self) -> kima.distributions.Distribution:
        """Prior for the degrees of freedom of the Student-t likelihood"""

    @nu_prior.setter
    def nu_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

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
    def AK_Pprior(self) -> list[kima.distributions.Distribution]:
        """Prior for AK orbital period"""

    @AK_Pprior.setter
    def AK_Pprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def AK_Kprior(self) -> list[kima.distributions.Distribution]:
        """Prior for AK semi-amplitude"""

    @AK_Kprior.setter
    def AK_Kprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def AK_eprior(self) -> list[kima.distributions.Distribution]:
        """Prior for AK eccentricity"""

    @AK_eprior.setter
    def AK_eprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def AK_wprior(self) -> list[kima.distributions.Distribution]:
        """Prior for AK argument of periastron"""

    @AK_wprior.setter
    def AK_wprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def AK_phiprior(self) -> list[kima.distributions.Distribution]:
        """Prior for AK mean anomaly(ies)"""

    @AK_phiprior.setter
    def AK_phiprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def AK_tauprior(self) -> list[kima.distributions.Distribution]:
        """Prior for AK apodization widths τ (days)"""

    @AK_tauprior.setter
    def AK_tauprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def AK_t0prior(self) -> list[kima.distributions.Distribution]:
        """Prior for AK center of apodizing windows (days)"""

    @AK_t0prior.setter
    def AK_t0prior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    def set_loguniform_prior_Np(self) -> None: ...

    @property
    def conditional(self) -> KeplerianConditionalPrior: ...

    @conditional.setter
    def conditional(self, arg: KeplerianConditionalPrior, /) -> None: ...
