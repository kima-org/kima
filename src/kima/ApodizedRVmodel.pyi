from collections.abc import Sequence
import enum

import kima.Data
import kima.RVmodel
import kima.distributions


class ApodType(enum.Enum):
    apod_gaussian = 0

    apod_boxcar = 1

    apod_plateau = 2

apod_gaussian: ApodType = ApodType.apod_gaussian

apod_boxcar: ApodType = ApodType.apod_boxcar

apod_plateau: ApodType = ApodType.apod_plateau

class ApodizedRVmodel:
    def __init__(self, fix: bool, npmax: int, data: kima.Data.RVdata) -> None:
        """
        Implements a sum-of-apodized-Keplerians model where the number of components can be free.

        Args:
            fix (bool):
                whether the number of apodized Keplerians should be fixed
            npmax (int):
                maximum number of apodized Keplerians
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

    @property
    def studentt(self) -> bool:
        """use a Student-t distribution for the likelihood (instead of Gaussian)"""

    @studentt.setter
    def studentt(self, arg: bool, /) -> None: ...

    @property
    def apodization(self) -> ApodType:
        """type of apodization"""

    @apodization.setter
    def apodization(self, arg: ApodType, /) -> None: ...

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
    def remove_label_switching_degeneracy(self) -> bool:
        """docs"""

    @remove_label_switching_degeneracy.setter
    def remove_label_switching_degeneracy(self, arg: bool, /) -> None: ...

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

    def set_loguniform_prior_Np(self) -> None: ...

    @property
    def conditional(self) -> kima.RVmodel.ApodizedKeplerianConditionalPrior: ...

    @conditional.setter
    def conditional(self, arg: kima.RVmodel.ApodizedKeplerianConditionalPrior, /) -> None: ...
