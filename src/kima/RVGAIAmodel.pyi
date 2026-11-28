from collections.abc import Sequence
from typing import overload

import kima.Data
import kima.distributions


class RVGAIAConditionalPrior:
    def __init__(self) -> None: ...

    @property
    def Pprior(self) -> kima.distributions.Distribution:
        """Prior for the orbital period(s)"""

    @Pprior.setter
    def Pprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def eprior(self) -> kima.distributions.Distribution:
        """Prior for the orbital eccentricity(ies)"""

    @eprior.setter
    def eprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def aprior(self) -> kima.distributions.Distribution:
        """Prior for the angular photocentre semi-major axis(es) (mas)"""

    @aprior.setter
    def aprior(self, arg: kima.distributions.Distribution, /) -> None: ...

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
    def Wprior(self) -> kima.distributions.Distribution:
        """Prior for the longitude(s) of ascending node"""

    @Wprior.setter
    def Wprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def cosiprior(self) -> kima.distributions.Distribution:
        """Prior for cosine(s) of the orbital inclination"""

    @cosiprior.setter
    def cosiprior(self, arg: kima.distributions.Distribution, /) -> None: ...

class RVGAIAmodel:
    @overload
    def __init__(self, fix: bool, npmax: int, RVdata: kima.Data.RVdata, GAIAdata: kima.Data.GAIAdata) -> None: ...

    @overload
    def __init__(self, fix: bool, npmax: int, GAIA_data: kima.Data.GAIAdata, RV_data: kima.Data.RVdata) -> None:
        """
        Combined analysis of Gaia epoch astrometry and radial velocity timeseries. 
        Implements a sum-of-Keplerians model where the number of Keplerians can be free.
        This model assumes white, uncorrelated noise. Both the central star and 
        potential planets are to be given Mass priors in Solar-Mass.

        Args:
            fix (bool, default=True):
                whether the number of Keplerians should be fixed
            npmax (int, default=0):
                maximum number of Keplerians
            RVdata (RVdata):
                the radial velocity data
            GAIAdata (GAIAdata):
                the astrometric data
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
    def RVdata(self) -> kima.Data.RVdata:
        """RV dataset"""

    @property
    def GAIAdata(self) -> kima.Data.GAIAdata:
        """Gaia epoch astrometry dataset"""

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
    def index_from(self) -> int:
        """
        what indexing convention to use for the labelling of keplerians, defaults to 1.
        """

    @index_from.setter
    def index_from(self, arg: int, /) -> None: ...

    @property
    def RA(self) -> float:
        """Right Ascension of the target star (degrees)"""

    @RA.setter
    def RA(self, arg: float, /) -> None: ...

    @property
    def DEC(self) -> float:
        """Declination of the target star (degrees)"""

    @DEC.setter
    def DEC(self, arg: float, /) -> None: ...

    @property
    def star_mass(self) -> float:
        """the mass of the central star (Msun)"""

    @star_mass.setter
    def star_mass(self, arg: float, /) -> None: ...

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
    def indicator_correlations(self) -> bool:
        """include in the model linear correlations with indicators"""

    @indicator_correlations.setter
    def indicator_correlations(self, arg: bool, /) -> None: ...

    def set_scan_dep_signal(self, arg: int, /) -> None:
        """
        set whether the model includes a model for potential scan-angle dependent signals
        """

    @property
    def scan_dep_signal(self) -> bool:
        """
        whether the model includes a model for potential scan-angle dependent signals that could bias towards certain frequencies
        """

    @property
    def n_scan_dep_components(self) -> int:
        """how many components of scan-angle harmonics are included"""

    def set_baseline_model(self, arg: int, /) -> None:
        """
        set the number of parameters for the baseline astrometric solution, either 5 (the default astrometric solution), 7, or 9 (which include acceleration and jerk terms)
        """

    @property
    def n_baseline_params(self) -> int:
        """how many baseline astrometric parameters are included the model"""

    def __getstate__(self) -> tuple[bool, int, kima.Data.RVdata, kima.Data.GAIAdata]: ...

    def __setstate__(self, arg: tuple[bool, int, kima.Data.RVdata, kima.Data.GAIAdata], /) -> None: ...

    @property
    def Cprior(self) -> kima.distributions.Distribution:
        """Prior for the systemic velocity"""

    @Cprior.setter
    def Cprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def J_GAIA_prior(self) -> kima.distributions.Distribution:
        """Prior for the extra white noise (jitter) for GAIA data"""

    @J_GAIA_prior.setter
    def J_GAIA_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def J_RV_prior(self) -> kima.distributions.Distribution:
        """Prior for the extra white noise (jitter) for RV data"""

    @J_RV_prior.setter
    def J_RV_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def nu_GAIA_prior(self) -> kima.distributions.Distribution:
        """
        Prior for the degrees of freedom of the Student-t likelihood for GAIA data
        """

    @nu_GAIA_prior.setter
    def nu_GAIA_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def nu_RV_prior(self) -> kima.distributions.Distribution:
        """
        Prior for the degrees of freedom of the Student-t likelihood for RV data
        """

    @nu_RV_prior.setter
    def nu_RV_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

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
    def Ak_prior(self) -> list[kima.distributions.Distribution]:
        """Prior for the amplitudes of scan-angle dependent signals"""

    @Ak_prior.setter
    def Ak_prior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def thetak_prior(self) -> list[kima.distributions.Distribution]:
        """Prior for the phase of scan-angle dependent signals"""

    @thetak_prior.setter
    def thetak_prior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def da_prior(self) -> kima.distributions.Distribution:
        """Prior for the offset in right-ascension (mas)"""

    @da_prior.setter
    def da_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def dd_prior(self) -> kima.distributions.Distribution:
        """Prior for the the offset in declination (mas)"""

    @dd_prior.setter
    def dd_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def mua_prior(self) -> kima.distributions.Distribution:
        """Prior for the proper-motion in right-ascension (mas/yr)"""

    @mua_prior.setter
    def mua_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def mud_prior(self) -> kima.distributions.Distribution:
        """Prior for the proper-motion in declination (mas/yr)"""

    @mud_prior.setter
    def mud_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def parallax_prior(self) -> kima.distributions.Distribution:
        """Prior for the parallax"""

    @parallax_prior.setter
    def parallax_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def accela_prior(self) -> kima.distributions.Distribution:
        """Prior for the proper-acceleration in right-ascension (mas/yr^2)"""

    @accela_prior.setter
    def accela_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def acceld_prior(self) -> kima.distributions.Distribution:
        """Prior for the proper-acceleration in declination (mas/yr^2)"""

    @acceld_prior.setter
    def acceld_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def jerka_prior(self) -> kima.distributions.Distribution:
        """Prior for the proper-jerk in right-ascension (mas/yr^3)"""

    @jerka_prior.setter
    def jerka_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def jerkd_prior(self) -> kima.distributions.Distribution:
        """Prior for the proper-jerk in declination (mas/yr^3)"""

    @jerkd_prior.setter
    def jerkd_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def KO_Pprior(self) -> list[kima.distributions.Distribution]:
        """Prior for KO orbital period(s) (days)"""

    @KO_Pprior.setter
    def KO_Pprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def KO_aprior(self) -> list[kima.distributions.Distribution]:
        """Prior for KO photocentre semi-major axis(es) (mas)"""

    @KO_aprior.setter
    def KO_aprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def KO_eprior(self) -> list[kima.distributions.Distribution]:
        """Prior for KO eccentricity(ies)"""

    @KO_eprior.setter
    def KO_eprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def KO_wprior(self) -> list[kima.distributions.Distribution]:
        """Prior for KO argument(s) of periastron"""

    @KO_wprior.setter
    def KO_wprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def KO_phiprior(self) -> list[kima.distributions.Distribution]:
        """Prior for KO mean anomaly(ies)"""

    @KO_phiprior.setter
    def KO_phiprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def KO_cosiprior(self) -> list[kima.distributions.Distribution]:
        """Prior for cosine of KO inclination(s)"""

    @KO_cosiprior.setter
    def KO_cosiprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def KO_Wprior(self) -> list[kima.distributions.Distribution]:
        """Prior for KO longitude(s) of ascending node"""

    @KO_Wprior.setter
    def KO_Wprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def TR_Pprior(self) -> list[kima.distributions.Distribution]:
        """Prior for TR orbital period(s)"""

    @TR_Pprior.setter
    def TR_Pprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def TR_aprior(self) -> list[kima.distributions.Distribution]:
        """Prior for TR photocentre semi-major axis(es) (mas)"""

    @TR_aprior.setter
    def TR_aprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def TR_eprior(self) -> list[kima.distributions.Distribution]:
        """Prior for TR eccentricity(ies)"""

    @TR_eprior.setter
    def TR_eprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def TR_wprior(self) -> list[kima.distributions.Distribution]:
        """Prior for TR argument(s) of periastron"""

    @TR_wprior.setter
    def TR_wprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def TR_Tcprior(self) -> list[kima.distributions.Distribution]:
        """Prior for TR mean anomaly(ies)"""

    @TR_Tcprior.setter
    def TR_Tcprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def TR_cosiprior(self) -> list[kima.distributions.Distribution]:
        """Prior for cosine of TR inclination(s)"""

    @TR_cosiprior.setter
    def TR_cosiprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def TR_Wprior(self) -> list[kima.distributions.Distribution]:
        """Prior for TR longitude(s) of ascending node"""

    @TR_Wprior.setter
    def TR_Wprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def conditional(self) -> RVGAIAConditionalPrior: ...

    @conditional.setter
    def conditional(self, arg: RVGAIAConditionalPrior, /) -> None: ...
