from collections.abc import Sequence

import kima.Data
import kima.distributions


class GAIAConditionalPrior:
    def __init__(self) -> None: ...

    @property
    def thiele_innes(self) -> bool:
        """use a Student-t distribution for the likelihood (instead of Gaussian)"""

    @thiele_innes.setter
    def thiele_innes(self, arg: bool, /) -> None: ...

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
        """Prior for the photocentre semi-major-axis(es) (mas)"""

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

    @property
    def Aprior(self) -> kima.distributions.Distribution:
        """Prior thiele_innes parameter(s) A"""

    @Aprior.setter
    def Aprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def Bprior(self) -> kima.distributions.Distribution:
        """Prior thiele_innes parameter(s) B"""

    @Bprior.setter
    def Bprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def Fprior(self) -> kima.distributions.Distribution:
        """Prior thiele_innes parameter(s) F"""

    @Fprior.setter
    def Fprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def Gprior(self) -> kima.distributions.Distribution:
        """Prior thiele_innes parameter(s) G"""

    @Gprior.setter
    def Gprior(self, arg: kima.distributions.Distribution, /) -> None: ...

class GAIAmodel:
    def __init__(self, fix: bool, npmax: int, data: kima.Data.GAIAdata) -> None:
        """
        Analysis of Gaia epoch astrometry. Implements a sum-of-Keplerians model where the number of Keplerians can be free.
        This model assumes white, uncorrelated noise. Known objects are given priors for geometric elements, free planet search
        has the choice of geometric or Thiele-Innes elements. An option to fit for a scan-angle dependent signal is included.

        Args:
            fix (bool, default=True):
                whether the number of Keplerians should be fixed
            npmax (int, default=0):
                maximum number of Keplerians
            data (GAIAdata):
                the astrometric data
        """

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
    def data(self) -> kima.Data.GAIAdata:
        """the data"""

    @property
    def studentt(self) -> bool:
        """use a Student-t distribution for the likelihood (instead of Gaussian)"""

    @studentt.setter
    def studentt(self, arg: bool, /) -> None: ...

    @property
    def thiele_innes(self) -> bool:
        """use the thiele-innes coefficients rather than geometric"""

    @thiele_innes.setter
    def thiele_innes(self, arg: bool, /) -> None: ...

    @property
    def star_mass(self) -> float:
        """the mass of the central star (Msun)"""

    @star_mass.setter
    def star_mass(self, arg: float, /) -> None: ...

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

    def set_known_object(self, arg: int, /) -> None:
        """set how many known objects to include"""

    @property
    def known_object(self) -> bool:
        """whether the model includes (better) known extra Keplerian curve(s)"""

    @property
    def n_known_object(self) -> int:
        """how many known objects"""

    @property
    def Jprior(self) -> kima.distributions.Distribution:
        """Prior for the extra white noise (jitter)"""

    @Jprior.setter
    def Jprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def nu_prior(self) -> kima.distributions.Distribution:
        """Prior for the degrees of freedom of the Student-t likelihood"""

    @nu_prior.setter
    def nu_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

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
        """Prior for KO orbital period(s)"""

    @KO_Pprior.setter
    def KO_Pprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def KO_aprior(self) -> list[kima.distributions.Distribution]:
        """Prior for KO photocentre semi-major-axis(es)"""

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
    def conditional(self) -> GAIAConditionalPrior: ...

    @conditional.setter
    def conditional(self, arg: GAIAConditionalPrior, /) -> None: ...
