from collections.abc import Sequence

import kima.Data
import kima.RVmodel
import kima.distributions


class BINARIESmodel:
    def __init__(self, fix: bool, npmax: int, data: kima.Data.RVdata) -> None:
        """
        Implements a sum-of-Keplerians model where the number of Keplerians can be free.
        This model assumes white, uncorrelated noise. This modules is tailored for the analysis
        of stellar binaries through the known object mode (without it this defaults to the RVmodel)
        The binary can have one set of RVs or two (one on each star). This model adds apsidal 
        precession as a free parameter and accounts for GR and Tidal effects on the radial velocities.

        Args:
            fix (bool, default=True): whether the number of Keplerians should be fixed
            npmax (int, default=0): maximum number of Keplerians
            data (RVdata): the RV data
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
    def index_from(self) -> int:
        """
        what indexing convention to use for the labelling of keplerians, defaults to 1.
        """

    @index_from.setter
    def index_from(self, arg: int, /) -> None: ...

    @property
    def known_object(self) -> bool:
        """
        whether to include (better) known extra Keplerian curve(s), should be true for BINARIESmodel
        """

    @known_object.setter
    def known_object(self, arg: bool, /) -> None: ...

    @property
    def n_known_object(self) -> int:
        """how many known objects, should be >=1 for BINARIESmodel"""

    @n_known_object.setter
    def n_known_object(self, arg: int, /) -> None: ...

    @property
    def star_mass(self) -> float:
        """primary mass [Msun]"""

    @star_mass.setter
    def star_mass(self, arg: float, /) -> None: ...

    @property
    def star_radius(self) -> float:
        """primary radius [Rsun]"""

    @star_radius.setter
    def star_radius(self, arg: float, /) -> None: ...

    @property
    def binary_mass(self) -> float:
        """secondary mass [Msun]"""

    @binary_mass.setter
    def binary_mass(self, arg: float, /) -> None: ...

    @property
    def binary_radius(self) -> float:
        """secondary radius [Rsun]"""

    @binary_radius.setter
    def binary_radius(self, arg: float, /) -> None: ...

    @property
    def enforce_stability(self) -> bool:
        """whether to enforce AMD-stability (Untested for binaries)"""

    @enforce_stability.setter
    def enforce_stability(self, arg: bool, /) -> None: ...

    @property
    def relativistic_correction(self) -> bool:
        """whether to perform the GR correction"""

    @relativistic_correction.setter
    def relativistic_correction(self, arg: bool, /) -> None: ...

    @property
    def tidal_correction(self) -> bool:
        """whether to perform the tidal correction"""

    @tidal_correction.setter
    def tidal_correction(self, arg: bool, /) -> None: ...

    @property
    def correction_K_precision(self) -> float:
        """
        To what precision K2 is calculated for the relativistic and tidal corrections, (default is 50 m/s and unless the orbit is very close to face-on this gives sub cm/s precision on the corrections), increasing this may speed up the sampling
        """

    @correction_K_precision.setter
    def correction_K_precision(self, arg: float, /) -> None: ...

    @property
    def double_lined(self) -> bool:
        """
        whether the data is for a double lined binary, and has two sets of RVs for each epoch
        """

    @double_lined.setter
    def double_lined(self, arg: bool, /) -> None: ...

    @property
    def eclipsing(self) -> bool:
        """
        whether binary is eclipsing and the inclination can be fixed to 90, defaults to true. If false, default prior is uniform in cosi
        """

    @eclipsing.setter
    def eclipsing(self, arg: bool, /) -> None: ...

    @property
    def use_binary_longitude(self) -> bool:
        """
        Whether to sample using the mean longitude rather than mean anomaly at epoch, for use when the binary is close to circular for better sampling
        """

    @use_binary_longitude.setter
    def use_binary_longitude(self, arg: bool, /) -> None: ...

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
    def nu_prior(self) -> kima.distributions.Distribution:
        """Prior for the degrees of freedom of the Student-t likelihood"""

    @nu_prior.setter
    def nu_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def KO_Pprior(self) -> list[kima.distributions.Distribution]:
        """Prior for Period of Binary orbit (+other known objects)"""

    @KO_Pprior.setter
    def KO_Pprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def KO_Kprior(self) -> list[kima.distributions.Distribution]:
        """Prior for Semi-amplitude of Binary orbit (+other known objects)"""

    @KO_Kprior.setter
    def KO_Kprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def KO_eprior(self) -> list[kima.distributions.Distribution]:
        """Prior for eccentricity of Binary orbit (+other known objects)"""

    @KO_eprior.setter
    def KO_eprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def KO_wprior(self) -> list[kima.distributions.Distribution]:
        """
        Prior for argument of periastron of Binary orbit (+other known objects)
        """

    @KO_wprior.setter
    def KO_wprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def KO_phiprior(self) -> list[kima.distributions.Distribution]:
        """
        Prior for phase of Binary orbit at the reference time (+other known objects)
        """

    @KO_phiprior.setter
    def KO_phiprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def KO_wdotprior(self) -> list[kima.distributions.Distribution]:
        """
        Prior for apsidal precession rate of Binary orbit (+other known objects)
        """

    @KO_wdotprior.setter
    def KO_wdotprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def KO_qprior(self) -> list[kima.distributions.Distribution]:
        """
        Prior for mass ratio of Binary (+other known objects), used if double lined
        """

    @KO_qprior.setter
    def KO_qprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def KO_cosiprior(self) -> list[kima.distributions.Distribution]:
        """
        Prior for cosine of the inclination of Binary orbit (+other known objects) to the line-of-sight
        """

    @KO_cosiprior.setter
    def KO_cosiprior(self, arg: Sequence[kima.distributions.Distribution], /) -> None: ...

    @property
    def conditional(self) -> kima.RVmodel.KeplerianConditionalPrior: ...

    @conditional.setter
    def conditional(self, arg: kima.RVmodel.KeplerianConditionalPrior, /) -> None: ...
