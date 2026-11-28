import kima.Data
import kima.distributions


class TRANSITConditionalPrior:
    def __init__(self) -> None: ...

    @property
    def Pprior(self) -> kima.distributions.Distribution:
        """Prior for the orbital period(s)"""

    @Pprior.setter
    def Pprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def t0prior(self) -> kima.distributions.Distribution:
        """Prior for the time(s) of inferior conjunction"""

    @t0prior.setter
    def t0prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def RPprior(self) -> kima.distributions.Distribution:
        """Prior for the planet(s) radius"""

    @RPprior.setter
    def RPprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def aprior(self) -> kima.distributions.Distribution:
        """Prior for the planet(s) semi-major axis"""

    @aprior.setter
    def aprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def incprior(self) -> kima.distributions.Distribution:
        """Prior for the inclinations(s)"""

    @incprior.setter
    def incprior(self, arg: kima.distributions.Distribution, /) -> None: ...

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

class TRANSITmodel:
    def __init__(self, fix: bool, npmax: int, data: kima.Data.PHOTdata) -> None:
        """
        Implements a sum-of-Keplerians model where the number of Keplerians can be free.
        This model assumes white, uncorrelated noise.

        Args:
            fix (bool, default=True): whether the number of Keplerians should be fixed
            npmax (int, default=0): maximum number of Keplerians
            data (PHOTdata): the photometric data
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
    def data(self) -> kima.Data.PHOTdata:
        """the data"""

    @property
    def directory(self) -> str:
        """directory where the model ran"""

    @directory.setter
    def directory(self, arg: str, /) -> None: ...

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
    def kipping_ld(self) -> bool:
        """
        sample limb darkening as (q1, q2) of Kipping (2013); if False, as (u1, u2)
        """

    @kipping_ld.setter
    def kipping_ld(self, arg: bool, /) -> None: ...

    @property
    def use_stellar_density(self) -> bool:
        """derive a/Rs of all planets from a common stellar density (g/cm^3)"""

    @use_stellar_density.setter
    def use_stellar_density(self, arg: bool, /) -> None: ...

    @property
    def use_transit_windows(self) -> bool:
        """only evaluate the transit model near the transits (exact speed-up)"""

    @use_transit_windows.setter
    def use_transit_windows(self, arg: bool, /) -> None: ...

    @property
    def window_margin(self) -> float:
        """safety factor (>= 1) for the half-width of the transit windows"""

    @window_margin.setter
    def window_margin(self, arg: float, /) -> None: ...

    def __getstate__(self) -> tuple[bool, int, kima.Data.PHOTdata]: ...

    def __setstate__(self, arg: tuple[bool, int, kima.Data.PHOTdata], /) -> None: ...

    @property
    def Cprior(self) -> kima.distributions.Distribution:
        """Prior for the baseline flux"""

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
    def ld1_prior(self) -> kima.distributions.Distribution:
        """Prior for the first limb-darkening parameter (q1 or u1)"""

    @ld1_prior.setter
    def ld1_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def ld2_prior(self) -> kima.distributions.Distribution:
        """Prior for the second limb-darkening parameter (q2 or u2)"""

    @ld2_prior.setter
    def ld2_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def rho_prior(self) -> kima.distributions.Distribution:
        """Prior for the stellar density in g/cm^3 (if use_stellar_density)"""

    @rho_prior.setter
    def rho_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def nu_prior(self) -> kima.distributions.Distribution:
        """Prior for the degrees of freedom of the Student-t likelihood"""

    @nu_prior.setter
    def nu_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def conditional(self) -> TRANSITConditionalPrior: ...

    @conditional.setter
    def conditional(self, arg: TRANSITConditionalPrior, /) -> None: ...
