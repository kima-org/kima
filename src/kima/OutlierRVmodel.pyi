import kima.Data
import kima.RVmodel
import kima.distributions


class OutlierRVmodel:
    def __init__(self, fix: bool, npmax: int, data: kima.Data.RVdata) -> None:
        """
        Implements a sum-of-Keplerians model where the number of Keplerians can be free.
        This model assumes white, uncorrelated noise.

        Args:
            fix (bool, default=True): whether the number of Keplerians should be fixed
            npmax (int, default=0): maximum number of Keplerians
            data (RVdata): the RV data
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
    def known_object(self) -> bool:
        """whether to include (better) known extra Keplerian curve(s)"""

    @known_object.setter
    def known_object(self, arg: bool, /) -> None: ...

    @property
    def n_known_object(self) -> int:
        """how many known objects"""

    @n_known_object.setter
    def n_known_object(self, arg: int, /) -> None: ...

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
    def conditional(self) -> kima.RVmodel.KeplerianConditionalPrior: ...

    @conditional.setter
    def conditional(self, arg: kima.RVmodel.KeplerianConditionalPrior, /) -> None: ...
