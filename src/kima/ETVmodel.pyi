from collections.abc import Sequence

import kima.Data
import kima.RVmodel
import kima.distributions


class ETVConditionalPrior:
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

class ETVmodel:
    def __init__(self, fix: bool, npmax: int, data: kima.Data.ETVData) -> None:
        """
        Implements a sum-of-Keplerians model for eclipse timing data where the number of Keplerians can be free. These are light-travel time ETVs.

        Args:
            fix (bool):
                whether the number of Keplerians should be fixed
            npmax (int):
                maximum number of Keplerians
            data (ETVData):
                the Eclipse timing data
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
    def data(self) -> kima.Data.ETVData:
        """the data"""

    @property
    def ephemeris(self) -> int:
        """order of the ephemeris used"""

    @ephemeris.setter
    def ephemeris(self, arg: int, /) -> None: ...

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
        """whether the model includes (better) known extra Keplerian curve(s)"""

    @property
    def n_known_object(self) -> int:
        """how many known objects"""

    @property
    def star_mass(self) -> float:
        """stellar mass [Msun]"""

    @star_mass.setter
    def star_mass(self, arg: float, /) -> None: ...

    @property
    def Jprior(self) -> kima.distributions.Distribution:
        """Prior for the extra white noise (jitter)"""

    @Jprior.setter
    def Jprior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def ref_time_prior(self) -> kima.distributions.Distribution:
        """Prior for the reference time (where epoch = 0)"""

    @ref_time_prior.setter
    def ref_time_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def ephem1_prior(self) -> kima.distributions.Distribution:
        """Prior for the linear ephemeris"""

    @ephem1_prior.setter
    def ephem1_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def ephem2_prior(self) -> kima.distributions.Distribution:
        """Prior for the quadratic term of the ephemeris"""

    @ephem2_prior.setter
    def ephem2_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

    @property
    def ephem3_prior(self) -> kima.distributions.Distribution:
        """Prior for the cubic term of the ephemeris"""

    @ephem3_prior.setter
    def ephem3_prior(self, arg: kima.distributions.Distribution, /) -> None: ...

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
    def conditional(self) -> ETVConditionalPrior: ...

    @conditional.setter
    def conditional(self, arg: kima.RVmodel.KeplerianConditionalPrior, /) -> None: ...
