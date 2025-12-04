import kima.Data
import kima.distributions
from _typeshed import Incomplete

class ETVConditionalPrior:
    Kprior: kima.distributions.Distribution
    Pprior: kima.distributions.Distribution
    eprior: kima.distributions.Distribution
    phiprior: kima.distributions.Distribution
    wprior: kima.distributions.Distribution
    def __init__(self) -> None: ...

class ETVmodel:
    Jprior: kima.distributions.Distribution
    KO_Kprior: list[kima.distributions.Distribution]
    KO_Pprior: list[kima.distributions.Distribution]
    KO_eprior: list[kima.distributions.Distribution]
    KO_phiprior: list[kima.distributions.Distribution]
    KO_wprior: list[kima.distributions.Distribution]
    conditional: Incomplete
    directory: str
    ephem1_prior: kima.distributions.Distribution
    ephem2_prior: kima.distributions.Distribution
    ephem3_prior: kima.distributions.Distribution
    ephemeris: int
    fix: bool
    npmax: int
    nu_prior: kima.distributions.Distribution
    ref_time_prior: kima.distributions.Distribution
    star_mass: float
    studentt: bool
    def __init__(self, fix: bool, npmax: int, data: kima.Data.ETVData) -> None: ...
    @property
    def data(self) -> kima.Data.ETVData: ...
    @property
    def known_object(self) -> bool: ...
    @property
    def n_known_object(self) -> int: ...
