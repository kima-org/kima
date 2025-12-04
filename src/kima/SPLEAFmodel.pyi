import kima.Data
import kima.GP
import kima.distributions
from _typeshed import Incomplete

class SPLEAFmodel:
    Cprior: kima.distributions.Distribution
    Jprior: kima.distributions.Distribution
    KO_Kprior: list[kima.distributions.Distribution]
    KO_Pprior: list[kima.distributions.Distribution]
    KO_eprior: list[kima.distributions.Distribution]
    KO_phiprior: list[kima.distributions.Distribution]
    KO_wprior: list[kima.distributions.Distribution]
    Q_prior: kima.distributions.Distribution
    TR_Kprior: list[kima.distributions.Distribution]
    TR_Pprior: list[kima.distributions.Distribution]
    TR_Tcprior: list[kima.distributions.Distribution]
    TR_eprior: list[kima.distributions.Distribution]
    TR_wprior: list[kima.distributions.Distribution]
    alpha_prior: Incomplete
    beta_prior: Incomplete
    conditional: Incomplete
    cubic_prior: kima.distributions.Distribution
    degree: int
    directory: str
    enforce_stability: bool
    eta1_prior: kima.distributions.Distribution
    eta2_prior: kima.distributions.Distribution
    eta3_prior: kima.distributions.Distribution
    eta4_prior: kima.distributions.Distribution
    fix: bool
    kernel: kima.GP.KernelType
    npmax: int
    quadr_prior: kima.distributions.Distribution
    series_jitters_prior: list[kima.distributions.Distribution]
    slope_prior: kima.distributions.Distribution
    star_mass: float
    trend: bool
    zero_points_prior: list[kima.distributions.Distribution]
    def __init__(self, fix: bool, npmax: int, data: kima.Data.RVData) -> None: ...
    def eta2_larger_eta3(self, factor: float = ...) -> None: ...
    def set_known_object(self, *args, **kwargs): ...
    def set_transiting_planet(self, *args, **kwargs): ...
    @property
    def data(self) -> kima.Data.RVData: ...
    @property
    def known_object(self) -> bool: ...
    @property
    def n_known_object(self) -> int: ...
    @property
    def n_transiting_planet(self) -> int: ...
    @property
    def transiting_planet(self) -> bool: ...
