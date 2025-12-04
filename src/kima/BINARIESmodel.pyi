import kima.Data
import kima.distributions
from _typeshed import Incomplete

class BINARIESmodel:
    Cprior: kima.distributions.Distribution
    Jprior: kima.distributions.Distribution
    KO_Kprior: list[kima.distributions.Distribution]
    KO_Pprior: list[kima.distributions.Distribution]
    KO_cosiprior: list[kima.distributions.Distribution]
    KO_eprior: list[kima.distributions.Distribution]
    KO_phiprior: list[kima.distributions.Distribution]
    KO_qprior: list[kima.distributions.Distribution]
    KO_wdotprior: list[kima.distributions.Distribution]
    KO_wprior: list[kima.distributions.Distribution]
    binary_mass: float
    binary_radius: float
    conditional: Incomplete
    cubic_prior: kima.distributions.Distribution
    degree: int
    directory: str
    double_lined: bool
    eclipsing: bool
    enforce_stability: bool
    fix: bool
    known_object: bool
    n_known_object: int
    npmax: int
    nu_prior: kima.distributions.Distribution
    offsets_prior: kima.distributions.Distribution
    quadr_prior: kima.distributions.Distribution
    relativistic_correction: bool
    slope_prior: kima.distributions.Distribution
    star_mass: float
    star_radius: float
    studentt: bool
    tidal_correction: bool
    trend: bool
    use_binary_longitude: bool
    def __init__(self, fix: bool, npmax: int, data: kima.Data.RVData) -> None: ...
    @property
    def data(self) -> kima.Data.RVData: ...
