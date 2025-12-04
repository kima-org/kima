import kima.Data
import kima.distributions
from _typeshed import Incomplete

class OutlierRVmodel:
    Cprior: kima.distributions.Distribution
    Jprior: kima.distributions.Distribution
    conditional: Incomplete
    cubic_prior: kima.distributions.Distribution
    degree: int
    enforce_stability: bool
    fix: bool
    known_object: bool
    n_known_object: int
    npmax: int
    nu_prior: kima.distributions.Distribution
    offsets_prior: kima.distributions.Distribution
    quadr_prior: kima.distributions.Distribution
    slope_prior: kima.distributions.Distribution
    star_mass: float
    studentt: bool
    trend: bool
    def __init__(self, fix: bool, npmax: int, data: kima.Data.RVData) -> None: ...
    @property
    def data(self) -> kima.Data.RVData: ...
