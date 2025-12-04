import kima.Data
import kima.distributions
from _typeshed import Incomplete

class RVFWHMmodel:
    Cfwhm_prior: kima.distributions.Distribution
    Cprior: kima.distributions.Distribution
    Jfwhm_prior: kima.distributions.Distribution
    Jprior: kima.distributions.Distribution
    KO_Kprior: list[kima.distributions.Distribution]
    KO_Pprior: list[kima.distributions.Distribution]
    KO_eprior: list[kima.distributions.Distribution]
    KO_phiprior: list[kima.distributions.Distribution]
    KO_wprior: list[kima.distributions.Distribution]
    TR_Kprior: list[kima.distributions.Distribution]
    TR_Pprior: list[kima.distributions.Distribution]
    TR_Tcprior: list[kima.distributions.Distribution]
    TR_eprior: list[kima.distributions.Distribution]
    TR_wprior: list[kima.distributions.Distribution]
    conditional: Incomplete
    cubic_fwhm_prior: kima.distributions.Distribution
    cubic_prior: kima.distributions.Distribution
    degree: int
    degree_fwhm: int
    directory: str
    enforce_stability: bool
    eta1_fwhm_prior: kima.distributions.Distribution
    eta1_prior: kima.distributions.Distribution
    eta2_fwhm_prior: kima.distributions.Distribution
    eta2_prior: kima.distributions.Distribution
    eta3_fwhm_prior: kima.distributions.Distribution
    eta3_prior: kima.distributions.Distribution
    eta4_fwhm_prior: kima.distributions.Distribution
    eta4_prior: kima.distributions.Distribution
    fix: bool
    npmax: int
    quadr_fwhm_prior: kima.distributions.Distribution
    quadr_prior: kima.distributions.Distribution
    share_eta2: bool
    share_eta3: bool
    share_eta4: bool
    slope_fwhm_prior: kima.distributions.Distribution
    slope_prior: kima.distributions.Distribution
    star_mass: float
    trend: bool
    trend_fwhm: bool
    def __init__(self, fix: bool, npmax: int, data: kima.Data.RVData) -> None: ...
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
