import kima.Data
import kima.distributions
from _typeshed import Incomplete

class RVGAIAConditionalPrior:
    Omegaprior: kima.distributions.Distribution
    Pprior: kima.distributions.Distribution
    a0prior: kima.distributions.Distribution
    cosiprior: kima.distributions.Distribution
    eprior: kima.distributions.Distribution
    omegaprior: kima.distributions.Distribution
    phiprior: kima.distributions.Distribution
    def __init__(self) -> None: ...

class RVGAIAmodel:
    Cprior: kima.distributions.Distribution
    DEC: float
    J_GAIA_prior: kima.distributions.Distribution
    J_RV_prior: kima.distributions.Distribution
    KO_Omegaprior: list[kima.distributions.Distribution]
    KO_Pprior: list[kima.distributions.Distribution]
    KO_a0prior: list[kima.distributions.Distribution]
    KO_cosiprior: list[kima.distributions.Distribution]
    KO_eprior: list[kima.distributions.Distribution]
    KO_omegaprior: list[kima.distributions.Distribution]
    KO_phiprior: list[kima.distributions.Distribution]
    RA: float
    conditional: Incomplete
    cubic_prior: kima.distributions.Distribution
    da_prior: kima.distributions.Distribution
    dd_prior: kima.distributions.Distribution
    degree: int
    directory: str
    fix: bool
    indicator_correlations: bool
    individual_offset_prior: list[kima.distributions.Distribution]
    mua_prior: kima.distributions.Distribution
    mud_prior: kima.distributions.Distribution
    npmax: int
    nu_GAIA_prior: kima.distributions.Distribution
    nu_RV_prior: kima.distributions.Distribution
    offsets_prior: kima.distributions.Distribution
    parallax_prior: kima.distributions.Distribution
    quadr_prior: kima.distributions.Distribution
    slope_prior: kima.distributions.Distribution
    star_mass: float
    studentt: bool
    trend: bool
    def __init__(self, fix: bool, npmax: int, GAIAdata: kima.Data.GAIAdata, RVData: kima.Data.RVData) -> None: ...
    def set_known_object(self, *args, **kwargs): ...
    @property
    def GAIAdata(self) -> kima.Data.GAIAdata: ...
    @property
    def RVdata(self) -> kima.Data.RVData: ...
    @property
    def known_object(self) -> bool: ...
    @property
    def n_known_object(self) -> int: ...
