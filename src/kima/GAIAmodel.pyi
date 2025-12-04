import kima.Data
import kima.distributions
from _typeshed import Incomplete

class GAIAConditionalPrior:
    Aprior: kima.distributions.Distribution
    Bprior: kima.distributions.Distribution
    Fprior: kima.distributions.Distribution
    Gprior: kima.distributions.Distribution
    Omegaprior: kima.distributions.Distribution
    Pprior: kima.distributions.Distribution
    a0prior: kima.distributions.Distribution
    cosiprior: kima.distributions.Distribution
    eprior: kima.distributions.Distribution
    omegaprior: kima.distributions.Distribution
    phiprior: kima.distributions.Distribution
    thiele_innes: bool
    def __init__(self) -> None: ...

class GAIAmodel:
    DEC: float
    Jprior: kima.distributions.Distribution
    KO_Omegaprior: list[kima.distributions.Distribution]
    KO_Pprior: list[kima.distributions.Distribution]
    KO_a0prior: list[kima.distributions.Distribution]
    KO_cosiprior: list[kima.distributions.Distribution]
    KO_eprior: list[kima.distributions.Distribution]
    KO_omegaprior: list[kima.distributions.Distribution]
    KO_phiprior: list[kima.distributions.Distribution]
    RA: float
    conditional: Incomplete
    da_prior: kima.distributions.Distribution
    dd_prior: kima.distributions.Distribution
    fix: bool
    mua_prior: kima.distributions.Distribution
    mud_prior: kima.distributions.Distribution
    npmax: int
    nu_prior: kima.distributions.Distribution
    parallax_prior: kima.distributions.Distribution
    star_mass: float
    studentt: bool
    thiele_innes: bool
    def __init__(self, fix: bool, npmax: int, data: kima.Data.GAIAdata) -> None: ...
    def set_known_object(self, *args, **kwargs): ...
    @property
    def data(self) -> kima.Data.GAIAdata: ...
    @property
    def known_object(self) -> bool: ...
    @property
    def n_known_object(self) -> int: ...
