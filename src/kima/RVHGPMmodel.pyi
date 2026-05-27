import kima.Data
import kima.distributions
from _typeshed import Incomplete

class RVHGPMConditionalPrior:
    Kprior: kima.distributions.Distribution
    Wprior: kima.distributions.Distribution
    Pprior: kima.distributions.Distribution
    eprior: kima.distributions.Distribution
    iprior: kima.distributions.Distribution
    phiprior: kima.distributions.Distribution
    wprior: kima.distributions.Distribution
    def __init__(self) -> None: ...

class RVHGPMmodel:
    AK_Kprior: list[kima.distributions.Distribution]
    AK_Pprior: list[kima.distributions.Distribution]
    AK_eprior: list[kima.distributions.Distribution]
    AK_phiprior: list[kima.distributions.Distribution]
    AK_t0prior: list[kima.distributions.Distribution]
    AK_tauprior: list[kima.distributions.Distribution]
    AK_wprior: list[kima.distributions.Distribution]
    Cprior: kima.distributions.Distribution
    Jprior: kima.distributions.Distribution
    KO_Kprior: list[kima.distributions.Distribution]
    KO_Pprior: list[kima.distributions.Distribution]
    KO_eprior: list[kima.distributions.Distribution]
    KO_phiprior: list[kima.distributions.Distribution]
    KO_wprior: list[kima.distributions.Distribution]
    KO_iprior: list[kima.distributions.Distribution]
    KO_Wprior: list[kima.distributions.Distribution]
    TR_Kprior: list[kima.distributions.Distribution]
    TR_Pprior: list[kima.distributions.Distribution]
    TR_Tcprior: list[kima.distributions.Distribution]
    TR_eprior: list[kima.distributions.Distribution]
    TR_wprior: list[kima.distributions.Distribution]
    beta_prior: Incomplete
    conditional: Incomplete
    cubic_prior: kima.distributions.Distribution
    degree: int
    directory: str
    enforce_stability: bool
    fix: bool
    indicator_correlations: bool
    individual_offset_prior: list[kima.distributions.Distribution]
    jitter_propto_indicator: bool
    jitter_propto_indicator_index: int
    npmax: int
    nu_prior: kima.distributions.Distribution
    offsets_prior: kima.distributions.Distribution
    parallax_prior: kima.distributions.Distribution
    pm_dec_bary_prior: kima.distributions.Distribution
    pm_ra_bary_prior: kima.distributions.Distribution
    quadr_prior: kima.distributions.Distribution
    remove_label_switching_degeneracy: bool
    slope_prior: kima.distributions.Distribution
    star_mass: float
    stellar_jitter_prior: kima.distributions.Distribution
    studentt: bool
    trend: bool
    def __init__(self, fix: bool, npmax: int, data: kima.Data.RVData, pm_data: kima.Data.HGPMdata) -> None: ...
    def set_apodized_keplerians(self, *args, **kwargs): ...
    def set_known_object(self, *args, **kwargs): ...
    def set_loguniform_prior_Np(self) -> None: ...
    def set_transiting_planet(self, *args, **kwargs): ...
    @property
    def apodized_keplerians(self) -> bool: ...
    @property
    def data(self) -> kima.Data.RVData: ...
    @property
    def known_object(self) -> bool: ...
    @property
    def n_apodized_keplerians(self) -> int: ...
    @property
    def n_known_object(self) -> int: ...
    @property
    def n_transiting_planet(self) -> int: ...
    @property
    def pm_data(self) -> kima.Data.HGPMdata: ...
    @property
    def transiting_planet(self) -> bool: ...
