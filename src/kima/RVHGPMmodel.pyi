from _typeshed import Incomplete
from typing import Any

class RVHGPMConditionalPrior:
    Kprior: Incomplete
    Omegaprior: Incomplete
    Pprior: Incomplete
    eprior: Incomplete
    iprior: Incomplete
    phiprior: Incomplete
    wprior: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...

class RVHGPMmodel:
    AK_Kprior: Incomplete
    AK_Pprior: Incomplete
    AK_eprior: Incomplete
    AK_phiprior: Incomplete
    AK_t0prior: Incomplete
    AK_tauprior: Incomplete
    AK_wprior: Incomplete
    Cprior: Incomplete
    Jprior: Incomplete
    KO_Kprior: Incomplete
    KO_Pprior: Incomplete
    KO_eprior: Incomplete
    KO_phiprior: Incomplete
    KO_wprior: Incomplete
    TR_Kprior: Incomplete
    TR_Pprior: Incomplete
    TR_Tcprior: Incomplete
    TR_eprior: Incomplete
    TR_wprior: Incomplete
    beta_prior: Any
    conditional: Incomplete
    cubic_prior: Incomplete
    degree: Incomplete
    directory: Incomplete
    enforce_stability: Incomplete
    fix: Incomplete
    indicator_correlations: Incomplete
    individual_offset_prior: Incomplete
    jitter_propto_indicator: Incomplete
    jitter_propto_indicator_index: Incomplete
    npmax: Incomplete
    nu_prior: Incomplete
    offsets_prior: Incomplete
    parallax_prior: Incomplete
    pm_dec_bary_prior: Incomplete
    pm_ra_bary_prior: Incomplete
    quadr_prior: Incomplete
    remove_label_switching_degeneracy: Incomplete
    slope_prior: Incomplete
    star_mass: Incomplete
    stellar_jitter_prior: Incomplete
    studentt: Incomplete
    trend: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def set_apodized_keplerians(self, *args, **kwargs): ...
    def set_known_object(self, *args, **kwargs): ...
    def set_loguniform_prior_Np(self, *args, **kwargs): ...
    def set_transiting_planet(self, *args, **kwargs): ...
    @property
    def apodized_keplerians(self): ...
    @property
    def data(self): ...
    @property
    def known_object(self): ...
    @property
    def n_apodized_keplerians(self): ...
    @property
    def n_known_object(self): ...
    @property
    def n_transiting_planet(self): ...
    @property
    def transiting_planet(self): ...
