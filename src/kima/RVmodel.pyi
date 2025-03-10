from _typeshed import Incomplete
from typing import Any

class RVConditionalPrior:
    Kprior: Incomplete
    Pprior: Incomplete
    eprior: Incomplete
    phiprior: Incomplete
    wprior: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...

class RVmodel:
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
    quadr_prior: Incomplete
    slope_prior: Incomplete
    star_mass: Incomplete
    stellar_jitter_prior: Incomplete
    studentt: Incomplete
    trend: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def set_known_object(self, *args, **kwargs): ...
    def set_loguniform_prior_Np(self, *args, **kwargs): ...
    def set_transiting_planet(self, *args, **kwargs): ...
    @property
    def data(self): ...
    @property
    def known_object(self): ...
    @property
    def n_known_object(self): ...
    @property
    def n_transiting_planet(self): ...
    @property
    def transiting_planet(self): ...

class TRANSITConditionalPrior:
    Pprior: Incomplete
    RPprior: Incomplete
    aprior: Incomplete
    eprior: Incomplete
    incprior: Incomplete
    t0prior: Incomplete
    wprior: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
