from _typeshed import Incomplete
from typing import Any

class GPmodel:
    Cprior: Incomplete
    Jprior: Incomplete
    KO_Kprior: Incomplete
    KO_Pprior: Incomplete
    KO_eprior: Incomplete
    KO_phiprior: Incomplete
    KO_wprior: Incomplete
    Q_prior: Incomplete
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
    eta1_prior: Incomplete
    eta2_prior: Incomplete
    eta3_prior: Incomplete
    eta4_prior: Incomplete
    eta5_prior: Incomplete
    eta6_prior: Incomplete
    eta7_prior: Incomplete
    fix: Incomplete
    indicator_correlations: Incomplete
    individual_offset_prior: Incomplete
    kernel: Incomplete
    magnetic_cycle_kernel: Incomplete
    npmax: Incomplete
    offsets_prior: Incomplete
    quadr_prior: Incomplete
    slope_prior: Incomplete
    star_mass: Incomplete
    trend: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def eta2_larger_eta3(self, *args, **kwargs): ...
    def set_known_object(self, *args, **kwargs): ...
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
