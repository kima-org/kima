from _typeshed import Incomplete

class SPLEAFmodel:
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
    alpha_prior: Incomplete
    beta_prior: Incomplete
    conditional: Incomplete
    cubic_prior: Incomplete
    degree: Incomplete
    directory: Incomplete
    enforce_stability: Incomplete
    eta1_prior: Incomplete
    eta2_prior: Incomplete
    eta3_prior: Incomplete
    eta4_prior: Incomplete
    fix: Incomplete
    kernel: Incomplete
    npmax: Incomplete
    quadr_prior: Incomplete
    series_jitters_prior: Incomplete
    slope_prior: Incomplete
    star_mass: Incomplete
    trend: Incomplete
    zero_points_prior: Incomplete
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
