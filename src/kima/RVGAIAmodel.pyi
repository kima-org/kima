from _typeshed import Incomplete

class RVGAIAConditionalPrior:
    Mprior: Incomplete
    Omegaprior: Incomplete
    Pprior: Incomplete
    cosiprior: Incomplete
    eprior: Incomplete
    omegaprior: Incomplete
    phiprior: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...

class RVGAIAmodel:
    Cprior: Incomplete
    J_GAIA_prior: Incomplete
    J_RV_prior: Incomplete
    KO_Mprior: Incomplete
    KO_Omegaprior: Incomplete
    KO_Pprior: Incomplete
    KO_cosiprior: Incomplete
    KO_eprior: Incomplete
    KO_omegaprior: Incomplete
    KO_phiprior: Incomplete
    conditional: Incomplete
    cubic_prior: Incomplete
    da_prior: Incomplete
    dd_prior: Incomplete
    degree: Incomplete
    directory: Incomplete
    fix: Incomplete
    indicator_correlations: Incomplete
    individual_offset_prior: Incomplete
    mua_prior: Incomplete
    mud_prior: Incomplete
    npmax: Incomplete
    nu_GAIA_prior: Incomplete
    nu_RV_prior: Incomplete
    offsets_prior: Incomplete
    parallax_prior: Incomplete
    quadr_prior: Incomplete
    slope_prior: Incomplete
    star_mass: Incomplete
    studentt: Incomplete
    trend: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def set_known_object(self, *args, **kwargs): ...
    @property
    def GAIAdata(self): ...
    @property
    def RVdata(self): ...
    @property
    def known_object(self): ...
    @property
    def n_known_object(self): ...
