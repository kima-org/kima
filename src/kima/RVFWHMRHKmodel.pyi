from _typeshed import Incomplete

class RVFWHMRHKmodel:
    Cfwhm_prior: Incomplete
    Cprior: Incomplete
    Crhk_prior: Incomplete
    Jfwhm_prior: Incomplete
    Jprior: Incomplete
    Jrhk_prior: Incomplete
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
    conditional: Incomplete
    cubic_prior: Incomplete
    degree: Incomplete
    directory: Incomplete
    enforce_stability: Incomplete
    eta1_fwhm_prior: Incomplete
    eta1_prior: Incomplete
    eta1_rhk_prior: Incomplete
    eta2_fwhm_prior: Incomplete
    eta2_prior: Incomplete
    eta2_rhk_prior: Incomplete
    eta3_fwhm_prior: Incomplete
    eta3_prior: Incomplete
    eta3_rhk_prior: Incomplete
    eta4_fwhm_prior: Incomplete
    eta4_prior: Incomplete
    eta4_rhk_prior: Incomplete
    fix: Incomplete
    magnetic_cycle_kernel: Incomplete
    npmax: Incomplete
    quadr_prior: Incomplete
    share_eta2: Incomplete
    share_eta3: Incomplete
    share_eta4: Incomplete
    slope_prior: Incomplete
    star_mass: Incomplete
    trend: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
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
