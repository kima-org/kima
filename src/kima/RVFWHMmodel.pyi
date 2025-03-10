from _typeshed import Incomplete
from typing import Any

class RVFWHMmodel:
    Cfwhm_prior: Incomplete
    Cprior: Incomplete
    Jfwhm_prior: Incomplete
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
    conditional: Incomplete
    cubic_fwhm_prior: Incomplete
    cubic_prior: Incomplete
    degree: Incomplete
    degree_fwhm: Incomplete
    directory: Incomplete
    enforce_stability: Incomplete
    eta1_fwhm_prior: Incomplete
    eta1_prior: Incomplete
    eta2_fwhm_prior: Incomplete
    eta2_prior: Incomplete
    eta3_fwhm_prior: Incomplete
    eta3_prior: Incomplete
    eta4_fwhm_prior: Incomplete
    eta4_prior: Incomplete
    fix: Incomplete
    npmax: Incomplete
    quadr_fwhm_prior: Incomplete
    quadr_prior: Incomplete
    share_eta2: Incomplete
    share_eta3: Incomplete
    share_eta4: Incomplete
    slope_fwhm_prior: Incomplete
    slope_prior: Incomplete
    star_mass: Incomplete
    trend: Any
    trend_fwhm: Incomplete
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
