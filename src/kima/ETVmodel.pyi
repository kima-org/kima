from _typeshed import Incomplete

class ETVConditionalPrior:
    Kprior: Incomplete
    Pprior: Incomplete
    eprior: Incomplete
    phiprior: Incomplete
    wprior: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...

class ETVmodel:
    Jprior: Incomplete
    KO_Kprior: Incomplete
    KO_Pprior: Incomplete
    KO_eprior: Incomplete
    KO_phiprior: Incomplete
    KO_wprior: Incomplete
    conditional: Incomplete
    directory: Incomplete
    ephem1_prior: Incomplete
    ephem2_prior: Incomplete
    ephem3_prior: Incomplete
    ephemeris: Incomplete
    fix: Incomplete
    npmax: Incomplete
    nu_prior: Incomplete
    ref_time_prior: Incomplete
    star_mass: Incomplete
    studentt: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def data(self): ...
    @property
    def known_object(self): ...
    @property
    def n_known_object(self): ...
