from _typeshed import Incomplete

class GAIAConditionalPrior:
    Aprior: Incomplete
    Bprior: Incomplete
    Fprior: Incomplete
    Gprior: Incomplete
    Omegaprior: Incomplete
    Pprior: Incomplete
    a0prior: Incomplete
    cosiprior: Incomplete
    eprior: Incomplete
    omegaprior: Incomplete
    phiprior: Incomplete
    thiele_innes: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...

class GAIAmodel:
    Jprior: Incomplete
    KO_Omegaprior: Incomplete
    KO_Pprior: Incomplete
    KO_a0prior: Incomplete
    KO_cosiprior: Incomplete
    KO_eprior: Incomplete
    KO_omegaprior: Incomplete
    KO_phiprior: Incomplete
    conditional: Incomplete
    da_prior: Incomplete
    dd_prior: Incomplete
    mua_prior: Incomplete
    mud_prior: Incomplete
    nu_prior: Incomplete
    parallax_prior: Incomplete
    studentt: Incomplete
    thiele_innes: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def set_known_object(self, *args, **kwargs): ...
    @property
    def known_object(self): ...
    @property
    def n_known_object(self): ...
