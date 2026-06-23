#Import modules
import os
import numpy as np
import kima
from kima.pykima.utils import chdir
from kima import RVData, BINARIESmodel
from kima.distributions import (
    Uniform,
    UniformAngle,
    Gaussian,
    ModifiedLogUniform,
    LogUniform,
    Kumaraswamy,
)

__all__ = ['TIC172900988']

here = os.path.dirname(__file__) # cwd


def TIC172900988(run=False, load=False, **kwargs):
    """
    Create (and optionally run) an RV model for analysis of TIC172900988 SB2
    binary data. This loads SOPHIE data from `SOPHIE.rv` and creates a model
    where the number of Keplerians is free from 0 to 3.

    Args:
            run (bool): whether to run the model
            load (bool): load results after running
            **kwargs: keyword arguments passed directly to `kima.run`
    """

    data = RVData(
        os.path.join(here, "SOPHIE.rv"), units="kms", skip=2, double_lined=True
    )

    model = BINARIESmodel(fix=False, npmax=3, data=data)

    # Systemic velocity (m/s)
    model.Cprior = Uniform(25000, 27000)  
    # Jitter (m/s) (added in quadrature to the RV uncertainties)
    model.Jprior = ModifiedLogUniform(0.1, 100)  

    # Period prior (days), inner limit set by binary instability limit)
    model.conditional.Pprior = LogUniform(80, 1000)  
    # RV Semi-amplitude (m/s)
    model.conditional.Kprior = ModifiedLogUniform(1, 200)  
    # Eccentricity (Kumaraswamy prior acts as beta distribution see Standing et al. (2022))
    model.conditional.eprior = Kumaraswamy(0.867, 3.03)  
    # Argument of periastron
    model.conditional.wprior = UniformAngle()  
    # Starting phase of orbit
    model.conditional.phiprior = UniformAngle()  

    # Known object priors (used to set tight priors on the binary)  
    model.KO_Pprior = [Gaussian(19.658, 0.01)]  # Binary period (days)
    model.KO_Kprior = [Gaussian(58549, 100)]    # Binary semi-amplitude (m/s)
    model.KO_eprior = [Gaussian(0.448, 0.01)]   # Binary eccentricity
    model.KO_wprior = [Gaussian(1.23, 0.01)]    # Binary argument of periastron
    model.KO_wdotprior = [Gaussian(0, 1000)]    # Binary orbital precession (arcsec/yr)
    model.KO_phiprior = [UniformAngle()]        # Binary starting phase of orbit
    model.KO_qprior = [Gaussian(0.972, 0.01)]   # Binary mass ratio

    # prior on cosin of inclination not necessary as the eclipsing flag is on
    # (see below) so this is by default fixed to 0
    # model.KO_cosiprior = [] 

    # Set the sampler parameters (change steps to set length of the run, change
    # threads depending on machine, others are reasonable to leave)
    kwargs.setdefault("steps", 10000)
    kwargs.setdefault("num_threads", 4)
    kwargs.setdefault("num_particles", 2)
    kwargs.setdefault("new_level_interval", 2000)
    kwargs.setdefault("save_interval", 500)
    kwargs.setdefault("thread_steps", 50)

    # Model settings only required for binary stars:

    # Mass of the primary star (used for calculating the GR and Tidal
    # corrections, not necessary for double-lined binaries
    model.star_mass = 1.2368
    # calculate Post-Newtonian effects due to Tides (only important for very
    # tight binaries)
    model.tidal_correction = False
    # calculate Post-Newtonian effects due to GR
    model.relativistic_correction = True
    # True if double lined (SB2) binary
    model.double_lined = True
    # Set this to True if you are fitting for an eclipsing binary (inc is then
    # fixed to 90 deg)
    model.eclipsing = True

    with chdir(here):
        if run:
            kima.run(model, **kwargs)
        if load:
            res = kima.load_results(model)
            return model, res

    return model

if __name__ == '__main__':
	model, res = TIC172900988(run=False, load=True, steps=10_000)

