from pathlib import Path
import sys
import numpy as np
import kima
from kima import RVData, GAIAdata, RVmodel, BINARIESmodel, GAIAmodel
from kima.pykima.utils import chdir
from kima.distributions import Uniform, Gaussian, ModifiedLogUniform, Kumaraswamy, LogUniform, Fixed

__all__ = ['GAIA_free']

here = Path(__file__).parent # cwd

def GAIA_free(run=False, load=False, **kwargs):
    """
    Create (and optionally run) a model for analysis of Gaia data for BH3.  
    This loads Gaia data from `Gaia_astrometry_BH3.gaia` and creates a model
    where the number of Keplerians is free from 0 to 2.

    Args:
        run (bool): 
            whether to run the model
        load (bool):
            load results after running
        **kwargs:
            keyword arguments passed directly to `kima.run`
    """
    # load the right data file
    data_file = (here.parent / "Gaia_astrometry_BH3.gaia").as_posix()
    data = GAIAdata(data_file, skip=2, units="mas")
    data.M0_epoch = 57936.875

    # create the model
    model = GAIAmodel(fix=False, npmax=2, data=data)

    model.Jprior = Fixed(0.5)

    model.conditional.eprior = Kumaraswamy(0.867, 3.03)
    model.conditional.Pprior = LogUniform(100.0, 10000.0)
    model.conditional.a0prior = ModifiedLogUniform(0.1, 100.0)
    model.conditional.cosiprior = Uniform(-1, 1)

    model.da_prior = Gaussian(4.2, 0.4)
    model.dd_prior = Gaussian(2.4, 0.3)
    model.mua_prior = Gaussian(-0.08, 0.2)
    model.mud_prior = Gaussian(-0.42, 0.2)
    model.parallax_prior = Uniform(1.3, 1.9)

    model.star_mass = 1.2
    model.RA = 294.82796478104
    model.DEC = 14.931669719919999

    kwargs.setdefault("steps", 5000)
    kwargs.setdefault("num_threads", 4)
    kwargs.setdefault("num_particles", 2)
    kwargs.setdefault("new_level_interval", 20000)
    kwargs.setdefault("save_interval", 2000)
    diagnostic = kwargs.pop('diagnostic', False)

    with chdir(here):
        if run:
            kima.run(model, **kwargs)
        if load:
            res = kima.load_results(model, diagnostic=diagnostic)
            return model, res

    return model

if __name__ == '__main__':
    model, res = GAIA_free(run=True, load=True, steps=10_000,
                           diagnostic='diagnostic' in sys.argv)


