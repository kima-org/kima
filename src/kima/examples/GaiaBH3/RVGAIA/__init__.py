from pathlib import Path
import sys
import numpy as np
import kima
from kima import RVData, GAIAdata, GAIAmodel, RVGAIAmodel
from kima.pykima.utils import chdir
from kima.distributions import Uniform, Gaussian, ModifiedLogUniform, Kumaraswamy, LogUniform, Fixed

__all__ = ['RVGAIA']

here = Path(__file__).parent # cwd
	
def RVGAIA(run=False, load=False, **kwargs):
    """
    Create (and optionally run) an RV model for analysis of 51 Peg data.
    This loads Keck/HIRES data from `51Peg.rv` and creates a model where
    the number of Keplerians is free from 0 to 1.

    Args:
            run (bool): whether to run the model
            **kwargs: keyword arguments passed directly to `kima.run`
    """
    # load the right data file
    G_data_file = (here.parent / "Gaia_astrometry_BH3.gaia").as_posix()
    Gdata = GAIAdata(G_data_file, skip=2, units="mas")
    Gdata.M0_epoch = 57936.875

    RV_data_file = (here.parent / "Gaia_RVs_BH3.rdb").as_posix()
    Vdata = RVData(RV_data_file, skip=2, units="kms")

    # create the model
    model = RVGAIAmodel(fix=False, npmax=2, GAIAdata=Gdata, RVData=Vdata)

    model.Cprior = Uniform(-400_000, -300_000)
    model.J_GAIA_prior = ModifiedLogUniform(0.01, 0.2)
    model.J_RV_prior = ModifiedLogUniform(100.0, 10_000.0)

    model.conditional.eprior = Kumaraswamy(0.867, 3.03)
    model.conditional.Pprior = LogUniform(10.0, 4000.0)
    model.conditional.a0prior = ModifiedLogUniform(0.01, 2)
    model.conditional.cosiprior = Uniform(-1, 1)

    model.da_prior = Gaussian(4.2, 0.4)
    model.dd_prior = Gaussian(2.4, 0.3)
    model.mua_prior = Gaussian(-0.08, 0.2)
    model.mud_prior = Gaussian(-0.42, 0.2)
    model.parallax_prior = Uniform(1.3, 1.9)

    model.star_mass = 1.2
    model.RA = 294.82796478104
    model.DEC = 14.931669719919999

    model.set_known_object(1)

    model.KO_Pprior = [Gaussian(4280, 500)]
    model.KO_a0prior = [Gaussian(27.3, 4.0)]
    model.KO_eprior = [Gaussian(0.73, 0.04)]
    model.KO_omegaprior = [Gaussian(1.36, 0.05)]
    model.KO_Omegaprior = [Gaussian(2.38, 0.05)]
    model.KO_phiprior = [Uniform(0, 2 * np.pi)]
    model.KO_cosiprior = [Gaussian(-0.35, 0.05)]

    kwargs.setdefault("steps", 5000)
    kwargs.setdefault("num_threads", 4)
    kwargs.setdefault("num_particles", 2)
    kwargs.setdefault("new_level_interval", 20_000)
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
	model, res = RVGAIA(run='run' in sys.argv, load=True, steps=5000,
                        diagnostic='diagnostic' in sys.argv)


