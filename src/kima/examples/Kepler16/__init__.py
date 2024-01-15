import os
import numpy as np
import kima
from kima import RVData, RVmodel, BINARIESmodel
from kima.pykima.utils import chdir
from kima.distributions import Uniform, Gaussian, ModifiedLogUniform

__all__ = ['Kepler16']

here = os.path.dirname(__file__) # cwd

def Kepler16(run=False, **kwargs):
    """
    Create (and optionally run) an RV model for analysis of 51 Peg data.
    This loads Keck/HIRES data from `51Peg.rv` and creates a model where
    the number of Keplerians is free from 0 to 1.

    Args:
        run (bool): whether to run the model
        **kwargs: keyword arguments passed directly to `kima.run`
    """
    # load the right data file
    data = RVData([os.path.join(here, 'Kepler16.rdb')],skip=2,units='kms')
    # create the model
    model = BINARIESmodel(fix=False, npmax=3, data=data)
    
    model.Cprior = Uniform(-44300,-23300)
    model.Jprior = ModifiedLogUniform(0.1,100)
    
    # model.set_known_object = 1
    # model.n_known_object = 1
    model.KO_Pprior = [Gaussian(41,1)]
    model.KO_Kprior = [Gaussian(13600,900)]
    model.KO_eprior = [Gaussian(0.16,0.02)]
    model.KO_wprior = [Uniform(0,2*np.pi)]
    model.KO_wdotprior = [Gaussian(0,1000)]
    model.KO_phiprior = [Uniform(0,2*np.pi)]

    kwargs.setdefault('steps', 5000)
    kwargs.setdefault('num_threads', 4)
    kwargs.setdefault('num_particles', 2)
    kwargs.setdefault('new_level_interval', 5000)
    kwargs.setdefault('save_interval', 1000)

    if run:
        kima.run(model, **kwargs)
        # with chdir(here):
        #     kima.run(model, **kwargs)

    return model

if __name__ == '__main__':
    model = Kepler16(run=True, steps=1000)
    res = kima.load_results()


