import os
import kima
from kima import RVData, GPmodel
from kima.distributions import Kumaraswamy, LogUniform
from kima.pykima.utils import chdir

__all__ = ['CoRoT7']

here = os.path.dirname(__file__) # cwd

def CoRoT7(run=False, load=False, **kwargs):
    """
    Create (and optionally run) a GP model for analysis of HARPS data for
    CoRoT-7. In this example, the number of Keplerians is fixed to 2 and the
    prior for the orbital periods is LogUniform(0.5, 10).

    Args:
        run (bool): whether to run the model
        **kwargs: keyword arguments passed directly to `kima.run`
    """
    # load the right data file
    data = RVData(os.path.join(here, 'corot7.txt'), skip=2)
    # create the model
    model = GPmodel(fix=True, npmax=2, data=data)
    
    model.conditional.Pprior = LogUniform(0.5, 10)
    model.conditional.eprior = Kumaraswamy(0.867, 3.03)

    kwargs.setdefault('steps', 1000)
    kwargs.setdefault('num_threads', 4)
    kwargs.setdefault('num_particles', 2)
    kwargs.setdefault('new_level_interval', 2000)
    kwargs.setdefault('save_interval', 200)

    with chdir(here):
        if run:
            kima.run(model, **kwargs)
        if load:
            res = kima.load_results(model)
            return model, res
    return model

if __name__ == '__main__':
    model, res = CoRoT7(run=False, load=True)
