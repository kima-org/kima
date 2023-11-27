import os
import kima
from kima import RVData, RVmodel, BINARIESmodel
from kima.pykima.utils import chdir

__all__ = ['_51Peg']

here = os.path.dirname(__file__) # cwd

def _51Peg(run=False, **kwargs):
    """
    Create (and optionally run) an RV model for analysis of 51 Peg data.
    This loads Keck/HIRES data from `51Peg.rv` and creates a model where
    the number of Keplerians is free from 0 to 1.

    Args:
        run (bool): whether to run the model
        **kwargs: keyword arguments passed directly to `kima.run`
    """
    # load the right data file
    data = RVData([os.path.join(here, '51Peg.rv')])
    # create the model
    model = RVmodel(fix=False, npmax=1, data=data)

    kwargs.setdefault('steps', 5000)
    kwargs.setdefault('num_threads', 4)
    kwargs.setdefault('num_particles', 2)
    kwargs.setdefault('new_level_interval', 2000)
    kwargs.setdefault('save_interval', 500)

    if run:
        kima.run(RVmodel, steps = 100, num_threads= 1, num_particles = 1, new_level_interval = 2000, save_interval = 100, thread_steps = 10, max_num_levels = 0, lambda_ = 10.0, beta = 100.0, compression = 2.718281828459045, seed = 0, print_thin = 50)
        # with chdir(here):
        #     kima.run(model, **kwargs)

    return model

if __name__ == '__main__':
    model = _51Peg(run=True, steps=80000)
    res = kima.load_results()
