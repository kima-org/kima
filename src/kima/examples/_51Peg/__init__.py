import os
import kima
from kima import RVData, RVmodel, BINARIESmodel
from kima.pykima.utils import chdir

__all__ = ['_51Peg']

here = os.path.dirname(__file__) # cwd

def _51Peg(run=False, load=False, **kwargs):
    """
    Create (and optionally run) an RV model for analysis of 51 Peg data.
    This loads Keck/HIRES data from `51Peg.rv` and creates a model where
    the number of Keplerians is free from 0 to 1.

    Args:
        run (bool): whether to run the model
        load (bool): load results after running
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
        with chdir(here):
            kima.run(model, **kwargs)
            if load:
                res = kima.load_results()
                return model, res

    return model

if __name__ == '__main__':
    model, res = _51Peg(run=True, load=True, steps=80000)
