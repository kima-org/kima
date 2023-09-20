import os
import kima
from kima import RVData, RVmodel
from kima import distributions
from kima.pykima.utils import chdir

__all__ = ['_51Peg']

here = os.path.dirname(__file__)

def _51Peg(run=False, **kwargs):
    """
    Create (and optionally run) an RV model for analysis of 51 Peg data.
    This loads Keck/HIRES data from `51Peg.rv` and creates a model where
    the number of Keplerians is free from 0 to 1.

    Args:
        run (bool): whether to run the model
        **kwargs: keyword arguments passed directly to `kima.run`
    """
    data = RVData(os.path.join(here, '51Peg.rv'))
    model = RVmodel(fix=True, npmax=1, data=data)

    kwargs.setdefault('steps', 100)
    kwargs.setdefault('num_threads', 4)
    kwargs.setdefault('num_particles', 2)
    kwargs.setdefault('new_level_interval', 1000)
    kwargs.setdefault('save_interval', 200)

    if run:
        with chdir(here):
            kima.run(model, **kwargs)

    return model

