import os
import kima
from kima import RVData, RVmodel
from kima import distributions
from kima.pykima.utils import chdir

__all__ = ['BL2009']

here = os.path.dirname(__file__) # cwd

def BL2009(run=False, which=1, **kwargs):
    """
    Create (and optionally run) an RV model for analysis of simulated datasets
    from Balan & Lahav (2009).

    Args:
        run (bool): whether to run the model
        **kwargs: keyword arguments passed directly to `kima.run`
    """
    # load the right data file
    file = os.path.join(here, f'BL2009_dataset{which}.rv')
    data = RVData(file)

    # create the model
    model = RVmodel(fix=False, npmax=which, data=data)

    # same priors as in Balan & Lahav (2009, DOI: 10.1111/j.1365-2966.2008.14385.x)
    model.Cprior = distributions.Uniform(-2000, 2000)
    model.Jprior = distributions.ModifiedLogUniform(1, 2000)
    model.conditional.Pprior = distributions.LogUniform(0.2, 15e3)
    model.conditional.Kprior = distributions.ModifiedLogUniform(1.0, 2e3)
    model.conditional.eprior = distributions.Uniform(0, 1)

    kwargs.setdefault('steps', 5000)
    kwargs.setdefault('num_threads', 4)
    kwargs.setdefault('num_particles', 2)
    kwargs.setdefault('new_level_interval', 2000)
    kwargs.setdefault('save_interval', 500)
    kwargs.setdefault('thin_print', 200)

    if run:
        # with chdir(here):
        kima.run(model, **kwargs)

    return model

if __name__ == '__main__':
    model = BL2009(run=True, which=1, steps=20_000)
    res = kima.load_results()
