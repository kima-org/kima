import os
import kima
from kima import RVData, RVmodel
from kima.pykima.utils import get_gaussian_prior_vsys, get_gaussian_priors_individual_offsets
from kima.pykima.utils import chdir

__all__ = ['multi_instruments']

here = os.path.dirname(__file__) # cwd

HD106252_ELODIE = RVData(os.path.join(here, 'HD106252_ELODIE.txt'))
HD106252_HET = RVData(os.path.join(here, 'HD106252_HET.txt'))
HD106252_HJS = RVData(os.path.join(here, 'HD106252_HJS.txt'))
HD106252_Lick = RVData(os.path.join(here, 'HD106252_Lick.txt'))
HD106252_combined = RVData([
    os.path.join(here, 'HD106252_ELODIE.txt'),
    os.path.join(here, 'HD106252_HET.txt'),
    os.path.join(here, 'HD106252_HJS.txt'),
    os.path.join(here, 'HD106252_Lick.txt'),
])

def multi_instruments(run=False, **kwargs):
    """
    Create (and optionally run) an RV model for analysis of HD106252 data from
    multiple instruments, namely ELODIE, HET, HJS, and Lick. 

    Args:
        run (bool): whether to run the model
        **kwargs: keyword arguments passed directly to `kima.run`
    """
    data = HD106252_combined

    # create the model
    model = RVmodel(fix=True, npmax=0, data=data)

    model.Cprior = get_gaussian_prior_vsys(data)
    model.individual_offset_prior = get_gaussian_priors_individual_offsets(data)

    kwargs.setdefault('steps', 20_000)
    kwargs.setdefault('num_threads', 4)
    kwargs.setdefault('num_particles', 2)
    kwargs.setdefault('new_level_interval', 2000)
    kwargs.setdefault('save_interval', 200)

    if run:
        with chdir(here):
            kima.run(model, **kwargs)

    return model

if __name__ == '__main__':
    model = multi_instruments(run=True)
    res = kima.load_results()
