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

def multi_instruments(run=False, load=False, **kwargs):
    """
    Create (and optionally run) an RV model for analysis of HD106252 data from
    multiple instruments, namely ELODIE, HET, HJS, and Lick. 

    Args:
        run (bool): whether to run the model
        load (bool): load results after running
        **kwargs: keyword arguments passed directly to `kima.run`
    """
    data = HD106252_combined

    # create the model
    model = RVmodel(fix=False, npmax=1, data=data)

    #model.Cprior = get_gaussian_prior_vsys(data)
    #model.individual_offset_prior = get_gaussian_priors_individual_offsets(data)

    kwargs.setdefault('steps', 50_000)
    kwargs.setdefault('num_threads', 4)
    kwargs.setdefault('num_particles', 4)
    kwargs.setdefault('new_level_interval', 5000)
    kwargs.setdefault('save_interval', 500)

    if run:
        with chdir(here):
            kima.run(model, **kwargs)
            if load:
                res = kima.load_results()
                return model, res

    return model

if __name__ == '__main__':
    model, res = multi_instruments(run=True, load=True)
