#Import modules
import os
import numpy as np
import kima
from kima import RVData, RVmodel, BINARIESmodel
from kima.pykima.utils import chdir
from kima.distributions import Uniform, Gaussian, ModifiedLogUniform, LogUniform, Kumaraswamy

__all__ = ['TIC172900988']

here = os.path.dirname(__file__) # cwd


def TIC172900988(run=False, load=False, **kwargs):
	"""
	Create (and optionally run) an RV model for analysis of TIC-1729 SB2 binary data.
	This loads SOPHIE data from `SOPHIE.rdb` and creates a model where
	the number of Keplerians is free from 0 to 3.

	Args:
		run (bool): whether to run the model
		**kwargs: keyword arguments passed directly to `kima.run`
	"""

	data = RVData(os.path.join(here, 'SOPHIE.rv'), units='kms', skip=2, double_lined=True)
	#Set the required model
	model = kima.BINARIESmodel(fix=False, npmax=3, data=data)

	#Set the priors on the model
	model.Cprior = Uniform(25000,27000) #Systemic velocity (m/s)
	model.Jprior = ModifiedLogUniform(0.1,100) #Jitter (m/s) (added in quadrature to the RV uncertainties)

	model.conditional.Pprior = LogUniform(80, 1000) #Period prior (days), inner limit set by binary instability limit)
	model.conditional.Kprior = ModifiedLogUniform(1,200) #RV Semi-amplitude (m/s)
	model.conditional.eprior = Kumaraswamy(0.867,3.03) #Eccentricity (Kumaraswamy prior acts as beta distribution see Standing et al. (2022))
	model.conditional.wprior = Uniform(0,2*np.pi) #Argument of periastron
	model.conditional.phiprior = Uniform(0,2*np.pi) #Starting phase of orbit

	#Known object priors (used to set tight priors on the binary) Note: the known object can also be used for any previously detected planet (i.e. transiting)
	model.KO_Pprior = [Gaussian(19.658,0.01)] #Binary period (days)
	model.KO_Kprior = [Gaussian(58549,100)] #Binary semi-amplitude (m/s)
	model.KO_eprior = [Gaussian(0.448,0.01)] #Binary eccentricity
	model.KO_wprior = [Gaussian(1.23,0.01)] #Binary argument of periastron
	model.KO_wdotprior = [Gaussian(0,1000)] #Binary orbital precession (arcsec/yr)
	model.KO_phiprior = [Uniform(0,2*np.pi)] #Binary starting phase of orbit
	model.KO_qprior = [Gaussian(0.972,0.01)] #Binary mass ratio
	# model.KO_cosiprior = [] prior on cosin of inclination not necessary as the eclipsing flag is on (see below) so this is by default fixed to 0

	#Set the sampler parameters (change steps to set length of the run, change threads depending on machine, others are reasonable to leave)
	kwargs.setdefault('steps', 10000) #Number of steps to run
	kwargs.setdefault('num_threads', 4) #Number of threads (depends on number of cores available)
	kwargs.setdefault('num_particles', 2) #Number of MCMC particles to explore parameter space
	kwargs.setdefault('new_level_interval', 2000) #Number of sample steps taken before a new level is created
	kwargs.setdefault('save_interval', 500) #Number of steps taken before a sample is saved to the output (thinning param)    
	kwargs.setdefault('thread_steps', 50) #Number of steps each thread takes before communicating level info to other threads

	#Model settings only required for binary stars:
	model.star_mass = 1.2368 #Mass of the primary star (used for calculating the GR and Tidal corrections, not necessary for double-lined binaries
	model.tidal_correction = False #Set the model to calculate Post-Newtonian effects due to Tides (only important for very tight binaries)
	model.relativistic_correction = True #Set the model to calculate Post-Newtonian effects due to GR
	model.double_lined=True #True if double lined (SB2) binary
	model.eclipsing=True #Set this to True if you are fitting for an eclipsing binary (inc is then fixed to 90 deg)

	
	with chdir(here):
		if run:
			kima.run(model, **kwargs)
		if load:
			res = kima.load_results(model)
			return model, res

	return model

def percentile68_ranges(a, min=None, max=None):
    """
    Calculate the 16th and 84th percentiles of values in `a`, clipped between
    `min` and `max`.

             minus     median    plus     
        -------==========|==========------
        16%    |        68%        |    16%

    Returns:
        median (float):
            Median value of `a`.
        plus (float):
            The 84th percentile minus the median.
        minus (float):
            The median minus the 16th percentile.
    """
    if min is None and max is None:
        mask = np.ones_like(a, dtype=bool)
    elif min is None:
        mask = a < max
    elif max is None:
        mask = a > min
    else:
        mask = (a > min) & (a < max)
    lp, median, up = np.percentile(a[mask], [16, 50, 84])
    return (median, up - median, median - lp)


def percentile68_ranges_latex(a, min=None, max=None, collapse=True,
                              include_dollar=True):
    r"""
    Return a LaTeX-formatted string of the 68% range of values in `a`, clipped
    between `a` and `b`, in the form

    $ median ^{+ plus} _{- minus} $
    
    Args:
        collapse (bool, optional):
            If True and plus=minus, return $ median \pm plus $
        include_dollar (bool, optional):
            Whether to include dollar signs in the output, so it's a valid LaTeX
            math mode string
    """
    median, plus, minus = percentile68_ranges(a, min, max)
    dollar = '$' if include_dollar else ''
    if collapse:
        high = urepr.uformat(median, plus, 'L').split(r'\pm')[1]
        low = urepr.uformat(median, minus, 'L').split(r'\pm')[1]
        if high == low:
            return dollar + urepr.uformat(median, plus, 'L') + dollar
    return dollar + urepr.uformatul(median, plus, minus, 'L') + dollar


if __name__ == '__main__':
	model = TIC1729(run=True, steps=200000)
	res = kima.load_results(model)

