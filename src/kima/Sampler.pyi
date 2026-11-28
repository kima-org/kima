from typing import overload

import kima.ApodizedRVmodel
import kima.BINARIESmodel
import kima.ETVmodel
import kima.GAIAmodel
import kima.GPmodel
import kima.OutlierRVmodel
import kima.RVFWHMRHKmodel
import kima.RVFWHMmodel
import kima.RVGAIAmodel
import kima.RVHGPMmodel
import kima.RVmodel
import kima.SPLEAFmodel
import kima.TRANSITmodel


@overload
def run(m: kima.RVmodel.RVmodel, steps: int = 100, num_threads: int = 1, num_particles: int = 1, new_level_interval: int = 2000, save_interval: int = 100, thread_steps: int = 10, max_num_levels: int = 0, lambda_: float = 10.0, beta: float = 100.0, compression: float = 2.718281828459045, target_loglikelihood: float = float('inf'), seed: int = 0, print_thin: int = 50, progress: bool = False) -> None: ...

@overload
def run(m: kima.GPmodel.GPmodel, steps: int = 100, num_threads: int = 1, num_particles: int = 1, new_level_interval: int = 2000, save_interval: int = 100, thread_steps: int = 10, max_num_levels: int = 0, lambda_: float = 10.0, beta: float = 100.0, compression: float = 2.718281828459045, target_loglikelihood: float = float('inf'), seed: int = 0, print_thin: int = 50, progress: bool = False) -> None: ...

@overload
def run(m: kima.RVFWHMmodel.RVFWHMmodel, steps: int = 100, num_threads: int = 1, num_particles: int = 1, new_level_interval: int = 2000, save_interval: int = 100, thread_steps: int = 10, max_num_levels: int = 0, lambda_: float = 10.0, beta: float = 100.0, compression: float = 2.718281828459045, target_loglikelihood: float = float('inf'), seed: int = 0, print_thin: int = 50, progress: bool = False) -> None: ...

@overload
def run(m: kima.TRANSITmodel.TRANSITmodel, steps: int = 100, num_threads: int = 1, num_particles: int = 1, new_level_interval: int = 2000, save_interval: int = 100, thread_steps: int = 10, max_num_levels: int = 0, lambda_: float = 10.0, beta: float = 100.0, compression: float = 2.718281828459045, target_loglikelihood: float = float('inf'), seed: int = 0, print_thin: int = 50, progress: bool = False) -> None: ...

@overload
def run(m: kima.OutlierRVmodel.OutlierRVmodel, steps: int = 100, num_threads: int = 1, num_particles: int = 1, new_level_interval: int = 2000, save_interval: int = 100, thread_steps: int = 10, max_num_levels: int = 0, lambda_: float = 10.0, beta: float = 100.0, compression: float = 2.718281828459045, target_loglikelihood: float = float('inf'), seed: int = 0, print_thin: int = 50, progress: bool = False) -> None: ...

@overload
def run(m: kima.BINARIESmodel.BINARIESmodel, steps: int = 100, num_threads: int = 1, num_particles: int = 1, new_level_interval: int = 2000, save_interval: int = 100, thread_steps: int = 10, max_num_levels: int = 0, lambda_: float = 10.0, beta: float = 100.0, compression: float = 2.718281828459045, target_loglikelihood: float = float('inf'), seed: int = 0, print_thin: int = 50, progress: bool = False) -> None: ...

@overload
def run(m: kima.GAIAmodel.GAIAmodel, steps: int = 100, num_threads: int = 1, num_particles: int = 1, new_level_interval: int = 2000, save_interval: int = 100, thread_steps: int = 10, max_num_levels: int = 0, lambda_: float = 10.0, beta: float = 100.0, compression: float = 2.718281828459045, target_loglikelihood: float = float('inf'), seed: int = 0, print_thin: int = 50, progress: bool = False) -> None: ...

@overload
def run(m: kima.RVGAIAmodel.RVGAIAmodel, steps: int = 100, num_threads: int = 1, num_particles: int = 1, new_level_interval: int = 2000, save_interval: int = 100, thread_steps: int = 10, max_num_levels: int = 0, lambda_: float = 10.0, beta: float = 100.0, compression: float = 2.718281828459045, target_loglikelihood: float = float('inf'), seed: int = 0, print_thin: int = 50, progress: bool = False) -> None: ...

@overload
def run(m: kima.RVFWHMRHKmodel.RVFWHMRHKmodel, steps: int = 100, num_threads: int = 1, num_particles: int = 1, new_level_interval: int = 2000, save_interval: int = 100, thread_steps: int = 10, max_num_levels: int = 0, lambda_: float = 10.0, beta: float = 100.0, compression: float = 2.718281828459045, target_loglikelihood: float = float('inf'), seed: int = 0, print_thin: int = 50, progress: bool = False) -> None: ...

@overload
def run(m: kima.ETVmodel.ETVmodel, steps: int = 100, num_threads: int = 1, num_particles: int = 1, new_level_interval: int = 2000, save_interval: int = 100, thread_steps: int = 10, max_num_levels: int = 0, lambda_: float = 10.0, beta: float = 100.0, compression: float = 2.718281828459045, target_loglikelihood: float = float('inf'), seed: int = 0, print_thin: int = 50, progress: bool = False) -> None: ...

@overload
def run(m: kima.SPLEAFmodel.SPLEAFmodel, steps: int = 100, num_threads: int = 1, num_particles: int = 1, new_level_interval: int = 2000, save_interval: int = 100, thread_steps: int = 10, max_num_levels: int = 0, lambda_: float = 10.0, beta: float = 100.0, compression: float = 2.718281828459045, target_loglikelihood: float = float('inf'), seed: int = 0, print_thin: int = 50, progress: bool = False) -> None: ...

@overload
def run(m: kima.ApodizedRVmodel.ApodizedRVmodel, steps: int = 100, num_threads: int = 1, num_particles: int = 1, new_level_interval: int = 2000, save_interval: int = 100, thread_steps: int = 10, max_num_levels: int = 0, lambda_: float = 10.0, beta: float = 100.0, compression: float = 2.718281828459045, target_loglikelihood: float = float('inf'), seed: int = 0, print_thin: int = 50, progress: bool = False) -> None: ...

@overload
def run(m: kima.RVHGPMmodel.RVHGPMmodel, steps: int = 100, num_threads: int = 1, num_particles: int = 1, new_level_interval: int = 2000, save_interval: int = 100, thread_steps: int = 10, max_num_levels: int = 0, lambda_: float = 10.0, beta: float = 100.0, compression: float = 2.718281828459045, target_loglikelihood: float = float('inf'), seed: int = 0, print_thin: int = 50, progress: bool = False) -> None:
    """
    Run the DNest4 sampler with the given model

    Args:
        m (RVmodel, GPmodel, ...):
            The model
        steps (int, optional):
            How many steps to run. Default is 100.
        num_threads (int, optional):
            How many threads to use for parallel processing. Default is 1.
        num_particles (int, optional):
            Number of MCMC particles. Default is 1.
        new_level_interval (int, optional):
            Number of steps required to create a new level. Default is 2000.
        save_interval (int, optional):
            Number of steps between saves. Default is 100.
        thread_steps (int, optional):
            Number of independent steps on each thread. Default is 10.
        max_num_levels (int, optional):
            Maximum number of levels, or 0 if it should be determined automatically. Default is 0.
        lambda_ (int, optional):
            Backtracking scale length, controls the degree to which particles are allowed to “backtrack” 
            down in level during the first stage of DNS when levels are being built. Default is 10.
        beta (int, optional):
            Equal weight enforcement, see Brewer et al. (2011). Default is 100.
        compression (int, optional):
            Each subsequent level is expected to be compressed by this factor. Default is exp(1).
        seed (int, optional):
            Random number seed value, or 0 to use current time. Default is 0.
        print_thin (int, optional):
            Thinning steps for terminal output. Default is 50.
    """
