from typing import Any, Optional, overload, Typing, Sequence
from enum import Enum
import kima.Sampler

def run(*args, **kwargs):
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
    DOC. Default is 10.0
    beta (int, optional):
    DOC. Default is 100.0,
    compression (int, optional):
    DOC. Default is exp(1.0)
    seed (int, optional):
    Random number seed value, or 0 to use current time. Default is 0.
    print_thin (int, optional):
    Thinning steps for terminal output. Default is 50.
    """
    ...

