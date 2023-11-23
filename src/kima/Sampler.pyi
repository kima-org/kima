from typing import Any, Optional, overload, Typing, Sequence
from enum import Enum
import kima.Sampler

def run(m: kima.BINARIESmodel.BINARIESmodel, steps: int = 100, num_threads: int = 1, num_particles: int = 1, new_level_interval: int = 2000, save_interval: int = 100, thread_steps: int = 100, max_num_levels: int = 0, lambda: float = 15.0, beta: float = 100.0, compression: float = 2.718281828459045, seed: int = 0, print_thin: int = 50) -> None:
    """
    Run the DNest4 sampler with the given model
    
    Args:
    m (model): the model
    steps (int, default=100): how many steps to run for
    """
    ...

@overload
def run(m: kima.RVmodel.RVmodel, steps: int = 100, num_threads: int = 1, num_particles: int = 1, new_level_interval: int = 2000, save_interval: int = 100, thread_steps: int = 10, max_num_levels: int = 0, lambda_: float = 10.0, beta: float = 100.0, compression: float = 2.718281828459045, seed: int = 0, print_thin: int = 50) -> None:
    """
    Run the DNest4 sampler with the given model
    
    Args:
    m (model): the model
    steps (int, default=100): how many steps to run for
    """
    ...

@overload
def run(m: kima.GPmodel.GPmodel, steps: int = 100, num_threads: int = 1, num_particles: int = 1, new_level_interval: int = 2000, save_interval: int = 100, thread_steps: int = 10, max_num_levels: int = 0, lambda_: float = 10.0, beta: float = 100.0, compression: float = 2.718281828459045, seed: int = 0, print_thin: int = 50) -> None:
    """
    Run the DNest4 sampler with the given model
    
    Args:
    m (model): the model
    steps (int, default=100): how many steps to run for
    """
    ...

@overload
def run(m: kima.RVFWHMmodel.RVFWHMmodel, steps: int = 100, num_threads: int = 1, num_particles: int = 1, new_level_interval: int = 2000, save_interval: int = 100, thread_steps: int = 10, max_num_levels: int = 0, lambda_: float = 10.0, beta: float = 100.0, compression: float = 2.718281828459045, seed: int = 0, print_thin: int = 50) -> None:
    """
    Run the DNest4 sampler with the given model
    
    Args:
    m (model): the model
    steps (int, default=100): how many steps to run for
    """
    ...

@overload
def run(m: kima.TRANSITmodel.TRANSITmodel, steps: int = 100, num_threads: int = 1, num_particles: int = 1, new_level_interval: int = 2000, save_interval: int = 100, thread_steps: int = 10, max_num_levels: int = 0, lambda_: float = 10.0, beta: float = 100.0, compression: float = 2.718281828459045, seed: int = 0, print_thin: int = 50) -> None:
    """
    Run the DNest4 sampler with the given model
    
    Args:
    m (model): the model
    steps (int, default=100): how many steps to run for
    """
    ...

@overload
def run(m: kima.OutlierRVmodel.OutlierRVmodel, steps: int = 100, num_threads: int = 1, num_particles: int = 1, new_level_interval: int = 2000, save_interval: int = 100, thread_steps: int = 10, max_num_levels: int = 0, lambda_: float = 10.0, beta: float = 100.0, compression: float = 2.718281828459045, seed: int = 0, print_thin: int = 50) -> None:
    """
    Run the DNest4 sampler with the given model
    
    Args:
    m (model): the model
    steps (int, default=100): how many steps to run for
    """
    ...

