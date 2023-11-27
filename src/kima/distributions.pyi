from typing import Any, Optional, overload, Typing, Sequence
from enum import Enum
import kima.distributions

class Cauchy:
    """
    Cauchy distribution
    """

    def __init__(self, loc: float, scale: float) -> None:
        ...
    
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`
        """
        ...
    
    def logpdf(self, x: float) -> float:
        """
        Log of the probability density function evaluated at `x`
        """
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
class Distribution:
    """
    None
    """

    def __init__(*args, **kwargs):
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...
    
class Exponential:
    """
    Exponential distribution
    """

    def __init__(self, scale: float) -> None:
        ...
    
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`
        """
        ...
    
    def logpdf(self, x: float) -> float:
        """
        Log of the probability density function evaluated at `x`
        """
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
class Fixed:
    """
    'Fixed' distribution
    """

    def __init__(self, value: float) -> None:
        ...
    
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`
        """
        ...
    
    def logpdf(self, x: float) -> float:
        """
        Log of the probability density function evaluated at `x`
        """
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
class Gaussian:
    """
    Gaussian distribution
    """

    def __init__(self, loc: float, scale: float) -> None:
        ...
    
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`
        """
        ...
    
    def logpdf(self, x: float) -> float:
        """
        Log of the probability density function evaluated at `x`
        """
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
class Kumaraswamy:
    """
    Kumaraswamy distribution (similar to a Beta distribution)
    """

    def __init__(self, a: float, b: float) -> None:
        ...
    
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`
        """
        ...
    
    def logpdf(self, x: float) -> float:
        """
        Log of the probability density function evaluated at `x`
        """
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
class Laplace:
    """
    Laplace distribution
    """

    def __init__(self, loc: float, scale: float) -> None:
        ...
    
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`
        """
        ...
    
    def logpdf(self, x: float) -> float:
        """
        Log of the probability density function evaluated at `x`
        """
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
class LogUniform:
    """
    LogUniform distribution (sometimes called reciprocal or Jeffrey's distribution)
    """

    def __init__(self, lower: float, upper: float) -> None:
        ...
    
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`
        """
        ...
    
    def logpdf(self, x: float) -> float:
        """
        Log of the probability density function evaluated at `x`
        """
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
class ModifiedLogUniform:
    """
    ModifiedLogUniform distribution
    """

    def __init__(self, knee: float, upper: float) -> None:
        ...
    
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`
        """
        ...
    
    def logpdf(self, x: float) -> float:
        """
        Log of the probability density function evaluated at `x`
        """
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
class RNG:
    """
    None
    """

    def __init__(self, seed: int) -> None:
        ...
    
    def rand(self) -> float:
        ...
    
    def rand_int(self, arg0: int, arg1: int, /) -> int:
        """
        rand_int(self, arg0: int, arg1: int, /) -> int
        """
        ...
    
    @overload
    def rand_int(self, arg: int, /) -> int:
        """
        rand_int(self, arg: int, /) -> int
        """
        ...
    
class Rayleigh:
    """
    Rayleigh distribution
    """

    def __init__(self, scale: float) -> None:
        ...
    
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`
        """
        ...
    
    def logpdf(self, x: float) -> float:
        """
        Log of the probability density function evaluated at `x`
        """
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
class Triangular:
    """
    Triangular distribution
    """

    def __init__(self, lower: float, center: float, upper: float) -> None:
        ...
    
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`
        """
        ...
    
    def logpdf(self, x: float) -> float:
        """
        Log of the probability density function evaluated at `x`
        """
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
class TruncatedCauchy:
    """
    docs
    """

    def __init__(self, loc: float, scale: float, lower: float, upper: float) -> None:
        ...
    
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`
        """
        ...
    
    def logpdf(self, x: float) -> float:
        """
        Log of the probability density function evaluated at `x`
        """
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
class TruncatedExponential:
    """
    Exponential distribution truncated to [lower, upper]
    """

    def __init__(self, scale: float, lower: float, upper: float) -> None:
        ...
    
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`
        """
        ...
    
    def logpdf(self, x: float) -> float:
        """
        Log of the probability density function evaluated at `x`
        """
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
class TruncatedRayleigh:
    """
    Rayleigh distribution truncated to [lower, upper]
    """

    def __init__(self, scale: float, lower: float, upper: float) -> None:
        ...
    
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`
        """
        ...
    
    def logpdf(self, x: float) -> float:
        """
        Log of the probability density function evaluated at `x`
        """
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
class Uniform:
    """
    Uniform distribuion in [lower, upper]
    """

    def __init__(self, lower: float, upper: float) -> None:
        ...
    
    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`
        """
        ...
    
    def logpdf(self, x: float) -> float:
        """
        Log of the probability density function evaluated at `x`
        """
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
