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
    
    @property
    def loc(self) -> float:
        ...
    @loc.setter
    def loc(self, arg: float, /) -> None:
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
    
    @property
    def scale(self) -> float:
        ...
    @scale.setter
    def scale(self, arg: float, /) -> None:
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
    
    @property
    def val(self) -> float:
        ...
    @val.setter
    def val(self, arg: float, /) -> None:
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
    
    @property
    def loc(self) -> float:
        ...
    @loc.setter
    def loc(self, arg: float, /) -> None:
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
    
    @property
    def scale(self) -> float:
        ...
    @scale.setter
    def scale(self, arg: float, /) -> None:
        ...
    
class Kumaraswamy:
    """
    Kumaraswamy distribution (similar to a Beta distribution)
    """

    def __init__(self, a: float, b: float) -> None:
        ...
    
    @property
    def a(self) -> float:
        ...
    @a.setter
    def a(self, arg: float, /) -> None:
        ...
    
    @property
    def b(self) -> float:
        ...
    @b.setter
    def b(self, arg: float, /) -> None:
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
    
    @property
    def loc(self) -> float:
        ...
    @loc.setter
    def loc(self, arg: float, /) -> None:
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
    
    @property
    def scale(self) -> float:
        ...
    @scale.setter
    def scale(self, arg: float, /) -> None:
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
    
    @property
    def lower(self) -> float:
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
    @property
    def upper(self) -> float:
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
    
    @property
    def knee(self) -> float:
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
    
    @property
    def upper(self) -> float:
        ...
    
class RNG:
    """
    None
    """

    def __init__(self, seed: int) -> None:
        ...
    
    def rand(self) -> float:
        ...
    
    def rand_int(self, arg: int, /) -> int:
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
    
    @property
    def scale(self) -> float:
        ...
    @scale.setter
    def scale(self, arg: float, /) -> None:
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
    
    @property
    def center(self) -> float:
        ...
    @center.setter
    def center(self, arg: float, /) -> None:
        ...
    
    def logpdf(self, x: float) -> float:
        """
        Log of the probability density function evaluated at `x`
        """
        ...
    
    @property
    def lower(self) -> float:
        ...
    @lower.setter
    def lower(self, arg: float, /) -> None:
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
    @property
    def upper(self) -> float:
        ...
    @upper.setter
    def upper(self, arg: float, /) -> None:
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
    
    @property
    def loc(self) -> float:
        ...
    
    def logpdf(self, x: float) -> float:
        """
        Log of the probability density function evaluated at `x`
        """
        ...
    
    @property
    def lower(self) -> float:
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
    @property
    def scale(self) -> float:
        ...
    
    @property
    def upper(self) -> float:
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
    
    @property
    def lower(self) -> float:
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
    @property
    def scale(self) -> float:
        ...
    
    @property
    def upper(self) -> float:
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
    
    @property
    def lower(self) -> float:
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
    @property
    def scale(self) -> float:
        ...
    
    @property
    def upper(self) -> float:
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
    
    @property
    def lower(self) -> float:
        ...
    @lower.setter
    def lower(self, arg: float, /) -> None:
        ...
    
    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`
        """
        ...
    
    @property
    def upper(self) -> float:
        ...
    @upper.setter
    def upper(self, arg: float, /) -> None:
        ...
    
class UniformAngle:
    """
    Uniform distribuion in [0, 2*PI]
    """

    def __init__(self) -> None:
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
    
