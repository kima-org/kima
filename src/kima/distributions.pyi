from collections.abc import Sequence
from typing import overload


class RNG:
    def __init__(self, seed: int) -> None: ...

    def rand(self) -> float:
        """Uniform(0, 1)"""

    def randh(self) -> float:
        """Brendon's heavy-tailed distribution"""

    def rand_int(self, arg: int, /) -> int:
        """IntegerUniform(0, N)"""

    @staticmethod
    def wrap(arg0: float, arg1: float, arg2: float, /) -> float:
        """Wrap a number to the range [a, b]"""

class Distribution:
    pass

class Cauchy(Distribution):
    def __init__(self, loc: float, scale: float) -> None:
        """
        Cauchy distribution

        Args:
            loc (float): location parameter
            scale (float): scale parameter
        """

    @property
    def loc(self) -> float:
        """location parameter"""

    @loc.setter
    def loc(self, arg: float, /) -> None: ...

    @property
    def scale(self) -> float:
        """scale parameter"""

    @scale.setter
    def scale(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class TruncatedCauchy(Distribution):
    def __init__(self, loc: float, scale: float, lower: float, upper: float) -> None:
        """
        Truncated Cauchy distribution

        Args:
            loc (float): location parameter
            scale (float): scale parameter
            lower (float): lower truncation limit
            upper (float): upper truncation limit
        """

    @property
    def loc(self) -> float:
        """location parameter"""

    @property
    def scale(self) -> float:
        """scale parameter"""

    @property
    def lower(self) -> float: ...

    @property
    def upper(self) -> float: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class Exponential(Distribution):
    def __init__(self, scale: float) -> None:
        """
        Exponential distribution

        Args:
            scale (float): scale parameter
        """

    @property
    def scale(self) -> float: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class TruncatedExponential(Distribution):
    def __init__(self, scale: float, lower: float, upper: float) -> None:
        """
        Truncated Exponential distribution

        Args:
            scale (float): scale parameter
            lower (float): lower truncation limit
            upper (float): upper truncation limit
        """

    @property
    def scale(self) -> float: ...

    @property
    def lower(self) -> float: ...

    @property
    def upper(self) -> float: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class Fixed(Distribution):
    def __init__(self, value: float) -> None:
        """
        A 'Fixed' distribution (akin to a Dirac delta distribution)

        Args:
            value (float): fixed value
        """

    @property
    def val(self) -> float:
        """fixed value of the parameter"""

    @val.setter
    def val(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class Gaussian(Distribution):
    def __init__(self, loc: float, scale: float) -> None:
        """
        Gaussian distribution

        Args:
            loc (float):
                location parameter (mean)
            scale (float):
                scale parameter (standard deviation)
        """

    @property
    def loc(self) -> float:
        """location parameter"""

    @loc.setter
    def loc(self, arg: float, /) -> None: ...

    @property
    def scale(self) -> float:
        """scale parameter"""

    @scale.setter
    def scale(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class HalfGaussian(Distribution):
    def __init__(self, scale: float) -> None:
        """
        Half-Gaussian distribution, with support in [0, inf)

        Args:
            scale (float):
                scale parameter (standard deviation)
        """

    @property
    def scale(self) -> float:
        """scale parameter"""

    @scale.setter
    def scale(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class TruncatedGaussian(Distribution):
    def __init__(self, loc: float, scale: float, lower: float, upper: float) -> None:
        """
        "Gaussian distribution truncated to [lower, upper] interval

        Args:
            loc (float):
                location parameter (mean)
            scale (float):
                scale parameter (standard deviation)
            lower (float):
                lower truncation limit
            upper (float):
                upper truncation limit
        """

    @property
    def loc(self) -> float:
        """location parameter"""

    @property
    def scale(self) -> float:
        """scale parameter"""

    @property
    def lower(self) -> float:
        """lower truncation limit"""

    @property
    def upper(self) -> float:
        """upper truncation limit"""

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class Kumaraswamy(Distribution):
    def __init__(self, a: float, b: float) -> None:
        """
        "Kumaraswamy distribution (similar to a Beta distribution)

        Args:
            a (float):
                first shape parameter
            b (float):
                second shape parameter
        """

    @property
    def a(self) -> float: ...

    @a.setter
    def a(self, arg: float, /) -> None: ...

    @property
    def b(self) -> float: ...

    @b.setter
    def b(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class Laplace(Distribution):
    def __init__(self, loc: float, scale: float) -> None:
        """
        "Laplace distribution

        Args:
            loc (float):
                location parameter
            scale (float):
                scale parameter
        """

    @property
    def loc(self) -> float:
        """location parameter"""

    @loc.setter
    def loc(self, arg: float, /) -> None: ...

    @property
    def scale(self) -> float:
        """scale parameter"""

    @scale.setter
    def scale(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class LogUniform(Distribution):
    def __init__(self, lower: float, upper: float) -> None:
        """
        "LogUniform distribution (sometimes called reciprocal or Jeffrey's distribution)

        Args:
            lower (float):
                lower limit (> 0)
            upper (float):
                upper limit (> lower)
        """

    @property
    def lower(self) -> float: ...

    @property
    def upper(self) -> float: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class ModifiedLogUniform(Distribution):
    def __init__(self, knee: float, upper: float) -> None:
        """
        "Modified Log-Uniform distribution, with support [0, upper]

        Args:
            knee (float):
                knee parameter
            upper (float):
                upper limit
        """

    @property
    def knee(self) -> float: ...

    @property
    def upper(self) -> float: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class Pareto(Distribution):
    """Pareto distribution"""

    def __init__(self, min: float, alpha: float) -> None: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

class TruncatedPareto(Distribution):
    """Pareto distribution truncated to [lower, upper] interval"""

    def __init__(self, min: float, alpha: float, lower: float, upper: float) -> None: ...

    @property
    def min(self) -> float:
        """location parameter"""

    @property
    def alpha(self) -> float:
        """scale parameter"""

    @property
    def lower(self) -> float: ...

    @property
    def upper(self) -> float: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class SingleTransitPeriodPrior(Distribution):
    """Prior for the orbital period when a single planet transit was observed"""

    def __init__(self, W: float, L: float, Pmax: float) -> None: ...

    @property
    def W(self) -> float:
        """Observational window"""

    @property
    def L(self) -> float:
        """Mid-transit time minus the start of the observational window"""

    @property
    def Pmax(self) -> float:
        """Maximum orbital period"""

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class Rayleigh(Distribution):
    def __init__(self, scale: float) -> None:
        """
        Rayleigh distribution

        Args:
            scale (float): scale parameter
        """

    @property
    def scale(self) -> float: ...

    @scale.setter
    def scale(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class TruncatedRayleigh(Distribution):
    """Rayleigh distribution truncated to [lower, upper] interval"""

    def __init__(self, scale: float, lower: float, upper: float) -> None: ...

    @property
    def scale(self) -> float: ...

    @property
    def lower(self) -> float: ...

    @property
    def upper(self) -> float: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class Triangular(Distribution):
    def __init__(self, lower: float, center: float, upper: float) -> None:
        """
        Triangular distribution

        Args:
            lower (float): lower bound
            center (float): center
            upper (float): upper bound
        """

    @property
    def lower(self) -> float: ...

    @lower.setter
    def lower(self, arg: float, /) -> None: ...

    @property
    def center(self) -> float: ...

    @center.setter
    def center(self, arg: float, /) -> None: ...

    @property
    def upper(self) -> float: ...

    @upper.setter
    def upper(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class Uniform(Distribution):
    def __init__(self, lower: float, upper: float) -> None:
        """
        Uniform distribution in [lower, upper]

        Args:
            lower (float): lower bound
            upper (float): upper bound
        """

    @property
    def lower(self) -> float:
        """lower bound"""

    @lower.setter
    def lower(self, arg: float, /) -> None: ...

    @property
    def upper(self) -> float:
        """upper bound"""

    @upper.setter
    def upper(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class UniformAngle(Distribution):
    def __init__(self) -> None:
        """Uniform distribution in [0, 2*PI]"""

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> int: ...

    def __setstate__(self, arg: int, /) -> None: ...

class InverseMoment(Distribution):
    """InverseMoment prior"""

    def __init__(self, tau: float, kmax: float) -> None: ...

    @property
    def tau(self) -> float: ...

    @tau.setter
    def tau(self, arg: float, /) -> None: ...

    @property
    def kmax(self) -> float: ...

    @kmax.setter
    def kmax(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class Gamma(Distribution):
    """Gamma distribution"""

    def __init__(self, alpha: float, theta: float) -> None:
        """Inverse gamma distribution"""

    @property
    def alpha(self) -> float:
        """Shape parameter α"""

    @alpha.setter
    def alpha(self, arg: float, /) -> None: ...

    @property
    def theta(self) -> float:
        """Scale parameter θ"""

    @theta.setter
    def theta(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class InverseGamma(Distribution):
    """Inverse gamma distribution"""

    def __init__(self, alpha: float, beta: float) -> None:
        """Inverse gamma distribution"""

    @property
    def alpha(self) -> float:
        """Shape parameter α"""

    @alpha.setter
    def alpha(self, arg: float, /) -> None: ...

    @property
    def beta(self) -> float:
        """scale parameter β"""

    @beta.setter
    def beta(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class ExponentialRayleighMixture(Distribution):
    def __init__(self, weight: float, scale: float, sigma: float) -> None:
        """
        Mixture of Exponential and Rayleigh distributions

        Args:
            weight (float): weight parameter
            scale (float): scale parameter
            sigma (float): sigma parameter
        """

    @property
    def weight(self) -> float: ...

    @weight.setter
    def weight(self, arg: float, /) -> None: ...

    @property
    def scale(self) -> float: ...

    @scale.setter
    def scale(self, arg: float, /) -> None: ...

    @property
    def sigma(self) -> float: ...

    @sigma.setter
    def sigma(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

class GaussianMixture(Distribution):
    """Mixture of Gaussian distributions"""

    @overload
    def __init__(self, means: Sequence[float], sigmas: Sequence[float]) -> None:
        """
        Instantiates a mixture of Gaussian distributions from lists of means and sigmas, with equal weights
        """

    @overload
    def __init__(self, means: Sequence[float], sigmas: Sequence[float], lower: float, upper: float) -> None:
        """
        Instantiates a mixture of Gaussian distributions from lists of means and sigmas, with equal weights, truncated to [lower, upper]
        """

    @overload
    def __init__(self, means: Sequence[float], sigmas: Sequence[float], weights: Sequence[float], lower: float, upper: float) -> None:
        """
        Instantiates a mixture of Gaussian distributions from lists of means, sigmas, and weights, truncated to [lower, upper]
        """

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

class BivariateGaussian(Distribution):
    """Bivariate Gaussian distribution for X and Y"""

    def __init__(self, mean_x: float, mean_y: float, sigma_x: float, sigma_y: float, rho: float) -> None:
        """docs"""

    @property
    def mean_x(self) -> float: ...

    @mean_x.setter
    def mean_x(self, arg: float, /) -> None: ...

    @property
    def mean_y(self) -> float: ...

    @mean_y.setter
    def mean_y(self, arg: float, /) -> None: ...

    @property
    def sigma_x(self) -> float: ...

    @sigma_x.setter
    def sigma_x(self, arg: float, /) -> None: ...

    @property
    def sigma_y(self) -> float: ...

    @sigma_y.setter
    def sigma_y(self, arg: float, /) -> None: ...

    @property
    def rho(self) -> float: ...

    @rho.setter
    def rho(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

    def logpdf(self, x: float, y: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

class Sine(Distribution):
    """
    The sine distribution (also called Gilbert's sine distribution) with support in [0, pi]
    """

    def __init__(self) -> None: ...

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """

    def __getstate__(self) -> tuple[float, float, float, float]: ...

    def __setstate__(self, arg: tuple[float, float, float, float], /) -> None: ...

class SpikeAndSlab(Distribution):
    """docs"""

    def __init__(self, arg0: float, arg1: Distribution, /) -> None:
        """A SpikeAndSlab distribution"""

    def __repr__(self) -> str: ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function evaluated at `x`

        Args:
            x (float): point at which to evaluate the CDF
        """

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse of cdf) evaluated at `q`

        Args:
            q (float): point at which to evaluate the PPF
        """

    def logpdf(self, x: float) -> float:
        """
        Logarithm of the probability density function evaluated at `x`

        Args:
            x (float): point at which to evaluate the PDF
        """
