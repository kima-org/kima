
from typing import Callable
from functools import partial
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from scipy.stats import (cauchy, expon, norm, halfnorm, truncnorm, invgamma, laplace,
                         loguniform, rayleigh, triang, uniform, pareto)
from kumaraswamy import kumaraswamy


from kima.distributions import (Cauchy, Exponential, Fixed, Kumaraswamy,
                                Gaussian, HalfGaussian, TruncatedGaussian, 
                                InverseGamma, Laplace, LogUniform, ModifiedLogUniform, Pareto,
                                Rayleigh, Triangular, Uniform, UniformAngle)

def test_creation():
    c = Cauchy(1, 3.0)
    e = Exponential(2.0)
    f = Fixed(2.5)
    k = Kumaraswamy(0.8, 3)
    g = Gaussian(0, 2)
    hg = HalfGaussian(2)
    tg = TruncatedGaussian(0, 2, -1, 1)
    u = Uniform(0, 10)
    lu = LogUniform(0.1, 10)
    mlu = ModifiedLogUniform(1, 10)

def test_repr():
    u = Uniform(0, 2)
    assert str(u) == 'Uniform(0; 2)'
    assert str(Gaussian(0, 1)) == 'Gaussian(0; 1)'


@pytest.fixture
def loc_scale():
    loc, scale = np.random.uniform(-5, 5), np.random.uniform(0, 10)
    return loc, scale

@pytest.fixture
def number(n=1):
    def gen_number(n=1):
        return np.random.uniform(-20, 20, size=n)
    return gen_number

@pytest.fixture
def positive(n=1):
    def gen_positive(n=1):
        return np.random.uniform(0, 10, size=n)
    return gen_positive


def test_cdf(loc_scale, number, positive):
    N = 50
    loc, scale = loc_scale
    a, b = positive(2)

    pairs = [
        (cauchy(loc, scale), Cauchy(loc, scale)),
        (expon(scale=scale), Exponential(scale)),
        (norm(loc, scale), Gaussian(loc, scale)),
        (halfnorm(scale=scale), HalfGaussian(scale)),
        (invgamma(a, scale=b), InverseGamma(a, b)),
        (laplace(loc, scale), Laplace(loc, scale)),
        (loguniform(0.1*scale, 3*scale), LogUniform(0.1*scale, 3*scale)),
        (rayleigh(scale=scale), Rayleigh(scale)),
        (uniform(0.1*scale, 5*scale-0.1*scale), Uniform(0.1*scale, 5*scale)),
        (pareto(b=b), Pareto(1.0, b)),
    ]

    for dist1, dist2 in pairs:
        for r in dist1.rvs(N):
            assert_allclose(dist2.cdf(r), dist1.cdf(r))

    # Fixed
    n = number()
    assert_allclose(Fixed(n).cdf(n), 1.0)
    assert_allclose(Fixed(n).cdf(n*2), 0.0)
    assert_allclose(Fixed(n).cdf(n/10), 0.0)

    # is Kumaraswamy(1, 1) the same as Uniform(0, 1) ?
    assert_allclose(Kumaraswamy(1, 1).cdf(0.5), uniform(0, 1).cdf(0.5))
    # assert_allclose(Kumaraswamy(1, 1).cdf(1.5), uniform(0, 1).cdf(1.5))

    # ModifiedLogUniform
    # ...

    # truncated distributions
    a, b = np.sort(number(2))
    pairs = [
        (truncnorm((a - loc) / scale, (b - loc) / scale, loc=loc, scale=scale), TruncatedGaussian(loc, scale, a, b)),
    ]
    for dist1, dist2 in pairs:
        for r in dist1.rvs(N):
            assert_allclose(dist2.cdf(r), dist1.cdf(r),
                            err_msg=f"{dist2} - {r} ({a=}, {b=}, {loc=}, {scale=})")

    # Triangular
    for r in triang(c=0.75, scale=4).rvs(N):
        assert_allclose(Triangular(0, 3, 4).cdf(r), triang(c=0.75, scale=4).cdf(r))

    # is UniformAngle the same as Uniform(0, 2pi) ?
    assert_equal(UniformAngle().cdf(1.0), Uniform(0.0, 2*np.pi).cdf(1.0))



def test_inverse_cdf(loc_scale, number, positive):
    N = 50
    loc, scale = loc_scale
    a, b = positive(2)

    pairs = [
        (cauchy(loc, scale), Cauchy(loc, scale)),
        (expon(scale=scale), Exponential(scale)),
        (norm(loc, scale), Gaussian(loc, scale)),
        (halfnorm(scale=scale), HalfGaussian(scale)),
        (invgamma(a, scale=b), InverseGamma(a, b)),
        (laplace(loc, scale), Laplace(loc, scale)),
        (loguniform(0.1*scale, 3*scale), LogUniform(0.1*scale, 3*scale)),
        (rayleigh(scale=scale), Rayleigh(scale)),
        (uniform(0.1*scale, 5*scale-0.1*scale), Uniform(0.1*scale, 5*scale)),
        (pareto(b=b), Pareto(1.0, b)),
    ]

    for dist1, dist2 in pairs:
        for r in np.random.rand(N):
            assert_allclose(dist2.ppf(r), dist1.ppf(r), err_msg=f'{dist1} != {dist2}')

    # infinites
    assert_equal(Gaussian(loc, scale).ppf(0), -np.inf)
    assert_equal(Gaussian(loc, scale).ppf(1), np.inf)
    assert_equal(HalfGaussian(scale).ppf(1), np.inf)
    assert_equal(Cauchy(loc, scale).ppf(0), -np.inf)
    assert_equal(Cauchy(loc, scale).ppf(1), np.inf)

    # # Fixed
    # assert_allclose(Fixed(number).cdf(number), 1.0)
    # assert_allclose(Fixed(number).cdf(number*2), 0.0)
    # assert_allclose(Fixed(number).cdf(number/10), 0.0)

    # # is Kumaraswamy(1, 1) the same as Uniform(0, 1) ?
    # assert_allclose(Kumaraswamy(1, 1).cdf(0.5), uniform(0, 1).cdf(0.5))
    # # assert_allclose(Kumaraswamy(1, 1).cdf(1.5), uniform(0, 1).cdf(1.5))

    # # ModifiedLogUniform
    # # ...

    # truncated distributions
    a, b = np.sort(number(2))
    pairs = [
        (truncnorm((a - loc) / scale, (b - loc) / scale, loc=loc, scale=scale), TruncatedGaussian(loc, scale, a, b)),
    ]
    for dist1, dist2 in pairs:
        for r in np.random.rand(N):
            assert_allclose(dist2.ppf(r), dist1.ppf(r),
                            err_msg=f"{r} ({a=}, {b=}, {loc=}, {scale=})")

    # # Triangular
    # for r in triang(c=0.75, scale=4).rvs(N):
    #     assert_allclose(Triangular(0, 3, 4).cdf(r), triang(c=0.75, scale=4).cdf(r))

    # # is UniformAngle the same as Uniform(0, 2pi) ?
    # assert_equal(UniformAngle().cdf(1.0), Uniform(0.0, 2*np.pi).cdf(1.0))



def test_logpdf(loc_scale, number, positive):
    N = 50
    loc, scale = loc_scale
    a, b = positive(2)

    # Cauchy
    for r in cauchy(loc, scale).rvs(N):
        assert_allclose(Cauchy(loc, scale).logpdf(r), cauchy(loc, scale).logpdf(r))
    
    # Exponential
    for r in expon(scale=scale).rvs(N):
        assert_allclose(Exponential(scale).logpdf(r), expon(scale=scale).logpdf(r))

    # Fixed
    n = number()
    assert_allclose(Fixed(n).logpdf(n), 0.0)
    assert_allclose(Fixed(n).logpdf(n*2), -np.inf)
    assert_allclose(Fixed(n).logpdf(n/10), -np.inf)
    
    # Gaussian
    for r in norm(loc, scale).rvs(N):
        assert_allclose(Gaussian(loc, scale).logpdf(r), norm(loc, scale).logpdf(r))

    # HalfGaussian
    for r in halfnorm(scale=scale).rvs(N):
        assert_allclose(HalfGaussian(scale).logpdf(r), halfnorm(scale=scale).logpdf(r))

    # InverseGamma
    for r in invgamma(a, scale=b).rvs(N):
        assert_allclose(InverseGamma(a, b).logpdf(r), invgamma(a, scale=b).logpdf(r))

    # Kumaraswamy
    for r in kumaraswamy(a, b).rvs(N):
        assert_allclose(Kumaraswamy(a, b).logpdf(r), kumaraswamy(a, b).logpdf(r))
    # is Kumaraswamy(1, 1) the same as Uniform(0, 1) ?
    assert_allclose(Kumaraswamy(1, 1).logpdf(0.5), uniform(0, 1).logpdf(0.5))
    # assert_allclose(Kumaraswamy(1, 1).logpdf(1.5), uniform(0, 1).logpdf(1.5))

    # Laplace
    for r in laplace(loc, scale).rvs(N):
        assert_allclose(Laplace(loc, scale).logpdf(r), laplace(loc, scale).logpdf(r))

    # LogUniform
    for r in loguniform(0.1*scale, 3*scale).rvs(N):
        assert_allclose(LogUniform(0.1*scale, 3*scale).logpdf(r),
                        loguniform(0.1*scale, 3*scale).logpdf(r))

    # ModifiedLogUniform
    # ...

    # TruncatedGaussian
    a, b = np.sort(number(2))
    dist1 = truncnorm((a - loc) / scale, (b - loc) / scale, loc=loc, scale=scale)
    for r in dist1.rvs(N):
        assert_allclose(TruncatedGaussian(loc, scale, a, b).logpdf(r), dist1.logpdf(r))

    # Rayleigh
    for r in rayleigh(scale=scale).rvs(N):
        assert_allclose(Rayleigh(scale).logpdf(r), rayleigh(scale=scale).logpdf(r))

    # TruncatedRayleigh
        
    # Triangular
    for r in triang(c=0.75, scale=4).rvs(N):
        assert_allclose(Triangular(0, 3, 4).logpdf(r), triang(c=0.75, scale=4).logpdf(r))

    # Uniform
    for r in uniform(0.1*scale, 5*scale-0.1*scale).rvs(N):
        assert_allclose(Uniform(0.1*scale, 5*scale).logpdf(r), 
                        uniform(0.1*scale, 5*scale-0.1*scale).logpdf(r))

    # is UniformAngle the same as Uniform(0, 2pi) ?
    assert_equal(UniformAngle().logpdf(1.0), Uniform(0.0, 2*np.pi).logpdf(1.0))


def test_mixture():
    from kima.distributions import ExponentialRayleighMixture as ERM
    scale = np.random.uniform(0.1, 2)
    sigma = np.random.uniform(0.1, 2)
    C1exp = expon(scale=scale).cdf(1.0)
    C1ray = rayleigh(scale=sigma).cdf(1.0)
    # pdf
    x = np.random.rand()
    assert_allclose(ERM(1.0, scale, sigma).logpdf(x), np.log(expon(scale=scale).pdf(x) / C1exp))
    assert_allclose(ERM(0.0, scale, sigma).logpdf(x), np.log(rayleigh(scale=sigma).pdf(x) / C1ray))
    # cdf
    x = np.random.rand()
    assert_allclose(ERM(1.0, scale, sigma).cdf(x), expon(scale=scale).cdf(x) / C1exp)
    assert_allclose(ERM(0.0, scale, sigma).cdf(x), rayleigh(scale=sigma).cdf(x) / C1ray)
    # 
    assert_allclose(ERM(0.1, scale, sigma).cdf(x), 
                    0.1 * expon(scale=scale).cdf(x) / C1exp + 0.9 * rayleigh(scale=sigma).cdf(x) / C1ray)



