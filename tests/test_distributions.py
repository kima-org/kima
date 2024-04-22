
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from scipy.stats import (cauchy, expon, norm, halfnorm, invgamma, laplace,
                         loguniform, rayleigh, triang, uniform)
from kumaraswamy import kumaraswamy


from kima.distributions import (Cauchy, Exponential, Fixed, Kumaraswamy,
                                Gaussian, HalfGaussian, InverseGamma, Laplace, LogUniform,
                                Rayleigh, Triangular, Uniform, UniformAngle)

def test_creation():
    c = Cauchy(1, 3.0)
    e = Exponential(2.0)
    f = Fixed(2.5)
    k = Kumaraswamy(0.8, 3)
    g = Gaussian(0, 2)
    u = Uniform(0, 10)

def test_repr():
    u = Uniform(0, 2)
    assert str(u) == 'Uniform(0; 2)'
    assert str(Gaussian(0, 1)) == 'Gaussian(0; 1)'


@pytest.fixture
def loc_scale():
    loc, scale = np.random.uniform(-5, 5), np.random.uniform(0, 10)
    return loc, scale

@pytest.fixture
def number():
    return np.random.uniform(-20, 20)

@pytest.fixture
def positive(n=1):
    def gen_positive(n):
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
    ]

    for dist1, dist2 in pairs:
        for r in dist1.rvs(N):
            assert_allclose(dist2.cdf(r), dist1.cdf(r))

    # Fixed
    assert_allclose(Fixed(number).cdf(number), 1.0)
    assert_allclose(Fixed(number).cdf(number*2), 0.0)
    assert_allclose(Fixed(number).cdf(number/10), 0.0)

    # is Kumaraswamy(1, 1) the same as Uniform(0, 1) ?
    assert_allclose(Kumaraswamy(1, 1).cdf(0.5), uniform(0, 1).cdf(0.5))
    # assert_allclose(Kumaraswamy(1, 1).cdf(1.5), uniform(0, 1).cdf(1.5))

    # ModifiedLogUniform
    # ...
        
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
    ]

    for dist1, dist2 in pairs:
        for r in np.random.rand(N):
            assert_allclose(dist2.ppf(r), dist1.ppf(r), err_msg=f'{dist1} != {dist2}')

    # # Fixed
    # assert_allclose(Fixed(number).cdf(number), 1.0)
    # assert_allclose(Fixed(number).cdf(number*2), 0.0)
    # assert_allclose(Fixed(number).cdf(number/10), 0.0)

    # # is Kumaraswamy(1, 1) the same as Uniform(0, 1) ?
    # assert_allclose(Kumaraswamy(1, 1).cdf(0.5), uniform(0, 1).cdf(0.5))
    # # assert_allclose(Kumaraswamy(1, 1).cdf(1.5), uniform(0, 1).cdf(1.5))

    # # ModifiedLogUniform
    # # ...
        
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
    assert_allclose(Fixed(number).logpdf(number), 0.0)
    assert_allclose(Fixed(number).logpdf(number*2), -np.inf)
    assert_allclose(Fixed(number).logpdf(number/10), -np.inf)
    
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
