#include "InverseMoment.h"
#include <stdexcept>
#include <cmath>
#include <limits>

namespace DNest4
{

InverseMoment::InverseMoment(double tau, double kmax) : tau(tau), kmax(kmax)
{
    if (tau <= 0.0)
        throw std::domain_error("InverseMoment distribution must have tau > 0");
    if (kmax <= 0.0)
        throw std::domain_error("InverseMoment distribution must have kmax > 0");
    // normalization constant
    _C = cdf(kmax);
}

double InverseMoment::cdf(double x) const
{
    if (x < 0.0)
        return 0.0;
    else if (x > kmax)
        return 1.0;
    else {
        double C = 0.5 - gcem::erf(sqrt(tau) / kmax) / 2.0;
        return (0.5 - gcem::erf(sqrt(tau) / x) / 2) / C;
    }
}

double InverseMoment::cdf_inverse(double x) const
{
    if( (x < 0.0) || (x > 1.0) )
        throw std::domain_error("Input to cdf_inverse must be in [0, 1].");
        
    double C = 0.5 - gcem::erf(sqrt(tau) / kmax) / 2;
    x = x * C;
    return sqrt(tau) / gcem::erf_inv(1 - 2*x);
}

double InverseMoment::log_pdf(double x) const
{
    if ( (x <= 0.0) || (x > kmax) )
        return -std::numeric_limits<double>::infinity();
    return 0.5 * log(tau) - log_sqrt_pi - log(x*x) - tau / (x*x);
}


} // namespace DNest4

