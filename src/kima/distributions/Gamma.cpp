#include "Gamma.h"
#include <stdexcept>
#include <cmath>
#include <limits>

namespace DNest4
{

Gamma::Gamma(double alpha, double theta) : alpha(alpha), theta(theta)
{
    if (alpha <= 0.0)
        throw std::domain_error("Gamma distribution must have alpha > 0");
    if (theta <= 0.0)
        throw std::domain_error("Gamma distribution must have theta > 0");
    // log of normalization constant
    _C = -std::lgamma(alpha) - alpha * std::log(theta);
}

double Gamma::cdf(double x) const
{
    if (x <= 0.0)
        return 0.0;
    return gcem::incomplete_gamma(alpha, x / theta);
}

double Gamma::cdf_inverse(double p) const
{
    if( (p < 0.0) || (p > 1.0) )
        throw std::domain_error("Gamma: input to cdf_inverse must be in [0, 1].");
    return theta * gcem::incomplete_gamma_inv(alpha, p);
}

double Gamma::log_pdf(double x) const
{
    if (x < 0.0)
        return -std::numeric_limits<double>::infinity();
    if (x == 0.0)
    {
        if (alpha < 1.0) 
            return std::numeric_limits<double>::infinity();
        if (alpha == 1.0)
            return -std::log(theta);
        return -std::numeric_limits<double>::infinity();
    }
    return (alpha - 1.0) * std::log(x) - x / theta + _C;
}


} // namespace DNest4

