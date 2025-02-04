#include "InverseGamma.h"
#include <stdexcept>
#include <cmath>
#include <limits>

namespace DNest4
{

InverseGamma::InverseGamma(double alpha, double beta) : alpha(alpha), beta(beta)
{
    if (alpha <= 0.0)
        throw std::domain_error("InverseGamma distribution must have alpha > 0");
    if (beta <= 0.0)
        throw std::domain_error("InverseGamma distribution must have beta > 0");
    // normalization constant
    _C = log(pow(beta, alpha) / gcem::tgamma(alpha));
}

double InverseGamma::cdf(double x) const
{
    if (x <= 0.0)
        return 0.0;
    return 1.0 - gcem::incomplete_gamma(alpha, beta / x);
}

double InverseGamma::cdf_inverse(double x) const
{
    if( (x < 0.0) || (x > 1.0) )
        throw std::domain_error("Input to cdf_inverse must be in [0, 1].");
    return beta / gcem::incomplete_gamma_inv(alpha, 1.0 - x);       
}

double InverseGamma::log_pdf(double x) const
{
    if (x <= 0.0)
        return -std::numeric_limits<double>::infinity();
    return _C - (alpha + 1.0) * log(x) - beta / x;
}


} // namespace DNest4

