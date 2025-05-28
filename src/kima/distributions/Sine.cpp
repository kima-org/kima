#include "Sine.h"
#include <stdexcept>
#include <cmath>
#include <limits>

namespace DNest4
{

Sine::Sine() {}

double Sine::cdf(double x) const
{
    if (x < 0.0 || x > M_PI)
        return 0.0;
    return 0.5 * (1.0 - cos(x));
}

double Sine::cdf_inverse(double x) const
{
    if( (x < 0.0) || (x > 1.0) )
        throw std::domain_error("Sine: input to cdf_inverse must be in [0, 1].");
    return acos(1.0 - 2.0 * x);
}

double Sine::log_pdf(double x) const
{
    if (x <= 0.0)
        return -std::numeric_limits<double>::infinity();
    return log(sin(x)) - log2;
}


} // namespace DNest4

