#include "PowerLaw.h"

namespace DNest4
{ 

TruncatedPareto::TruncatedPareto(double min, double alpha, double lower, double upper)
:min(min), alpha(alpha), lower(lower), upper(upper)
{
    if(min <= 0.0 || alpha <= 0.0)
        throw std::domain_error("Pareto distribution must have `min` > 0 and `alpha` > 0");
    if(lower >= upper)
        throw std::domain_error("TruncatedPareto: lower bound should be less than upper bound.");
    // the original, untruncated, Pareto distribution
    unP = Pareto(min, alpha);
    c = unP.cdf(upper) - unP.cdf(lower);
}

double TruncatedPareto::cdf(double x) const
{
    double up = std::max(std::min(x,upper), lower);
    return (unP.cdf(up) - unP.cdf(lower)) / c;
}

double TruncatedPareto::cdf_inverse(double p) const
{
    if( (p < 0.0) || (p > 1.0) )
        throw std::domain_error("Input to cdf_inverse must be in [0, 1].");
    return unP.cdf_inverse(p * c + unP.cdf(lower));
}

double TruncatedPareto::log_pdf(double x) const
{
    if ( (x < lower) || (x > upper) )
        return -std::numeric_limits<double>::infinity();
    return unP.log_pdf(x) - std::log(c);
}

}