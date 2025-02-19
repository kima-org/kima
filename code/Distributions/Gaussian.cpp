#include "Gaussian.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include "../Utils.h"
//#include <boost/math/special_functions/erf.hpp>

namespace DNest4
{

Gaussian::Gaussian(double center, double width) : center(center), width(width)
{
    if(width <= 0.0)
        throw std::domain_error("Gaussian distribution must have positive width.");
}

double Gaussian::cdf(double x) const
{
    return normal_cdf((x-center)/width);
}

double Gaussian::cdf_inverse(double x) const
{
    if (x < 0.0 || x > 1.0)
        throw std::domain_error("Input to cdf_inverse must be in [0, 1].");
    if (x == 0.0)
        return -std::numeric_limits<double>::infinity();
    if (x == 1.0)
        return std::numeric_limits<double>::infinity();
    return center + width*normal_inverse_cdf(x);
    //return center + width * sqrt(2) * boost::math::erf_inv(2*x - 1);
}

double Gaussian::log_pdf(double x) const
{
	double r = (x - center)/width;
    return -0.5*r*r - log(width) - _norm_pdf_logC;
}



HalfGaussian::HalfGaussian(double width) : width(width)
{
    if(width <= 0.0)
        throw std::domain_error("HalfGaussian distribution must have positive width.");
}

double HalfGaussian::cdf(double x) const
{
    return erf(x / width / sqrt(2.0));
}

double HalfGaussian::cdf_inverse(double x) const
{
    if(x < 0.0 || x > 1.0)
        throw std::domain_error("Input to cdf_inverse must be in [0, 1].");
    if(x == 0.0)
        return 0.0;
    if(x == 1.0)
        return std::numeric_limits<double>::infinity();
    return width * normal_inverse_cdf(0.5*(x+1.0));
    //return center + width * sqrt(2) * boost::math::erf_inv(2*x - 1);
}

double HalfGaussian::log_pdf(double x) const
{
	double r = x / width;
    return _halfnorm_pdf_logC - log(width) - 0.5*r*r;
}


TruncatedGaussian::TruncatedGaussian(double center, double width, double lower, double upper)
: center(center), width(width), lower(lower), upper(upper)
{
    if(width <= 0.0)
        throw std::domain_error("TruncatedGaussian distribution must have positive width.");
    if(lower >= upper)
        throw std::domain_error("TruncatedGaussian: lower bound should be less than upper bound.");
    alpha = (lower - center) / width;
    beta = (upper - center) / width;

    Z = normal_cdf(beta) - normal_cdf(alpha);
    // some combinations of center, width, lower, upper are numerically unstable
    // TODO: calculations in log to solve this?
    if (Z == 0.0)
        throw std::domain_error("TruncatedGaussian: numerically imprecise, check input parameters.");
}

double TruncatedGaussian::cdf(double x) const
{
    if (x < lower)
        return 0.0;
    if (x > upper)
        return 1.0;
    double r = (x - center) / width;
    return (normal_cdf(r) - normal_cdf(alpha)) / Z;
}

double TruncatedGaussian::cdf_inverse(double x) const
{
    if(x < 0.0 || x > 1.0)
        throw std::domain_error("Input to cdf_inverse must be in [0, 1].");
    if(x == 0.0)
        return lower;
    if(x == 1.0)
        return upper;
    double x_cdf = Z * x + normal_cdf(alpha);
    return center + width * normal_inverse_cdf(x_cdf);
}

double TruncatedGaussian::log_pdf(double x) const
{
    if( (x < lower) || (x > upper) )
        return -std::numeric_limits<double>::infinity();
	double r = (x - center)/width;
    return -0.5*r*r - log(width) - _norm_pdf_logC - log(Z);
}

} // namespace DNest4