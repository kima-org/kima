#include "ExponentialRayleighMixture.h"

namespace DNest4
{

ExponentialRayleighMixture::ExponentialRayleighMixture(double weight, double scale, double sigma)
: weight(weight), scale(scale), sigma(sigma)
{
    if (scale <= 0.0)
        throw std::domain_error("ExponentialRayleighMixture distribution must have scale > 0");
    if (sigma <= 0.0)
        throw std::domain_error("ExponentialRayleighMixture distribution must have sigma > 0");
    if (weight < 0.0 || weight > 1.0)
        throw std::domain_error("ExponentialRayleighMixture distribution must have 0 <= weight <= 1");
    
    Exp = Exponential(scale);
    Ray = Rayleigh(sigma);
    C1exp = 1.0 / Exp.cdf(1.0);
    C1ray = 1.0 / Ray.cdf(1.0);
}

double ExponentialRayleighMixture::cdf(double x) const
{
    if (x < 0.0)
        return 0.0;
    else if (x > 1.0)
        return 1.0;

    double _x = std::max(std::min(x, 1.0), 0.0);
    double cdf_exp = Exp.cdf(_x) * C1exp;
    double cdf_ray = Ray.cdf(_x) * C1ray;
    return weight * cdf_exp + (1.0 - weight) * cdf_ray;
}

double ExponentialRayleighMixture::cdf_inverse(double p) const
{
    if( (p < 0.0) || (p > 1.0) )
        throw std::domain_error("Input to cdf_inverse must be in [0, 1].");
    return brenth([p, this](double x){return cdf(x) - p;}, 0.0, 1.0);
}

double ExponentialRayleighMixture::log_pdf(double x) const
{
    if (x <= 0.0 || x >= 1.0)
        return -std::numeric_limits<double>::infinity();
    
    double pdf_exp = weight * exp(Exp.log_pdf(x)) * C1exp;
    double pdf_ray = (1.0 - weight) * exp(Ray.log_pdf(x)) * C1ray;
    return log(pdf_exp + pdf_ray);
}


} // namespace DNest4

