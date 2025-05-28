#include "BivariateGaussian.h"
#include <stdexcept>
#include <cmath>
#include <limits>

namespace DNest4
{

BivariateGaussian::BivariateGaussian(double mean_x, double mean_y, double sigma_x, double sigma_y, double rho)
: mean_x(mean_x), mean_y(mean_y), sigma_x(sigma_x), sigma_y(sigma_y), rho(rho)
{
    if (sigma_x <= 0.0)
        throw std::domain_error("BivariateGaussian distribution must have sigma_x > 0");
    if (sigma_y <= 0.0)
        throw std::domain_error("BivariateGaussian distribution must have sigma_y > 0");
    
    omr2 = 1.0 - rho*rho;
    C = -log(2.0 * M_PI * sigma_x * sigma_y * sqrt(omr2));
}

double BivariateGaussian::log_pdf(double x, double y) const
{
    double dx = (x - mean_x) / sigma_x;
    double dy = (y - mean_y) / sigma_y;
    return C - 0.5 * (dx*dx + dy*dy - 2.0*rho*dx*dy) / omr2;
}


} // namespace DNest4

