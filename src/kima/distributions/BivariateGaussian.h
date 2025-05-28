#pragma once

#include "DNest4.h"

namespace DNest4
{

class BivariateGaussian:public ContinuousDistribution
{
    private:
        const double log_sqrt_pi = log(sqrt(M_PI));
        double omr2, C;

    public:
        double mean_x, mean_y, sigma_x, sigma_y, rho;

        BivariateGaussian(double mean_x, double mean_y, double sigma_x, double sigma_y)
            : BivariateGaussian(mean_x, mean_y, sigma_x, sigma_y, 0.0) {}

        BivariateGaussian(double mean_x, double mean_y, double sigma_x, double sigma_y, double rho);

        double cdf(double x) const override { return 0.0; };
        double cdf_inverse(double p) const override { return 0.0; };
        double log_pdf(double) const override { return 0.0; };
        double log_pdf(double x, double y) const;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "BivariateGaussian(" << mean_x << ", " << mean_y << ", " << sigma_x << ", " << sigma_y << ", " << rho << ")";
            return out;
        }
};

} // namespace DNest4

