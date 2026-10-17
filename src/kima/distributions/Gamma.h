#pragma once

#include "gcem.hpp"
#include "DNest4.h"

namespace DNest4
{

class Gamma:public ContinuousDistribution
{
    private:
        const double log_sqrt_pi = log(sqrt(M_PI));
        // Log of the normalization constant 1 / (Gamma(alpha) * theta^alpha)
        double _C;

    public:
        double alpha, theta;

        Gamma(double alpha, double theta);

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "Gamma(" << alpha << "; " << theta << ")";
            return out;
        }
};

} // namespace DNest4

