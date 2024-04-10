#pragma once

#include "gcem.hpp"
#include "DNest4.h"

namespace DNest4
{

class InverseGamma:public ContinuousDistribution
{
    private:
        const double log_sqrt_pi = log(sqrt(M_PI));

    public:
        double alpha, beta;
        double _C;

        InverseGamma(double alpha, double beta);

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "InverseGamma(" << alpha << "; " << beta << ")";
            return out;
        }
};

} // namespace DNest4

