#pragma once

#include "gcem.hpp"
#include "DNest4.h"

namespace DNest4
{

class InverseMoment:public ContinuousDistribution
{
    private:
        const double log_sqrt_pi = log(sqrt(M_PI));

    public:
        double tau, kmax;
        double _C;

        InverseMoment(double tau=1.0, double kmax=10.0);

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "InverseMoment(" << tau << "; " << kmax << ")";
            return out;
        }
};

} // namespace DNest4

