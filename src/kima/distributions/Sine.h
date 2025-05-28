#pragma once

#include "DNest4.h"

namespace DNest4
{

class Sine:public ContinuousDistribution
{
    private:
        double log2 = log(2);

    public:
        Sine();

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "Sine()";
            return out;
        }
};

} // namespace DNest4

