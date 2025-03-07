#pragma once

#include <stdexcept>
#include <cmath>
#include <limits>
#include <algorithm>    // std::min, std::max
#include "DNest4.h"
#include "../utils.h"

namespace DNest4
{

class ExponentialRayleighMixture:public ContinuousDistribution
{
    private:
        const double log_sqrt_pi = log(sqrt(M_PI));
        Exponential Exp;
        Rayleigh Ray;
        double C1exp, C1ray;

    public:
        double weight, scale, sigma;

        ExponentialRayleighMixture(double weight, double scale, double sigma);

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "ExponentialRayleighMixture(" << weight << "; " << scale << "; " << sigma << ")";
            return out;
        }
};

} // namespace DNest4

