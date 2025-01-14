#pragma once

#include <stdexcept>
#include <cmath>
#include <limits>
#include <vector>
#include "DNest4.h"
#include "utils.h"

const double inf = std::numeric_limits<double>::infinity();

namespace DNest4
{

class GaussianMixture:public ContinuousDistribution
{
    private:
        std::vector<DNest4::TruncatedGaussian> gaussians;
        double min_support_approx, max_support_approx;

    public:
        std::vector<double> means, sigmas;
        std::vector<double> weights;
        double lower, upper;
        size_t n;

        GaussianMixture(std::vector<double> means, std::vector<double> sigmas)
            : GaussianMixture(means, sigmas, std::vector<double>(means.size(), 1.0 / means.size()), -inf, inf) {};
        GaussianMixture(std::vector<double> means, std::vector<double> sigmas, double lower, double upper)
            : GaussianMixture(means, sigmas, std::vector<double>(means.size(), 1.0 / means.size()), lower, upper) {};
        GaussianMixture(std::vector<double> means, std::vector<double> sigmas, std::vector<double> weights, double lower, double upper);

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "GaussianMixture()";
            return out;
        }
};

} // namespace DNest4