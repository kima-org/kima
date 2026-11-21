#pragma once

#include "DNest4.h"

namespace DNest4
{

/// The sine distribution (also called Gilbert's sine distribution) with support in [0, pi]
/// See Edwards (2000), https://doi.org/10.1111/1467-9639.00026 (paywalled)
/// or
/// https://stats.libretexts.org/Bookshelves/Probability_Theory/Probability_Mathematical_Statistics_and_Stochastic_Processes_(Siegrist)/05%3A_Special_Distributions/5.27%3A_The_Sine_Distribution
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

