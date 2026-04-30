#include "ContinuousDistribution.h"
#include "../Utils.h"

namespace DNest4
{

double ContinuousDistribution::generate(RNG& rng) const
{
    return cdf_inverse(rng.rand());
}

double ContinuousDistribution::perturb(double& x, RNG& rng) const
{
    double log_pdf_x = log_pdf(x);
    x = cdf(x);
    x += rng.randh();
    wrap(x, 0.0, 1.0);
    x = cdf_inverse(x);
    double log_pdf_x_new = log_pdf(x);
    return log_pdf_x_new - log_pdf_x;
}

} // namespace DNest4

