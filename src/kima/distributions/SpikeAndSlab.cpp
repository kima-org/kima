#include "SpikeAndSlab.h"

namespace DNest4
{

SpikeAndSlab::SpikeAndSlab(double p0, std::shared_ptr<ContinuousDistribution> slab)
: p0(p0), slab(slab)
{
    if (!slab)
        throw std::invalid_argument("SpikeAndSlab: must provide slab distribution");
    if (p0 < 0.0 || p0 > 1.0)
        throw std::domain_error("SpikeAndSlab distribution must have 0 <= p0 <= 1");
}

double SpikeAndSlab::cdf(double x) const
{
    if (x < 0.0)
        return 0.0;
    // right-continuous: the atom at 0 is included in cdf(0)
    return p0 + (1.0 - p0) * slab->cdf(x);
}

double SpikeAndSlab::cdf_inverse(double u) const
{
    if (u < 0.0 || u > 1.0)
        throw std::domain_error("Input to cdf_inverse must be in [0, 1].");

    if (u < p0)
        return 0.0;  // spike

    if (p0 >= 1.0)
        return 0.0;

    double v = (u - p0) / (1.0 - p0);
    v = std::min(std::max(v, 0.0), 1.0);
    return slab->cdf_inverse(v);
}

double SpikeAndSlab::log_pdf(double x) const
{
    if (x < 0.0)
        return -std::numeric_limits<double>::infinity();
    if (x == 0.0)
        return std::log(p0);  // probability *mass* of the spike, not density
    if (p0 >= 1.0)
        return -std::numeric_limits<double>::infinity();
    return std::log(1.0 - p0) + slab->log_pdf(x);
}

double SpikeAndSlab::perturb(double& x, RNG& rng) const
{
    // u is uniform in the spike if x == 0, otherwise it is cdf(x)
    double u = (x > 0.0) ? cdf(x) : p0 * rng.rand();
    u += rng.randh();
    DNest4::wrap(u, 0.0, 1.0);
    x = cdf_inverse(u);
    return 0.0;
}

} // namespace DNest4