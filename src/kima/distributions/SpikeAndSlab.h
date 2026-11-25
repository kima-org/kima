#pragma once

#include <memory>
#include <stdexcept>
#include <cmath>
#include <limits>
#include "DNest4.h"

namespace DNest4
{

/**
 * Spike-and-slab prior: with probability p0 the value is exactly 0
 * (the spike), otherwise it is drawn from the `slab` distribution.
 *
 * The slab must be supported on x > 0 (cdf(0) = 0), e.g. an eccentricity prior.
 * The prior odds spike:slab are p0 : (1 - p0), independently of the slab.
 */
class SpikeAndSlab : public ContinuousDistribution
{
    public:
        double p0;
        std::shared_ptr<ContinuousDistribution> slab;

        SpikeAndSlab(double p0, std::shared_ptr<ContinuousDistribution> slab);

        double cdf(double x) const override;
        double cdf_inverse(double u) const override;
        // NOTE: not a density at x == 0 (see implementation)
        double log_pdf(double x) const override;

        // Needed for reversibility: a point in the spike has no unique u
        double perturb(double& x, RNG& rng) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "SpikeAndSlab(" << p0 << "; ";
            slab->print(out);
            out << ")";
            return out;
        }
};

} // namespace DNest4