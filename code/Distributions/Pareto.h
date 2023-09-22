#ifndef DNest4_Pareto
#define DNest4_Pareto

#include "ContinuousDistribution.h"
#include "../RNG.h"
#include <limits>

namespace DNest4
{

/*
* Pareto distributions
*/
class Pareto:public ContinuousDistribution
{
    private:

        // Location and scale parameter
        double min, alpha;

    public:
        Pareto(double _min=1.0, double _alpha=1.0);

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "Pareto(" << min << "; " << alpha << ")";
            return out;
        }
};

} // namespace DNest4

#endif

