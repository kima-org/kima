#ifndef DNest4_Kumaraswamy
#define DNest4_Kumaraswamy

#include "ContinuousDistribution.h"
#include "../RNG.h"

namespace DNest4
{

/*
* Kumaraswamy distribution, a Beta-like distribution
* https://en.wikipedia.org/wiki/Kumaraswamy_distribution
*/
class Kumaraswamy:public ContinuousDistribution
{
    public:
        double a, b;

        Kumaraswamy(double a=1.0, double b=1.0);

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "Kumaraswamy(" << a << "; " << b << ")";
            return out;
        }
};

} // namespace DNest4

#endif

