#ifndef DNest4_Triangular
#define DNest4_Triangular

#include "ContinuousDistribution.h"
#include "../RNG.h"

namespace DNest4
{

/*
* Triangular distribution
*/
class Triangular:public ContinuousDistribution
{
    public:
        double lower, centre, upper;

        Triangular(double lower=0.0, double centre=0.0, double upper=1.0);

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "Triangular(" << lower << "; " << centre << "; " << upper << ")";
            return out;
        }
};



} // namespace DNest4

#endif
