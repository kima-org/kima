#ifndef DNest4_Laplace
#define DNest4_Laplace

#include "ContinuousDistribution.h"
#include "../RNG.h"
#include <limits>

namespace DNest4
{

/*
* Laplace distributions
*/
class Laplace:public ContinuousDistribution
{
    private:

    public:
        // Location and scale parameter
        double center, width;

        Laplace(double center=0.0, double width=1.0);
        // setter
        void setpars(double center, double width);

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "Laplace(" << center << "; " << width << ")";
            return out;
        }

        // Sign function
        static int sign(double x);
};

} // namespace DNest4

#endif

