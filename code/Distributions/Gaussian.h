#ifndef DNest4_Gaussian
#define DNest4_Gaussian

#include "ContinuousDistribution.h"
#include "../RNG.h"
#include "../Utils.h"

namespace DNest4
{

class Gaussian:public ContinuousDistribution
{
    private:
        double _norm_pdf_logC = log(sqrt(2*M_PI));

    public:
        // Location and scale parameter
        double center, width;

        Gaussian(double center=0.0, double width=1.0);

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "Gaussian(" << center << "; " << width << ")";
            return out;
        }
};

} // namespace DNest4

#endif

