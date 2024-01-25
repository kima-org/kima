#ifndef DNest4_Exponential
#define DNest4_Exponential

#include "ContinuousDistribution.h"
#include "../RNG.h"
#include <limits>

namespace DNest4
{

/*
* Exponential distribution
*/
class Exponential:public ContinuousDistribution
{
    public:
        double scale; // scale parameter

        Exponential(double scale=1.0);
        void setpars(double scale); // setter

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "Exponential(" << scale << ")";
            return out;
        }
};

/*
* truncated Exponential distribution
*/
class TruncatedExponential:public ContinuousDistribution
{
    private:
        Exponential unE; // the original, untruncated, Exponential distribution
        double c;

    public:
        double scale; // scale parameter
        double lower, upper; // truncation bounds

        TruncatedExponential(double scale=1.0, double lower=0., double upper=1./0.);
        void setpars(double scale); // setter

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;
};


} // namespace DNest4

#endif

