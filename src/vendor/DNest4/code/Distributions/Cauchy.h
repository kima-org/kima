#ifndef DNest4_Cauchy
#define DNest4_Cauchy

#include "ContinuousDistribution.h"
#include "../RNG.h"
#include <limits>

namespace DNest4
{

/*
* Cauchy distributions
*/
class Cauchy:public ContinuousDistribution
{
    public:
        // Location and scale parameter
        double center, width;

        Cauchy(double center=0.0, double width=1.0);

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "Cauchy(" << center << "; " << width << ")";
            return out;
        }
};

class TruncatedCauchy:public ContinuousDistribution
{
    private:
        Cauchy unC; // the original, untruncated, Cauchy distribution
        double c;

    public:
        double center, width; // Location and scale parameter
        double lower, upper; // truncation bounds

        TruncatedCauchy(double center=0.0, double width=1.0,
                        double lower=-std::numeric_limits<double>::infinity(),
                        double upper=std::numeric_limits<double>::infinity());

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double pdf(double x) const;
        double log_pdf(double x) const override;
        double rvs(RNG& rng) const;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "TruncatedCauchy(" << center << "; " << width << "; [" << lower << " , " << upper << "])";
            return out;
        }
};


} // namespace DNest4

#endif

