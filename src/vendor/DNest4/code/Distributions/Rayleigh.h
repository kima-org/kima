#ifndef DNest4_Rayleigh
#define DNest4_Rayleigh

#include "ContinuousDistribution.h"
#include "../RNG.h"

namespace DNest4
{

/*
* Rayleigh distributions
*/
class Rayleigh:public ContinuousDistribution
{
    public:
        double scale;

        Rayleigh(double scale=1.0);

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "Rayleigh(" << scale << ")";
            return out;
        }
};

class TruncatedRayleigh:public ContinuousDistribution
{
    private:
        double lcdf, ucdf;
        double tp, logtp;

    public:
        double scale;  // scale parameter
        double lower, upper;  // lower and upper truncation boundaries

        TruncatedRayleigh(double scale=1.0, double lower=0.0, double upper=1.0);

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "TruncatedRayleigh(" << scale << "; [" << lower << " , " << upper << "])";
            return out;
        }
};


} // namespace DNest4

#endif

