#ifndef DNest4_Uniform
#define DNest4_Uniform

#include "ContinuousDistribution.h"
#include "../RNG.h"

namespace DNest4
{

/*
* Uniform distribution
*/
class Uniform:public ContinuousDistribution
{
    public:
        double lower, upper;

        Uniform(double lower=0.0, double upper=1.0);

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "Uniform(" << lower << "; " << upper << ")";
            return out;
        }
};



/*
* UniformAngle distribution
*/
class UniformAngle:public ContinuousDistribution
{
    private:
        double TWOPI = 2 * M_PI;

    public:
        UniformAngle();

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "UniformAngle(0, 2*PI)";
            return out;
        }
};


} // namespace DNest4

#endif

