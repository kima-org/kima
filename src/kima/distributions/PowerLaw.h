#pragma once

#include <stdexcept>
#include <cmath>
#include <limits>
#include <string>
#include "DNest4.h"
#include "../utils.h"

namespace DNest4
{

class TruncatedPareto : public DNest4::ContinuousDistribution
{
    private:
        DNest4::Pareto unP; // the original, untruncated, Pareto distribution
        double c;

    public:
        double min, alpha; // Location and scale parameter
        double lower, upper; // truncation bounds

        TruncatedPareto(double min=1.0, double alpha=1.0,
                        double lower=-std::numeric_limits<double>::infinity(), 
                        double upper=std::numeric_limits<double>::infinity());

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "TruncatedPareto(" << min << "; " << alpha << "; [" << lower << " , " << upper << "])";
            return out;
        }

};

class SingleTransitPeriodPrior : public TruncatedPareto
{
    public:
        double W, L;
        double Pmax;

        SingleTransitPeriodPrior(double W, double L, double Pmax) 
        : TruncatedPareto(1.0, 4.0/3.0, std::max(W-L, L), Pmax), W(W), L(L), Pmax(Pmax) {}

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "SingleTransitPeriodPrior(" << W << "; " << L << "; " << Pmax << ")";
            return out;
        }

};


} // namespace DNest4