#pragma once

#include <stdexcept>
#include <cmath>
#include <limits>
#include "DNest4.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

namespace DNest4
{

class ExponentialRayleighMixture:public ContinuousDistribution
{
    private:
        const double log_sqrt_pi = log(sqrt(M_PI));
        Exponential Exp;
        Rayleigh Ray;
        double C1exp, C1ray;

    public:
        double weight, scale, sigma;

        ExponentialRayleighMixture(double weight, double scale, double sigma);

        double cdf(double x) const override;
        double cdf_inverse(double p) const override;
        double log_pdf(double x) const override;

        virtual std::ostream& print(std::ostream& out) const override
        {
            out << "ExponentialRayleighMixture(" << weight << "; " << scale << "; " << sigma << ")";
            return out;
        }
};



// modified from https://github.com/scipy/scipy/blob/main/scipy/optimize/Zeros/brenth.c
template<typename Functor>
double brenth(Functor f, double xa, double xb,
              double xtol=2e-12, double rtol=1e-15, int maxiter=100)
{
    double xpre = xa, xcur = xb;
    double xblk = 0., fpre, fcur, fblk = 0., spre = 0., scur = 0., sbis;
    /* the tolerance is 2*delta */
    double delta;
    double stry, dpre, dblk;
    int i;

    fpre = f(xpre);
    fcur = f(xcur);
    // solver_stats->funcalls = 2;
    if (fpre == 0) {
        // solver_stats->error_num = CONVERGED;
        return xpre;
    }
    if (fcur == 0) {
        // solver_stats->error_num = CONVERGED;
        return xcur;
    }
    if (std::signbit(fpre) == std::signbit(fcur)) {
        // solver_stats->error_num = SIGNERR;
        return 0.;
    }
    // solver_stats->iterations = 0;
    for (i = 0; i < maxiter; i++) {
        // solver_stats->iterations++;
        if (fpre != 0 && fcur != 0 &&
	    (std::signbit(fpre) != std::signbit(fcur))) {
            xblk = xpre;
            fblk = fpre;
            spre = scur = xcur - xpre;
        }
        if (fabs(fblk) < fabs(fcur)) {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;

            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        delta = (xtol + rtol*fabs(xcur))/2;
        sbis = (xblk - xcur)/2;
        if (fcur == 0 || fabs(sbis) < delta) {
            // solver_stats->error_num = CONVERGED;
            return xcur;
        }

        if (fabs(spre) > delta && fabs(fcur) < fabs(fpre)) {
            if (xpre == xblk) {
                /* interpolate */
                stry = -fcur*(xcur - xpre)/(fcur - fpre);
            }
            else {
                /* extrapolate */
                dpre = (fpre - fcur)/(xpre - xcur);
                dblk = (fblk - fcur)/(xblk - xcur);
                stry = -fcur*(fblk - fpre)/(fblk*dpre - fpre*dblk);
            }

            if (2*fabs(stry) < MIN(fabs(spre), 3*fabs(sbis) - delta)) {
                /* accept step */
                spre = scur;
                scur = stry;
            }
            else {
                /* bisect */
                spre = sbis;
                scur = sbis;
            }
        }
        else {
            /* bisect */
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;
        if (fabs(scur) > delta) {
            xcur += scur;
        }
        else {
            xcur += (sbis > 0 ? delta : -delta);
        }

        fcur = f(xcur);
        // solver_stats->funcalls++;
    }
    // solver_stats->error_num = CONVERR;
    return xcur;
}

} // namespace DNest4

