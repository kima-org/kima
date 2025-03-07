#pragma once

#include <ctime>
#include <mutex>
#include <cmath>
#include <memory>       // std::shared_ptr
#include <algorithm>    // std::min, std::max


inline bool approx_equal(double x, double y, double reltol = 1e-6)
{
    return std::fabs(x - y) < reltol * std::max(std::fabs(x), std::fabs(y));
}

inline std::tm localtime_xp(std::time_t timer)
{
    std::tm bt {};
#if defined(__unix__)
    localtime_r(&timer, &bt);
#elif defined(_MSC_VER)
    localtime_s(&bt, &timer);
#else
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    bt = *std::localtime(&timer);
#endif
    return bt;
}

// default = "YYYY-MM-DD HH:MM:SS"
inline std::string timestamp(const std::string& fmt = "%F %T")
{
    auto bt = localtime_xp(std::time(0));
    char buf[64];
    return {buf, std::strftime(buf, sizeof(buf), fmt.c_str(), &bt)};
}


// this creates an alias for std::make_shared
/**
 * @brief Assign a prior distribution.
 * 
 * This function defines, initializes, and assigns a prior distribution.
 * Possible distributions are ...
 * 
 * For example:
 * 
 * @code{.cpp}
 *          Cprior = make_prior<Uniform>(0, 1);
 * @endcode
 * 
 * @tparam T     ContinuousDistribution
 * @param args   Arguments for constructor of distribution
*/
template <class T, class... Args>
std::shared_ptr<T> make_prior(Args&&... args)
{
    return std::make_shared<T>(args...);
}


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

            if (2*fabs(stry) < std::min(fabs(spre), 3*fabs(sbis) - delta)) {
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
