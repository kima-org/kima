#pragma once

#include <cmath>
#include <iostream>
#include "kepler.h"

namespace postKep
{
    double period_correction(double p_obs, double wdot);
    double change_omega(double w, double wdot, double ti, double Tp);
    inline double semiamp(double M0, double M1, double P, double ecc);
    inline double f_M(double K,double M0, double M1, double P, double ecc);
    inline double f_dash_M(double K,double M0, double M1, double P, double ecc);
    inline double get_K2_v1(double K1, double M, double P, double ecc);
    inline double get_K2_v2(double K1, double M, double P, double ecc);
    inline double light_travel_time(double K1, double f, double w, double ecc);
    inline double transverse_doppler(double K1, double f, double ecc, double cosi);
    inline double gravitational_redshift(double K1, double K2, double f, double ecc, double cosi);
    inline double v_tide(double R1, double M1, double M2, double P, double f, double w);
    double post_Newtonian(double K1, double f, double ecc, double w, double P, double M1, double M2, double R1, bool GR, bool Tid);
    
    std::vector<double> keplerian_prec(const std::vector<double> &t, const double &P,
                                  const double &K, const double &ecc,
                                  const double &w, const double &wdot, const double &M0,
                                  const double &M0_epoch);
}

namespace MassConv
{
    double SemiAmp(double P, double ecc, double M0, double M1, double cosi);
    double SemiPhotPl(double P, double M0, double M1, double plx);
    double SemiPhotSt(double P, double M0, double M1, double plx, double eps);
}