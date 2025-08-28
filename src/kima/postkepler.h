#pragma once

#include <cmath>
#include <iostream>
#include "kepler.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>
namespace nb = nanobind;
using namespace nb::literals;

namespace postKep
{
    double period_correction(double p_obs, double wdot);
    double change_omega(double w, double wdot, double ti, double Tp);
    inline double semiamp(double M0, double M1, double P, double ecc);
    inline double f_M(double K,double M0, double M1, double P, double ecc);
    inline double f_dash_M(double K,double M0, double M1, double P, double ecc);
    inline double get_K2_v1(double K1, double M, double P, double ecc);
    inline double get_K2_v2(double K1, double M, double P, double ecc);
    inline double light_travel_time(double K1, double sinf, double cosf, double w, double ecc);
    inline double transverse_doppler(double K1, double sinf, double cosf, double ecc, double cosi);
    inline double gravitational_redshift(double K1, double K2, double sinf, double cosf, double ecc, double cosi);
    inline double v_tide(double R1, double M1, double M2, double P, double sinf, double cosf, double w);
    double post_Newtonian(double K1, double sinf, double cosf, double ecc, double w, double P, double M1, double M2, double R1, bool GR, bool Tid);
    double post_Newtonian_sb2(double K1, double K2, double sinf, double cosf, double ecc, double w, double P, double q, double R1, double R2, bool GR, bool Tid);
    
    std::vector<double> keplerian_prec(const std::vector<double> &t, const double &P,
                                  const double &K, const double &ecc,
                                  const double &w, const double &wdot, const double &M0,
                                  const double &M0_epoch, const double &cosi, const double &M1, const double &M2, 
                                  const double &R1, bool GR, bool Tid);
    std::tuple<std::vector<double>, std::vector<double>> keplerian_prec_sb2(const std::vector<double> &t, const double &P,
                                  const double &K, const double &q, const double &ecc,
                                  const double &w, const double &wdot, const double &M0,
                                  const double &M0_epoch, const double &cosi,
                                  const double &R1, const double &R2, bool GR, bool Tid);
}

namespace MassConv
{
    double SemiAmp(double P, double ecc, double M0, double M1, double cosi);
    double SemiPhotPl(double P, double M0, double M1, double plx);
    double SemiPhotSt(double P, double M0, double M1, double plx, double eps);
    double SemiPhotfromK(double P, double K, double ecc, double cosi, double plx);
}