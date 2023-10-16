#pragma once

#include <vector>

#include "kepler.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;
using namespace nb::literals;

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

inline double ellpic_bulirsch(double n, double k);
inline double ellec(double k);
inline double ellk(double k);

std::vector<double> rsky(const std::vector<double>& time, double tc, double P, 
                         double a, double inc, double ecc, double omega);

std::vector<double> quadratic_ld(const std::vector<double>& ds,
                                 double c1, double c2, double p);
