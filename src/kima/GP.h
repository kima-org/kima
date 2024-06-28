#pragma once

#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

// for nanobind
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
// #include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
namespace nb = nanobind;
using namespace nb::literals;


using namespace Eigen;

/* The "standard" quasi-periodic kernel, see R&W2006 */
Eigen::MatrixXd QP(std::vector<double> &t, double eta1, double eta2, double eta3, double eta4);

/* The periodic kernel, oringinally created by MacKay, see R&W2006 */
Eigen::MatrixXd PER(std::vector<double> &t, double eta1, double eta3, double eta4);