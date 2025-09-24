// kima math utilities
#include <vector>
#include <numeric>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
namespace nb = nanobind;
using namespace nb::literals;

double mean(nb::ndarray<double, nb::ndim<1>> a);
double standard_deviation(nb::ndarray<double, nb::ndim<1>> a);

std::vector<double> gaussian(const std::vector<double> &x, double x0, double width);
std::vector<double> boxcar(const std::vector<double> &x, double x0, double width);
std::vector<double> plateau(const std::vector<double> &x, double x0, double width, double shape);