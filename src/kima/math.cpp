#include <vector>
#include <numeric>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
namespace nb = nanobind;
using namespace nb::literals;

double mean(nb::ndarray<double, nb::ndim<1>> a) {
    double sum = 0.0;
    // auto v = a.view();
    for (size_t i = 0; i < a.shape(0); ++i)
        sum += a(i);
    return sum / a.size();
}


double standard_deviation(nb::ndarray<double, nb::ndim<1>> a) {
    double sum = 0.0;
    // auto v = a.view();
    for (size_t i = 0; i < a.shape(0); ++i)
        sum += a(i);
    double mean = sum / a.size();

    double var = 0.0;
    for (size_t i = 0; i < a.shape(0); ++i)
        var += (a(i) - mean)*(a(i) - mean);
    return sqrt(var / a.size());
}

NB_MODULE(math, m)
{
    m.def("mean", &mean);
    m.def("std", &standard_deviation);
}