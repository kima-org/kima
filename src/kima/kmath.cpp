#include "kmath.h"

double mean(nb::ndarray<double, nb::ndim<1>> a)
{
    double sum = 0.0;
    // auto v = a.view();
    for (size_t i = 0; i < a.shape(0); ++i)
        sum += a(i);
    return sum / a.size();
}

double standard_deviation(nb::ndarray<double, nb::ndim<1>> a)
{
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

std::vector<double> gaussian(const std::vector<double> &x, double x0, double width)
{
    std::vector<double> y(x.size());
    for (size_t i = 0; i < x.size(); i++)
    {
        double term = (x[i] - x0) / width;
        y[i] = exp(-0.5 * term * term);
    }
    return y;
}

std::vector<double> boxcar(const std::vector<double> &x, double x0, double width)
{
    std::vector<double> y(x.size());
    for (size_t i = 0; i < x.size(); i++)
    {
        double term = (x[i] - x0) / width;
        y[i] = (abs(term) < 0.5) ? 1.0 : 0.0;
    }
    return y;
}

std::vector<double> plateau(const std::vector<double> &x, double x0, double width, double shape)
{
    std::vector<double> y(x.size());
    for (size_t i = 0; i < x.size(); i++)
    {
        y[i] = 1.0 / (1.0 + pow(2.0 * abs(x[i] - x0) / width, 2.0 * shape));
    }
    return y;
}

NB_MODULE(kmath, m)
{
    m.def("mean", &mean);
    m.def("std", &standard_deviation);

    m.def("gaussian", &gaussian);
    m.def("boxcar", &boxcar);
    m.def("plateau", &plateau);
}