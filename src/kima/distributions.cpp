#include <nanobind/nanobind.h>
namespace nb = nanobind;
using namespace nb::literals;

#include "DNest4.h"


NB_MODULE(distributions, m)
{
    nb::class_<DNest4::ContinuousDistribution>(m, "Distribution");
    // 
    nb::class_<DNest4::Uniform, DNest4::ContinuousDistribution>(m, "Uniform", "docs")
        .def(nb::init<>())
        .def(nb::init<double, double>(), "loc"_a, "scale"_a)
        .def("cdf", &DNest4::Uniform::cdf)
        .def("ppf", &DNest4::Uniform::cdf_inverse)
        .def("logpdf", &DNest4::Uniform::log_pdf);
    // 
    nb::class_<DNest4::Gaussian, DNest4::ContinuousDistribution>(m, "Gaussian", "docs")
        .def(nb::init<>())
        .def(nb::init<double, double>(), "loc"_a, "scale"_a)
        .def("cdf", &DNest4::Gaussian::cdf)
        .def("ppf", &DNest4::Gaussian::cdf_inverse)
        .def("logpdf", &DNest4::Gaussian::log_pdf);
    //
    nb::class_<DNest4::Kumaraswamy, DNest4::ContinuousDistribution>(m, "Kumaraswamy")
        .def(nb::init<>())
        .def(nb::init<double, double>(), "a"_a, "b"_a)
        .def("cdf", &DNest4::Kumaraswamy::cdf)
        .def("ppf", &DNest4::Kumaraswamy::cdf_inverse)
        .def("logpdf", &DNest4::Kumaraswamy::log_pdf);
    //
    nb::class_<DNest4::LogUniform, DNest4::ContinuousDistribution>(m, "LogUniform")
        .def(nb::init<>())
        .def(nb::init<double, double>(), "lower"_a, "upper"_a)
        .def("cdf", &DNest4::LogUniform::cdf)
        .def("ppf", &DNest4::LogUniform::cdf_inverse)
        .def("logpdf", &DNest4::LogUniform::log_pdf);
    //
    nb::class_<DNest4::ModifiedLogUniform, DNest4::ContinuousDistribution>(m, "ModifiedLogUniform")
        .def(nb::init<>())
        .def(nb::init<double, double>(), "knee"_a, "upper"_a)
        .def("cdf", &DNest4::ModifiedLogUniform::cdf)
        .def("ppf", &DNest4::ModifiedLogUniform::cdf_inverse)
        .def("logpdf", &DNest4::ModifiedLogUniform::log_pdf);
}