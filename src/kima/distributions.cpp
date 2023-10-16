#include <nanobind/nanobind.h>
namespace nb = nanobind;
using namespace nb::literals;

#include "DNest4.h"


NB_MODULE(distributions, m)
{
    nb::class_<DNest4::RNG>(m, "RNG")
        .def(nb::init<unsigned int>());
    //
    nb::class_<DNest4::ContinuousDistribution>(m, "Distribution");

    // Cauchy.cpp
    nb::class_<DNest4::Cauchy, DNest4::ContinuousDistribution>(m, "Cauchy", "docs")
        .def(nb::init<double, double>(), "loc"_a, "scale"_a)
        .def("cdf", &DNest4::Cauchy::cdf)
        .def("ppf", &DNest4::Cauchy::cdf_inverse)
        .def("logpdf", &DNest4::Cauchy::log_pdf);
    nb::class_<DNest4::TruncatedCauchy, DNest4::ContinuousDistribution>(m, "TruncatedCauchy", "docs")
        .def(nb::init<double, double, double, double>(), "loc"_a, "scale"_a, "lower"_a, "upper"_a)
        .def("cdf", &DNest4::TruncatedCauchy::cdf)
        .def("ppf", &DNest4::TruncatedCauchy::cdf_inverse)
        .def("logpdf", &DNest4::TruncatedCauchy::log_pdf);

    // Exponential.cpp
    nb::class_<DNest4::Exponential, DNest4::ContinuousDistribution>(m, "Exponential", "docs")
        .def(nb::init<double>(), "scale"_a)
        .def("cdf", &DNest4::Exponential::cdf)
        .def("ppf", &DNest4::Exponential::cdf_inverse)
        .def("logpdf", &DNest4::Exponential::log_pdf);
    nb::class_<DNest4::TruncatedExponential, DNest4::ContinuousDistribution>(m, "TruncatedExponential", "docs")
        .def(nb::init<double, double, double>(), "scale"_a, "lower"_a, "upper"_a)
        .def("cdf", &DNest4::TruncatedExponential::cdf)
        .def("ppf", &DNest4::TruncatedExponential::cdf_inverse)
        .def("logpdf", &DNest4::TruncatedExponential::log_pdf);
    
    // Fixed.cpp
    nb::class_<DNest4::Fixed, DNest4::ContinuousDistribution>(m, "Fixed", "docs")
        .def(nb::init<double>(), "value"_a)
        .def("cdf", &DNest4::Fixed::cdf)
        .def("ppf", &DNest4::Fixed::cdf_inverse)
        .def("logpdf", &DNest4::Fixed::log_pdf);

    // Gaussian.cpp
    nb::class_<DNest4::Gaussian, DNest4::ContinuousDistribution>(m, "Gaussian", "docs")
        .def(nb::init<double, double>(), "loc"_a, "scale"_a)
        .def("cdf", &DNest4::Gaussian::cdf)
        .def("ppf", &DNest4::Gaussian::cdf_inverse)
        .def("logpdf", &DNest4::Gaussian::log_pdf);

    // Kumaraswamy.cpp
    nb::class_<DNest4::Kumaraswamy, DNest4::ContinuousDistribution>(m, "Kumaraswamy")
        .def(nb::init<double, double>(), "a"_a, "b"_a)
        .def("cdf", &DNest4::Kumaraswamy::cdf)
        .def("ppf", &DNest4::Kumaraswamy::cdf_inverse)
        .def("logpdf", &DNest4::Kumaraswamy::log_pdf);

    // Laplace.cpp
    nb::class_<DNest4::Laplace, DNest4::ContinuousDistribution>(m, "Laplace")
        .def(nb::init<double, double>(), "loc"_a, "scale"_a)
        .def("cdf", &DNest4::Laplace::cdf)
        .def("ppf", &DNest4::Laplace::cdf_inverse)
        .def("logpdf", &DNest4::Laplace::log_pdf);

    // LogUniform.cpp
    nb::class_<DNest4::LogUniform, DNest4::ContinuousDistribution>(m, "LogUniform")
        .def(nb::init<double, double>(), "lower"_a, "upper"_a)
        .def("cdf", &DNest4::LogUniform::cdf)
        .def("ppf", &DNest4::LogUniform::cdf_inverse)
        .def("logpdf", &DNest4::LogUniform::log_pdf);
    nb::class_<DNest4::ModifiedLogUniform, DNest4::ContinuousDistribution>(m, "ModifiedLogUniform")
        .def(nb::init<double, double>(), "knee"_a, "upper"_a)
        .def("cdf", &DNest4::ModifiedLogUniform::cdf)
        .def("ppf", &DNest4::ModifiedLogUniform::cdf_inverse)
        .def("logpdf", &DNest4::ModifiedLogUniform::log_pdf);

    // Rayleigh.cpp
    nb::class_<DNest4::Rayleigh, DNest4::ContinuousDistribution>(m, "Rayleigh")
        .def(nb::init<double>(), "scale"_a)
        .def("cdf", &DNest4::Rayleigh::cdf)
        .def("ppf", &DNest4::Rayleigh::cdf_inverse)
        .def("logpdf", &DNest4::Rayleigh::log_pdf);
    nb::class_<DNest4::TruncatedRayleigh, DNest4::ContinuousDistribution>(m, "TruncatedRayleigh")
        .def(nb::init<double, double, double>(), "scale"_a, "lower"_a, "upper"_a)
        .def("cdf", &DNest4::TruncatedRayleigh::cdf)
        .def("ppf", &DNest4::TruncatedRayleigh::cdf_inverse)
        .def("logpdf", &DNest4::TruncatedRayleigh::log_pdf);

    // Triangular.cpp
    nb::class_<DNest4::Triangular, DNest4::ContinuousDistribution>(m, "Triangular", "docs")
        .def(nb::init<double, double, double>(), "lower"_a, "center"_a, "upper"_a)
        .def("cdf", &DNest4::Triangular::cdf)
        .def("ppf", &DNest4::Triangular::cdf_inverse)
        .def("logpdf", &DNest4::Triangular::log_pdf);

    // Uniform.cpp
    nb::class_<DNest4::Uniform, DNest4::ContinuousDistribution>(m, "Uniform", "docs")
        .def(nb::init<double, double>(), "lower"_a, "upper"_a)
        .def("cdf", &DNest4::Uniform::cdf)
        .def("ppf", &DNest4::Uniform::cdf_inverse)
        .def("logpdf", &DNest4::Uniform::log_pdf);
    //
}