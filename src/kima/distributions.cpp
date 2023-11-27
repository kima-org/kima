#include <nanobind/nanobind.h>
namespace nb = nanobind;
using namespace nb::literals;

#include "DNest4.h"


NB_MODULE(distributions, m)
{
    nb::class_<DNest4::RNG>(m, "RNG")
        .def(nb::init<unsigned int>(), "seed"_a)
        #
        .def("rand", &DNest4::RNG::rand)
        .def("rand_int", [](DNest4::RNG& rng, int N){ return rng.rand_int(N); });
        //.def("rand_int", [](DNest4::RNG& rng, int L, int U){ return rng.rand_int(L, U); });
    //
    nb::class_<DNest4::ContinuousDistribution>(m, "Distribution");

    // Cauchy.cpp
    nb::class_<DNest4::Cauchy, DNest4::ContinuousDistribution>(m, "Cauchy", "Cauchy distribution")
        .def(nb::init<double, double>(), "loc"_a, "scale"_a)
        .def("cdf", &DNest4::Cauchy::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::Cauchy::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::Cauchy::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`");
    nb::class_<DNest4::TruncatedCauchy, DNest4::ContinuousDistribution>(m, "TruncatedCauchy", "docs")
        .def(nb::init<double, double, double, double>(), "loc"_a, "scale"_a, "lower"_a, "upper"_a)
        .def("cdf", &DNest4::TruncatedCauchy::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::TruncatedCauchy::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::TruncatedCauchy::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`");

    // Exponential.cpp
    nb::class_<DNest4::Exponential, DNest4::ContinuousDistribution>(m, "Exponential", "Exponential distribution")
        .def(nb::init<double>(), "scale"_a)
        .def("cdf", &DNest4::Exponential::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::Exponential::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::Exponential::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`");

    nb::class_<DNest4::TruncatedExponential, DNest4::ContinuousDistribution>(m, "TruncatedExponential", "Exponential distribution truncated to [lower, upper]")
        .def(nb::init<double, double, double>(), "scale"_a, "lower"_a, "upper"_a)
        .def("cdf", &DNest4::TruncatedExponential::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::TruncatedExponential::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::TruncatedExponential::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`");
    
    // Fixed.cpp
    nb::class_<DNest4::Fixed, DNest4::ContinuousDistribution>(m, "Fixed", "'Fixed' distribution")
        .def(nb::init<double>(), "value"_a)
        .def("cdf", &DNest4::Fixed::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::Fixed::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::Fixed::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`");

    // Gaussian.cpp
    nb::class_<DNest4::Gaussian, DNest4::ContinuousDistribution>(m, "Gaussian", "Gaussian distribution")
        .def(nb::init<double, double>(), "loc"_a, "scale"_a)
        .def("cdf", &DNest4::Gaussian::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::Gaussian::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::Gaussian::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`");

    // Kumaraswamy.cpp
    nb::class_<DNest4::Kumaraswamy, DNest4::ContinuousDistribution>(m, "Kumaraswamy", "Kumaraswamy distribution (similar to a Beta distribution)")
        .def(nb::init<double, double>(), "a"_a, "b"_a)
        .def("cdf", &DNest4::Kumaraswamy::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::Kumaraswamy::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::Kumaraswamy::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`");

    // Laplace.cpp
    nb::class_<DNest4::Laplace, DNest4::ContinuousDistribution>(m, "Laplace", "Laplace distribution")
        .def(nb::init<double, double>(), "loc"_a, "scale"_a)
        .def("cdf", &DNest4::Laplace::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::Laplace::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::Laplace::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`");

    // LogUniform.cpp
    nb::class_<DNest4::LogUniform, DNest4::ContinuousDistribution>(m, "LogUniform", "LogUniform distribution (sometimes called reciprocal or Jeffrey's distribution)")
        .def(nb::init<double, double>(), "lower"_a, "upper"_a)
        .def("cdf", &DNest4::LogUniform::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::LogUniform::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::LogUniform::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`");

    nb::class_<DNest4::ModifiedLogUniform, DNest4::ContinuousDistribution>(m, "ModifiedLogUniform", "ModifiedLogUniform distribution")
        .def(nb::init<double, double>(), "knee"_a, "upper"_a)
        .def("cdf", &DNest4::ModifiedLogUniform::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::ModifiedLogUniform::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::ModifiedLogUniform::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`");

    // Rayleigh.cpp
    nb::class_<DNest4::Rayleigh, DNest4::ContinuousDistribution>(m, "Rayleigh", "Rayleigh distribution")
        .def(nb::init<double>(), "scale"_a)
        .def("cdf", &DNest4::Rayleigh::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::Rayleigh::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::Rayleigh::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`");

    nb::class_<DNest4::TruncatedRayleigh, DNest4::ContinuousDistribution>(m, "TruncatedRayleigh", "Rayleigh distribution truncated to [lower, upper]")
        .def(nb::init<double, double, double>(), "scale"_a, "lower"_a, "upper"_a)
        .def("cdf", &DNest4::TruncatedRayleigh::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::TruncatedRayleigh::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::TruncatedRayleigh::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`");

    // Triangular.cpp
    nb::class_<DNest4::Triangular, DNest4::ContinuousDistribution>(m, "Triangular", "Triangular distribution")
        .def(nb::init<double, double, double>(), "lower"_a, "center"_a, "upper"_a)
        .def("cdf", &DNest4::Triangular::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::Triangular::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::Triangular::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`");

    // Uniform.cpp
    nb::class_<DNest4::Uniform, DNest4::ContinuousDistribution>(m, "Uniform", "Uniform distribuion in [lower, upper]")
        .def(nb::init<double, double>(), "lower"_a, "upper"_a)
        .def("cdf", &DNest4::Uniform::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::Uniform::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::Uniform::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`");
    //
}