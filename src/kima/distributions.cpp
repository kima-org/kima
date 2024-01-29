#include <sstream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>
namespace nb = nanobind;
using namespace nb::literals;

#include "DNest4.h"

// the types of objects in the distributions state (for pickling)
using _state_type = std::tuple<double, double, double, double>;


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
        .def_rw("loc", &DNest4::Cauchy::center)
        .def_rw("scale", &DNest4::Cauchy::width)
        .def("__repr__", [](const DNest4::Cauchy &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Cauchy::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::Cauchy::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::Cauchy::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`")
        // for pickling
        .def("__getstate__",
             [](const DNest4::Cauchy &d) { 
                return std::make_tuple(d.center, d.width, 0.0, 0.0); 
        })
        .def("__setstate__",
             [](DNest4::Cauchy &d, const _state_type &state) {
                new (&d) DNest4::Cauchy(std::get<0>(state), std::get<1>(state));
            }
        );

    nb::class_<DNest4::TruncatedCauchy, DNest4::ContinuousDistribution>(m, "TruncatedCauchy", "docs")
        .def(nb::init<double, double, double, double>(), "loc"_a, "scale"_a, "lower"_a, "upper"_a)
        .def_ro("loc", &DNest4::TruncatedCauchy::center)
        .def_ro("scale", &DNest4::TruncatedCauchy::width)
        .def_ro("lower", &DNest4::TruncatedCauchy::lower)
        .def_ro("upper", &DNest4::TruncatedCauchy::upper)
        .def("__repr__", [](const DNest4::TruncatedCauchy &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::TruncatedCauchy::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::TruncatedCauchy::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::TruncatedCauchy::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`")
        // for pickling
        .def("__getstate__",
             [](const DNest4::TruncatedCauchy &d) { 
                return std::make_tuple(d.center, d.width, d.lower, d.upper); 
        })
        .def("__setstate__",
             [](DNest4::TruncatedCauchy &d, const _state_type &state) {
                new (&d) DNest4::TruncatedCauchy(std::get<0>(state), std::get<1>(state), 
                                                 std::get<2>(state), std::get<3>(state));
            }
        );

    // Exponential.cpp
    nb::class_<DNest4::Exponential, DNest4::ContinuousDistribution>(m, "Exponential", "Exponential distribution")
        .def(nb::init<double>(), "scale"_a)
        .def("__repr__", [](const DNest4::Exponential &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Exponential::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::Exponential::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::Exponential::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`")
        // for pickling
        .def("__getstate__",
             [](const DNest4::Exponential &d) { 
                return std::make_tuple(d.scale, 0.0, 0.0, 0.0); 
        })
        .def("__setstate__",
             [](DNest4::Exponential &d, const _state_type &state) {
                new (&d) DNest4::Exponential(std::get<0>(state));
            }
        );

    nb::class_<DNest4::TruncatedExponential, DNest4::ContinuousDistribution>(m, "TruncatedExponential", "Exponential distribution truncated to [lower, upper]")
        .def(nb::init<double, double, double>(), "scale"_a, "lower"_a, "upper"_a)
        .def_ro("scale", &DNest4::TruncatedExponential::scale)
        .def_ro("lower", &DNest4::TruncatedExponential::lower)
        .def_ro("upper", &DNest4::TruncatedExponential::upper)
        .def("__repr__", [](const DNest4::TruncatedExponential &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::TruncatedExponential::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::TruncatedExponential::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::TruncatedExponential::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`")
        // for pickling
        .def("__getstate__",
             [](const DNest4::TruncatedExponential &d) { 
                return std::make_tuple(d.scale, d.lower, d.upper, 0.0); 
        })
        .def("__setstate__",
             [](DNest4::TruncatedExponential &d, const _state_type &state) {
                new (&d) DNest4::TruncatedExponential(std::get<0>(state), std::get<1>(state), std::get<2>(state));
            }
        );
    
    // Fixed.cpp
    nb::class_<DNest4::Fixed, DNest4::ContinuousDistribution>(m, "Fixed", "'Fixed' distribution")
        .def(nb::init<double>(), "value"_a)
        .def_rw("val", &DNest4::Fixed::val)
        .def("__repr__", [](const DNest4::Fixed &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Fixed::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::Fixed::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::Fixed::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`")
        // for pickling
        .def("__getstate__",
             [](const DNest4::Fixed &d) { 
                return std::make_tuple(d.val, 0.0, 0.0, 0.0); 
        })
        .def("__setstate__",
             [](DNest4::Fixed &d, const _state_type &state) {
                new (&d) DNest4::Fixed(std::get<0>(state));
            }
        );

    // Gaussian.cpp
    nb::class_<DNest4::Gaussian, DNest4::ContinuousDistribution>(m, "Gaussian", "Gaussian distribution")
        .def(nb::init<double, double>(), "loc"_a, "scale"_a)
        .def_rw("loc", &DNest4::Gaussian::center)
        .def_rw("scale", &DNest4::Gaussian::width)
        .def("__repr__", [](const DNest4::Gaussian &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Gaussian::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::Gaussian::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::Gaussian::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`")
        // for pickling
        .def("__getstate__",
             [](const DNest4::Gaussian &d) { 
                return std::make_tuple(d.center, d.width, 0.0, 0.0); 
        })
        .def("__setstate__",
             [](DNest4::Gaussian &d, const _state_type &state) {
                new (&d) DNest4::Gaussian(std::get<0>(state), std::get<1>(state));
            }
        );

    // Kumaraswamy.cpp
    nb::class_<DNest4::Kumaraswamy, DNest4::ContinuousDistribution>(m, "Kumaraswamy", "Kumaraswamy distribution (similar to a Beta distribution)")
        .def(nb::init<double, double>(), "a"_a, "b"_a)
        .def_rw("a", &DNest4::Kumaraswamy::a)
        .def_rw("b", &DNest4::Kumaraswamy::b)
        .def("__repr__", [](const DNest4::Kumaraswamy &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Kumaraswamy::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::Kumaraswamy::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::Kumaraswamy::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`")
        // for pickling
        .def("__getstate__",
             [](const DNest4::Kumaraswamy &d) { 
                return std::make_tuple(d.a, d.b, 0.0, 0.0); 
        })
        .def("__setstate__",
             [](DNest4::Kumaraswamy &d, const _state_type &state) {
                new (&d) DNest4::Kumaraswamy(std::get<0>(state), std::get<1>(state));
            }
        );

    // Laplace.cpp
    nb::class_<DNest4::Laplace, DNest4::ContinuousDistribution>(m, "Laplace", "Laplace distribution")
        .def(nb::init<double, double>(), "loc"_a, "scale"_a)
        .def_rw("loc", &DNest4::Laplace::center)
        .def_rw("scale", &DNest4::Laplace::width)
        .def("__repr__", [](const DNest4::Laplace &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Laplace::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::Laplace::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::Laplace::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`")
        // for pickling
        .def("__getstate__",
             [](const DNest4::Laplace &d) { 
                return std::make_tuple(d.center, d.width, 0.0, 0.0); 
        })
        .def("__setstate__",
             [](DNest4::Laplace &d, const _state_type &state) {
                new (&d) DNest4::Laplace(std::get<0>(state), std::get<1>(state));
            }
        );

    // LogUniform.cpp
    nb::class_<DNest4::LogUniform, DNest4::ContinuousDistribution>(m, "LogUniform", "LogUniform distribution (sometimes called reciprocal or Jeffrey's distribution)")
        .def(nb::init<double, double>(), "lower"_a, "upper"_a)
        .def_ro("lower", &DNest4::LogUniform::lower)
        .def_ro("upper", &DNest4::LogUniform::upper)
        .def("__repr__", [](const DNest4::LogUniform &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::LogUniform::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::LogUniform::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::LogUniform::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`")
        // for pickling
        .def("__getstate__",
             [](const DNest4::LogUniform &d) { 
                return std::make_tuple(d.lower, d.upper, 0.0, 0.0); 
        })
        .def("__setstate__",
             [](DNest4::LogUniform &d, const _state_type &state) {
                new (&d) DNest4::LogUniform(std::get<0>(state), std::get<1>(state));
            }
        );

    nb::class_<DNest4::ModifiedLogUniform, DNest4::ContinuousDistribution>(m, "ModifiedLogUniform", "ModifiedLogUniform distribution")
        .def(nb::init<double, double>(), "knee"_a, "upper"_a)
        .def_ro("knee", &DNest4::ModifiedLogUniform::knee)
        .def_ro("upper", &DNest4::ModifiedLogUniform::upper)
        .def("__repr__", [](const DNest4::ModifiedLogUniform &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::ModifiedLogUniform::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::ModifiedLogUniform::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::ModifiedLogUniform::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`")
        // for pickling
        .def("__getstate__",
             [](const DNest4::ModifiedLogUniform &d) { 
                return std::make_tuple(d.knee, d.upper, 0.0, 0.0); 
        })
        .def("__setstate__",
             [](DNest4::ModifiedLogUniform &d, const _state_type &state) {
                new (&d) DNest4::ModifiedLogUniform(std::get<0>(state), std::get<1>(state));
            }
        );

    // Rayleigh.cpp
    nb::class_<DNest4::Rayleigh, DNest4::ContinuousDistribution>(m, "Rayleigh", "Rayleigh distribution")
        .def(nb::init<double>(), "scale"_a)
        .def_rw("scale", &DNest4::Rayleigh::scale)
        .def("__repr__", [](const DNest4::Rayleigh &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Rayleigh::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::Rayleigh::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::Rayleigh::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`")
        // for pickling
        .def("__getstate__",
             [](const DNest4::Rayleigh &d) { 
                return std::make_tuple(d.scale, 0.0, 0.0, 0.0); 
        })
        .def("__setstate__",
             [](DNest4::Rayleigh &d, const _state_type &state) {
                new (&d) DNest4::Rayleigh(std::get<0>(state));
            }
        );

    nb::class_<DNest4::TruncatedRayleigh, DNest4::ContinuousDistribution>(m, "TruncatedRayleigh", "Rayleigh distribution truncated to [lower, upper]")
        .def(nb::init<double, double, double>(), "scale"_a, "lower"_a, "upper"_a)
        .def_ro("scale", &DNest4::TruncatedRayleigh::scale)
        .def_ro("lower", &DNest4::TruncatedRayleigh::lower)
        .def_ro("upper", &DNest4::TruncatedRayleigh::upper)
        .def("__repr__", [](const DNest4::TruncatedRayleigh &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::TruncatedRayleigh::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::TruncatedRayleigh::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::TruncatedRayleigh::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`")
        // for pickling
        .def("__getstate__",
             [](const DNest4::TruncatedRayleigh &d) { 
                return std::make_tuple(d.scale, d.lower, d.upper, 0.0); 
        })
        .def("__setstate__",
             [](DNest4::TruncatedRayleigh &d, const _state_type &state) {
                new (&d) DNest4::TruncatedRayleigh(std::get<0>(state), std::get<1>(state), std::get<2>(state));
            }
        );

    // Triangular.cpp
    nb::class_<DNest4::Triangular, DNest4::ContinuousDistribution>(m, "Triangular", "Triangular distribution")
        .def(nb::init<double, double, double>(), "lower"_a, "center"_a, "upper"_a)
        .def_rw("lower", &DNest4::Triangular::lower)
        .def_rw("center", &DNest4::Triangular::centre)
        .def_rw("upper", &DNest4::Triangular::upper)
        .def("__repr__", [](const DNest4::Triangular &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Triangular::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::Triangular::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::Triangular::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`")
        // for pickling
        .def("__getstate__",
             [](const DNest4::Triangular &d) { 
                return std::make_tuple(d.lower, d.centre, d.upper, 0.0); 
        })
        .def("__setstate__",
             [](DNest4::Triangular &d, const _state_type &state) {
                new (&d) DNest4::Triangular(std::get<0>(state), std::get<1>(state), std::get<2>(state));
            }
        );

    // Uniform.cpp
    nb::class_<DNest4::Uniform, DNest4::ContinuousDistribution>(m, "Uniform", "Uniform distribuion in [lower, upper]")
        .def(nb::init<double, double>(), "lower"_a, "upper"_a)
        .def_rw("lower", &DNest4::Uniform::lower)
        .def_rw("upper", &DNest4::Uniform::upper)
        .def("__repr__", [](const DNest4::Uniform &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Uniform::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::Uniform::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::Uniform::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`")
        // for pickling
        .def("__getstate__",
             [](const DNest4::Uniform &d) { 
                return std::make_tuple(d.lower, d.upper, 0.0, 0.0); 
        })
        .def("__setstate__",
             [](DNest4::Uniform &d, const _state_type &state) {
                new (&d) DNest4::Uniform(std::get<0>(state), std::get<1>(state));
            }
        );
        
    
    nb::class_<DNest4::UniformAngle, DNest4::ContinuousDistribution>(m, "UniformAngle", "Uniform distribuion in [0, 2*PI]")
        .def(nb::init<>())
        .def("__repr__", [](const DNest4::UniformAngle &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::UniformAngle::cdf, "x"_a, "Cumulative distribution function evaluated at `x`")
        .def("ppf", &DNest4::UniformAngle::cdf_inverse, "q"_a, "Percent point function (inverse of cdf) evaluated at `q`")
        .def("logpdf", &DNest4::UniformAngle::log_pdf, "x"_a, "Log of the probability density function evaluated at `x`")
        // for pickling
        .def("__getstate__", [](const DNest4::UniformAngle &d) { return 0; })
        .def("__setstate__", [](DNest4::UniformAngle &d, const int &state) { new (&d) DNest4::UniformAngle(); });
    //
}