#include "distributions.h"


auto CDF_DOC = R"D(
Cumulative distribution function evaluated at `x`

Args:
    x (float): point at which to evaluate the CDF
)D";

auto PPF_DOC = R"D(
Percent point function (inverse of cdf) evaluated at `q`

Args:
    q (float): point at which to evaluate the PPF
)D";

auto LOG_PDF_DOC = R"D(
Logarithm of the probability density function evaluated at `x`

Args:
    x (float): point at which to evaluate the PDF
)D";

NB_MODULE(distributions, m)
{
    nb::set_leak_warnings(false);
    nb::class_<DNest4::RNG>(m, "RNG")
        .def(nb::init<unsigned int>(), "seed"_a)
        // 
        .def("rand", &DNest4::RNG::rand, "Uniform(0, 1)")
        .def("rand_int", [](DNest4::RNG& rng, int N){ return rng.rand_int(N+1); }, "IntegerUniform(0, N)");
        //.def("rand_int", [](DNest4::RNG& rng, int L, int U){ return rng.rand_int(L, U); });
    //
    nb::class_<DNest4::ContinuousDistribution>(m, "Distribution");
    // nb::class_<DNest4::DiscreteDistribution>(m, "DiscreteDistribution");

    // Cauchy.cpp
    nb::class_<DNest4::Cauchy, DNest4::ContinuousDistribution>(m, "Cauchy")
        .def(nb::init<double, double>(), "loc"_a, "scale"_a, R"D(
        Cauchy distribution

        Args:
            loc (float): location parameter
            scale (float): scale parameter
        )D")
        .def_rw("loc", &DNest4::Cauchy::center, "location parameter")
        .def_rw("scale", &DNest4::Cauchy::width, "scale parameter")
        .def("__repr__", [](const DNest4::Cauchy &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Cauchy::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::Cauchy::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::Cauchy::log_pdf, "x"_a, LOG_PDF_DOC)
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

    nb::class_<DNest4::TruncatedCauchy, DNest4::ContinuousDistribution>(m, "TruncatedCauchy")
        .def(nb::init<double, double, double, double>(), "loc"_a, "scale"_a, "lower"_a, "upper"_a, R"D(
        Truncated Cauchy distribution

        Args:
            loc (float): location parameter
            scale (float): scale parameter
            lower (float): lower truncation limit
            upper (float): upper truncation limit
        )D")
        .def_ro("loc", &DNest4::TruncatedCauchy::center, "location parameter")
        .def_ro("scale", &DNest4::TruncatedCauchy::width, "scale parameter")
        .def_ro("lower", &DNest4::TruncatedCauchy::lower)
        .def_ro("upper", &DNest4::TruncatedCauchy::upper)
        .def("__repr__", [](const DNest4::TruncatedCauchy &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::TruncatedCauchy::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::TruncatedCauchy::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::TruncatedCauchy::log_pdf, "x"_a, LOG_PDF_DOC)
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
    nb::class_<DNest4::Exponential, DNest4::ContinuousDistribution>(m, "Exponential")
        .def(nb::init<double>(), "scale"_a, R"D(
        Exponential distribution

        Args:
            scale (float): scale parameter
        )D")
        .def_ro("scale", &DNest4::Exponential::scale)
        .def("__repr__", [](const DNest4::Exponential &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Exponential::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::Exponential::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::Exponential::log_pdf, "x"_a, LOG_PDF_DOC)
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

    nb::class_<DNest4::TruncatedExponential, DNest4::ContinuousDistribution>(m, "TruncatedExponential")
        .def(nb::init<double, double, double>(), "scale"_a, "lower"_a, "upper"_a, R"D(
        Truncated Exponential distribution

        Args:
            scale (float): scale parameter
            lower (float): lower truncation limit
            upper (float): upper truncation limit
        )D")
        .def_ro("scale", &DNest4::TruncatedExponential::scale)
        .def_ro("lower", &DNest4::TruncatedExponential::lower)
        .def_ro("upper", &DNest4::TruncatedExponential::upper)
        .def("__repr__", [](const DNest4::TruncatedExponential &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::TruncatedExponential::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::TruncatedExponential::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::TruncatedExponential::log_pdf, "x"_a, LOG_PDF_DOC)
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
    nb::class_<DNest4::Fixed, DNest4::ContinuousDistribution>(m, "Fixed")
        .def(nb::init<double>(), "value"_a, R"D(
        A 'Fixed' distribution (akin to a Dirac delta distribution)

        Args:
            value (float): fixed value
        )D")
        .def_rw("val", &DNest4::Fixed::val, "fixed value of the parameter")
        .def("__repr__", [](const DNest4::Fixed &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Fixed::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::Fixed::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::Fixed::log_pdf, "x"_a, LOG_PDF_DOC)
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
    nb::class_<DNest4::Gaussian, DNest4::ContinuousDistribution>(m, "Gaussian")
        .def(nb::init<double, double>(), "loc"_a, "scale"_a, R"D(
        Gaussian distribution

        Args:
            loc (float):
                location parameter (mean)
            scale (float):
                scale parameter (standard deviation)
        )D")
        .def_rw("loc", &DNest4::Gaussian::center, "location parameter")
        .def_rw("scale", &DNest4::Gaussian::width, "scale parameter")
        .def("__repr__", [](const DNest4::Gaussian &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Gaussian::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::Gaussian::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::Gaussian::log_pdf, "x"_a, LOG_PDF_DOC)
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

    nb::class_<DNest4::HalfGaussian, DNest4::ContinuousDistribution>(m, "HalfGaussian")
        .def(nb::init<double>(), "scale"_a, R"D(
        Half-Gaussian distribution, with support in [0, inf)

        Args:
            scale (float):
                scale parameter (standard deviation)
        )D")
        .def_rw("scale", &DNest4::HalfGaussian::width, "scale parameter")
        .def("__repr__", [](const DNest4::HalfGaussian &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::HalfGaussian::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::HalfGaussian::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::HalfGaussian::log_pdf, "x"_a, LOG_PDF_DOC)
        // for pickling
        .def("__getstate__",
             [](const DNest4::HalfGaussian &d) { 
                return std::make_tuple(d.width, 0.0, 0.0, 0.0); 
        })
        .def("__setstate__",
             [](DNest4::HalfGaussian &d, const _state_type &state) {
                new (&d) DNest4::HalfGaussian(std::get<0>(state));
            }
        );

    nb::class_<DNest4::TruncatedGaussian, DNest4::ContinuousDistribution>(m, "TruncatedGaussian")
        .def(nb::init<double, double, double, double>(), "loc"_a, "scale"_a, "lower"_a, "upper"_a, R"D(
        "Gaussian distribution truncated to [lower, upper] interval

        Args:
            loc (float):
                location parameter (mean)
            scale (float):
                scale parameter (standard deviation)
            lower (float):
                lower truncation limit
            upper (float):
                upper truncation limit
        )D")
        .def_ro("loc", &DNest4::TruncatedGaussian::center, "location parameter")
        .def_ro("scale", &DNest4::TruncatedGaussian::width, "scale parameter")
        .def_ro("lower", &DNest4::TruncatedGaussian::lower, "lower truncation limit")
        .def_ro("upper", &DNest4::TruncatedGaussian::upper, "upper truncation limit")
        .def("__repr__", [](const DNest4::TruncatedGaussian &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::TruncatedGaussian::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::TruncatedGaussian::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::TruncatedGaussian::log_pdf, "x"_a, LOG_PDF_DOC)
        // for pickling
        .def("__getstate__",
             [](const DNest4::TruncatedGaussian &d) { 
                return std::make_tuple(d.center, d.width, d.lower, d.upper); 
        })
        .def("__setstate__",
             [](DNest4::TruncatedGaussian &d, const _state_type &state) {
                new (&d) DNest4::TruncatedGaussian(std::get<0>(state), std::get<1>(state), 
                                                   std::get<2>(state), std::get<3>(state));
            }
        );

    // Kumaraswamy.cpp
    nb::class_<DNest4::Kumaraswamy, DNest4::ContinuousDistribution>(m, "Kumaraswamy")
        .def(nb::init<double, double>(), "a"_a, "b"_a, R"D(
        "Kumaraswamy distribution (similar to a Beta distribution)

        Args:
            a (float):
                first shape parameter
            b (float):
                second shape parameter
        )D")
        .def_rw("a", &DNest4::Kumaraswamy::a)
        .def_rw("b", &DNest4::Kumaraswamy::b)
        .def("__repr__", [](const DNest4::Kumaraswamy &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Kumaraswamy::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::Kumaraswamy::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::Kumaraswamy::log_pdf, "x"_a, LOG_PDF_DOC)
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
    nb::class_<DNest4::Laplace, DNest4::ContinuousDistribution>(m, "Laplace")
        .def(nb::init<double, double>(), "loc"_a, "scale"_a, R"D(
        "Laplace distribution

        Args:
            loc (float):
                location parameter
            scale (float):
                scale parameter
        )D")
        .def_rw("loc", &DNest4::Laplace::center, "location parameter")
        .def_rw("scale", &DNest4::Laplace::width, "scale parameter")
        .def("__repr__", [](const DNest4::Laplace &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Laplace::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::Laplace::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::Laplace::log_pdf, "x"_a, LOG_PDF_DOC)
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
    nb::class_<DNest4::LogUniform, DNest4::ContinuousDistribution>(m, "LogUniform")
        .def(nb::init<double, double>(), "lower"_a, "upper"_a, R"D(
        "LogUniform distribution (sometimes called reciprocal or Jeffrey's distribution)

        Args:
            lower (float):
                lower limit (> 0)
            upper (float):
                upper limit (> lower)
        )D")
        .def_ro("lower", &DNest4::LogUniform::lower)
        .def_ro("upper", &DNest4::LogUniform::upper)
        .def("__repr__", [](const DNest4::LogUniform &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::LogUniform::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::LogUniform::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::LogUniform::log_pdf, "x"_a, LOG_PDF_DOC)
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

    nb::class_<DNest4::ModifiedLogUniform, DNest4::ContinuousDistribution>(m, "ModifiedLogUniform")
        .def(nb::init<double, double>(), "knee"_a, "upper"_a, R"D(
        "Modified Log-Uniform distribution, with support [0, upper]

        Args:
            knee (float):
                knee parameter
            upper (float):
                upper limit
        )D")
        .def_ro("knee", &DNest4::ModifiedLogUniform::knee)
        .def_ro("upper", &DNest4::ModifiedLogUniform::upper)
        .def("__repr__", [](const DNest4::ModifiedLogUniform &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::ModifiedLogUniform::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::ModifiedLogUniform::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::ModifiedLogUniform::log_pdf, "x"_a, LOG_PDF_DOC)
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

    // Pareto.cpp
    nb::class_<DNest4::Pareto, DNest4::ContinuousDistribution>(m, "Pareto", "Pareto distribution")
        .def(nb::init<double, double>(), "min"_a, "alpha"_a)
        // .def_rw("min", &DNest4::Pareto::min)
        // .def_rw("alpha", &DNest4::Pareto::alpha)
        .def("__repr__", [](const DNest4::Pareto &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Pareto::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::Pareto::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::Pareto::log_pdf, "x"_a, LOG_PDF_DOC);
        // for pickling
        // .def("__getstate__",
        //      [](const DNest4::Pareto &d) { 
        //         return std::make_tuple(d.min, d.alpha, 0.0, 0.0);
        // })
        // .def("__setstate__",
        //      [](DNest4::Pareto &d, const _state_type &state) {
        //         new (&d) DNest4::Pareto(std::get<0>(state), std::get<1>(state));
        // })
    
    // PowerLaw.cpp
    nb::class_<DNest4::TruncatedPareto, DNest4::ContinuousDistribution>(m, "TruncatedPareto", "Pareto distribution truncated to [lower, upper] interval")
    .def(nb::init<double, double, double, double>(), "min"_a, "alpha"_a, "lower"_a, "upper"_a)
    .def_ro("min", &DNest4::TruncatedPareto::min, "location parameter")
    .def_ro("alpha", &DNest4::TruncatedPareto::alpha, "scale parameter")
    .def_ro("lower", &DNest4::TruncatedPareto::lower)
    .def_ro("upper", &DNest4::TruncatedPareto::upper)
    .def("__repr__", [](const DNest4::TruncatedPareto &d){ std::ostringstream out; d.print(out); return out.str(); })
    .def("cdf", &DNest4::TruncatedPareto::cdf, "x"_a, CDF_DOC)
    .def("ppf", &DNest4::TruncatedPareto::cdf_inverse, "q"_a, PPF_DOC)
    .def("logpdf", &DNest4::TruncatedPareto::log_pdf, "x"_a, LOG_PDF_DOC)
    // for pickling
    .def("__getstate__",
            [](const DNest4::TruncatedPareto &d) { 
            return std::make_tuple(d.min, d.alpha, d.lower, d.upper); 
    })
    .def("__setstate__",
            [](DNest4::TruncatedPareto &d, const _state_type &state) {
            new (&d) DNest4::TruncatedPareto(std::get<0>(state), std::get<1>(state), 
                                                std::get<2>(state), std::get<3>(state));
        }
    );

    nb::class_<DNest4::SingleTransitPeriodPrior, DNest4::ContinuousDistribution>(m, "SingleTransitPeriodPrior", "Prior for the orbital period when a single planet transit was observed")
    .def(nb::init<double, double, double>(), "W"_a, "L"_a, "Pmax"_a)
    .def_ro("W", &DNest4::SingleTransitPeriodPrior::W, "Observational window")
    .def_ro("L", &DNest4::SingleTransitPeriodPrior::L, "Mid-transit time minus the start of the observational window")
    .def_ro("Pmax", &DNest4::SingleTransitPeriodPrior::Pmax, "Maximum orbital period")
    .def("__repr__", [](const DNest4::SingleTransitPeriodPrior &d){ std::ostringstream out; d.print(out); return out.str(); })
    .def("cdf", &DNest4::SingleTransitPeriodPrior::cdf, "x"_a, CDF_DOC)
    .def("ppf", &DNest4::SingleTransitPeriodPrior::cdf_inverse, "q"_a, PPF_DOC)
    .def("logpdf", &DNest4::SingleTransitPeriodPrior::log_pdf, "x"_a, LOG_PDF_DOC)
    // for pickling
    .def("__getstate__",
            [](const DNest4::SingleTransitPeriodPrior &d) { 
            return std::make_tuple(d.W, d.L, d.Pmax); 
    })
    .def("__setstate__",
            [](DNest4::SingleTransitPeriodPrior &d, const _state_type &state) {
            new (&d) DNest4::SingleTransitPeriodPrior(std::get<0>(state), std::get<1>(state), std::get<2>(state));
        }
    );


    // Rayleigh.cpp
    nb::class_<DNest4::Rayleigh, DNest4::ContinuousDistribution>(m, "Rayleigh")
        .def(nb::init<double>(), "scale"_a, R"D(
        Rayleigh distribution

        Args:
            scale (float): scale parameter
        )D")
        .def_rw("scale", &DNest4::Rayleigh::scale)
        .def("__repr__", [](const DNest4::Rayleigh &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Rayleigh::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::Rayleigh::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::Rayleigh::log_pdf, "x"_a, LOG_PDF_DOC)
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

    nb::class_<DNest4::TruncatedRayleigh, DNest4::ContinuousDistribution>(m, "TruncatedRayleigh", "Rayleigh distribution truncated to [lower, upper] interval")
        .def(nb::init<double, double, double>(), "scale"_a, "lower"_a, "upper"_a)
        .def_ro("scale", &DNest4::TruncatedRayleigh::scale)
        .def_ro("lower", &DNest4::TruncatedRayleigh::lower)
        .def_ro("upper", &DNest4::TruncatedRayleigh::upper)
        .def("__repr__", [](const DNest4::TruncatedRayleigh &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::TruncatedRayleigh::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::TruncatedRayleigh::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::TruncatedRayleigh::log_pdf, "x"_a, LOG_PDF_DOC)
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
    nb::class_<DNest4::Triangular, DNest4::ContinuousDistribution>(m, "Triangular")
        .def(nb::init<double, double, double>(), "lower"_a, "center"_a, "upper"_a, R"D(
        Triangular distribution

        Args:
            lower (float): lower bound
            center (float): center
            upper (float): upper bound
        )D")
        .def_rw("lower", &DNest4::Triangular::lower)
        .def_rw("center", &DNest4::Triangular::centre)
        .def_rw("upper", &DNest4::Triangular::upper)
        .def("__repr__", [](const DNest4::Triangular &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Triangular::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::Triangular::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::Triangular::log_pdf, "x"_a, LOG_PDF_DOC)
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
    nb::class_<DNest4::Uniform, DNest4::ContinuousDistribution>(m, "Uniform")
        .def(nb::init<double, double>(), "lower"_a, "upper"_a, R"D(
        "Uniform distribution in [lower, upper]

        Args:
            lower (float): lower bound
            upper (float): upper bound
        )D")
        .def_rw("lower", &DNest4::Uniform::lower)
        .def_rw("upper", &DNest4::Uniform::upper)
        .def("__repr__", [](const DNest4::Uniform &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Uniform::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::Uniform::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::Uniform::log_pdf, "x"_a, LOG_PDF_DOC)
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
    
    nb::class_<DNest4::UniformAngle, DNest4::ContinuousDistribution>(m, "UniformAngle")
        .def(nb::init<>(), "Uniform distribution in [0, 2*PI]")
        .def("__repr__", [](const DNest4::UniformAngle &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::UniformAngle::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::UniformAngle::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::UniformAngle::log_pdf, "x"_a, LOG_PDF_DOC)
        // for pickling
        .def("__getstate__", [](const DNest4::UniformAngle &d) { return 0; })
        .def("__setstate__", [](DNest4::UniformAngle &d, const int &state) { new (&d) DNest4::UniformAngle(); });

    // nb::class_<DNest4::UniformInt, DNest4::DiscreteDistribution>(m, "UniformInt", "Uniform distribution in [lower, upper]")
    //     .def(nb::init<int, int>(), "lower"_a, "upper"_a)
    //     .def_rw("lower", &DNest4::UniformInt::lower)
    //     .def_rw("upper", &DNest4::UniformInt::upper)
    //     .def("__repr__", [](const DNest4::UniformInt &d){ std::ostringstream out; d.print(out); return out.str(); })
    //     .def("cdf", &DNest4::UniformInt::cdf, "x"_a, CDF_DOC)
    //     .def("ppf", &DNest4::UniformInt::cdf_inverse, "q"_a, PPF_DOC)
    //     .def("logpdf", &DNest4::UniformInt::log_pdf, "x"_a, LOG_PDF_DOC)
    //     // for pickling
    //     .def("__getstate__",
    //          [](const DNest4::UniformInt &d) { 
    //             return std::make_tuple(d.lower, d.upper, 0.0, 0.0); 
    //     })
    //     .def("__setstate__",
    //          [](DNest4::UniformInt &d, const _state_type &state) {
    //             new (&d) DNest4::UniformInt(std::get<0>(state), std::get<1>(state));
    //         }
    //     );


    // InverseMoment
    nb::class_<DNest4::InverseMoment, DNest4::ContinuousDistribution>(m, "InverseMoment", "InverseMoment prior")
        .def(nb::init<double, double>(), "tau"_a, "kmax"_a)
        .def_rw("tau", &DNest4::InverseMoment::tau)
        .def_rw("kmax", &DNest4::InverseMoment::kmax)
        .def("__repr__", [](const DNest4::InverseMoment &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::InverseMoment::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::InverseMoment::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::InverseMoment::log_pdf, "x"_a, LOG_PDF_DOC)
        // for pickling
        .def("__getstate__",
             [](const DNest4::InverseMoment &d) { 
                return std::make_tuple(d.tau, d.kmax, 0.0, 0.0); 
        })
        .def("__setstate__",
             [](DNest4::InverseMoment &d, const _state_type &state) {
                new (&d) DNest4::InverseMoment(std::get<0>(state), std::get<1>(state));
            }
        );


    // InverseGamma
    nb::class_<DNest4::InverseGamma, DNest4::ContinuousDistribution>(m, "InverseGamma", "Inverse gamma distribution")
        .def(nb::init<double, double>(), "alpha"_a, "beta"_a, "Inverse gamma distribution")
        .def_rw("alpha", &DNest4::InverseGamma::alpha, "Shape parameter α")
        .def_rw("beta", &DNest4::InverseGamma::beta, "scale parameter β")
        .def("__repr__", [](const DNest4::InverseGamma &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::InverseGamma::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::InverseGamma::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::InverseGamma::log_pdf, "x"_a, LOG_PDF_DOC)
        // for pickling
        .def("__getstate__",
             [](const DNest4::InverseGamma &d) { 
                return std::make_tuple(d.alpha, d.beta, 0.0, 0.0); 
        })
        .def("__setstate__",
             [](DNest4::InverseGamma &d, const _state_type &state) {
                new (&d) DNest4::InverseGamma(std::get<0>(state), std::get<1>(state));
            }
        );

    // ExponentialRayleighMixture
    nb::class_<DNest4::ExponentialRayleighMixture, DNest4::ContinuousDistribution>(m, "ExponentialRayleighMixture")
        .def(nb::init<double, double, double>(), "weight"_a, "scale"_a, "sigma"_a, R"D(
        Mixture of Exponential and Rayleigh distributions

        Args:
            weight (float): weight parameter
            scale (float): scale parameter
            sigma (float): sigma parameter
        )D")
        .def_rw("weight", &DNest4::ExponentialRayleighMixture::weight, "")
        .def_rw("scale", &DNest4::ExponentialRayleighMixture::scale, "")
        .def_rw("sigma", &DNest4::ExponentialRayleighMixture::sigma, "")
        .def("__repr__", [](const DNest4::ExponentialRayleighMixture &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::ExponentialRayleighMixture::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::ExponentialRayleighMixture::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::ExponentialRayleighMixture::log_pdf, "x"_a, LOG_PDF_DOC);
        // // for pickling
        // .def("__getstate__",
        //      [](const DNest4::ExponentialRayleighMixture &d) { 
        //         return std::make_tuple(d.alpha, d.beta, 0.0, 0.0); 
        // })
        // .def("__setstate__",
        //      [](DNest4::ExponentialRayleighMixture &d, const _state_type &state) {
        //         new (&d) DNest4::ExponentialRayleighMixture(std::get<0>(state), std::get<1>(state));
        //     }
        // );
    
    // GaussianMixture
    nb::class_<DNest4::GaussianMixture, DNest4::ContinuousDistribution>(m, "GaussianMixture", "Mixture of Gaussian distributions")
        .def(nb::init<std::vector<double>, std::vector<double>>(), "means"_a, "sigmas"_a, 
             "Instantiates a mixture of Gaussian distributions from lists of means and sigmas, with equal weights")
        .def(nb::init<std::vector<double>, std::vector<double>, double, double>(), "means"_a, "sigmas"_a, "lower"_a, "upper"_a,
             "Instantiates a mixture of Gaussian distributions from lists of means and sigmas, with equal weights, truncated to [lower, upper]")
        .def(nb::init<std::vector<double>, std::vector<double>, std::vector<double>, double, double>(), "means"_a, "sigmas"_a, "weights"_a, "lower"_a, "upper"_a,
             "Instantiates a mixture of Gaussian distributions from lists of means, sigmas, and weights, truncated to [lower, upper]")
        .def("__repr__", [](const DNest4::GaussianMixture &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::GaussianMixture::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::GaussianMixture::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::GaussianMixture::log_pdf, "x"_a, LOG_PDF_DOC);

    // BivariateGaussian
    nb::class_<DNest4::BivariateGaussian, DNest4::ContinuousDistribution>(m, "BivariateGaussian", 
        "Bivariate Gaussian distribution for X and Y")
        .def(nb::init<double, double, double, double, double>(), "mean_x"_a, "mean_y"_a, "sigma_x"_a, "sigma_y"_a, "rho"_a, "docs")
        .def_rw("mean_x", &DNest4::BivariateGaussian::mean_x, "")
        .def_rw("mean_y", &DNest4::BivariateGaussian::mean_y, "")
        .def_rw("sigma_x", &DNest4::BivariateGaussian::sigma_x, "")
        .def_rw("sigma_y", &DNest4::BivariateGaussian::sigma_y, "")
        .def_rw("rho", &DNest4::BivariateGaussian::rho, "")
        .def("__repr__", [](const DNest4::BivariateGaussian &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("logpdf", [](const DNest4::BivariateGaussian &d, double x, double y){ return d.log_pdf(x, y); }, "x"_a, "y"_a, LOG_PDF_DOC);

    // Sine
    nb::class_<DNest4::Sine, DNest4::ContinuousDistribution>(m, "Sine", "docs")
        .def(nb::init<>(), "A Sine distribution with support in [0, pi]")
        .def("__repr__", [](const DNest4::Sine &d){ std::ostringstream out; d.print(out); return out.str(); })
        .def("cdf", &DNest4::Sine::cdf, "x"_a, CDF_DOC)
        .def("ppf", &DNest4::Sine::cdf_inverse, "q"_a, PPF_DOC)
        .def("logpdf", &DNest4::Sine::log_pdf, "x"_a, LOG_PDF_DOC)
        // for pickling
        .def("__getstate__",
             [](const DNest4::Sine &d) { return std::make_tuple(0.0, 0.0, 0.0, 0.0); })
        .def("__setstate__",
             [](DNest4::Sine &d, const _state_type &state) { new (&d) DNest4::Sine(); });

}