#include "ConditionalPrior.h"

using namespace std;
using namespace DNest4;

RVConditionalPrior::RVConditionalPrior():hyperpriors(false)
{
    // if (hyperpriors){
    //     if (!log_muP_prior)
    //         /** 
    //          * By default, Cauchy prior centered on log(365 days), scale=1
    //          * for log(muP), with muP in days
    //          * truncated to (~-15.1, ~26.9)
    //         */
    //         log_muP_prior = make_shared<TruncatedCauchy>(log(365), 1., log(365)-21, log(365)+21);
    //     // TruncatedCauchy *log_muP_prior = new TruncatedCauchy(log(365), 1., log(365)-21, log(365)+21);

    //     /**
    //      * By default, uniform prior for wP
    //     */
    //     if (!wP_prior)
    //         wP_prior = make_shared<Uniform>(0.1, 3);
    //     // Uniform *wP_prior = new Uniform(0.1, 3.);
    

    //     /**
    //      * By default, Cauchy prior centered on log(1), scale=1
    //      * for log(muK), with muK in m/s
    //      * truncated to (-21, 21)
    //      * NOTE: we actually sample on muK itself, just the prior is for log(muK)
    //     */
    //     if (!log_muK_prior)
    //         log_muK_prior = make_shared<TruncatedCauchy>(0., 1., 0.-21, 0.+21);
    //     // TruncatedCauchy *log_muK_prior = new TruncatedCauchy(0., 1., 0.-21, 0.+21);

    //     Pprior = make_shared<Laplace>();
    //     Kprior = make_shared<Exponential>();
    // }

    if (!Pprior)
        Pprior = make_shared<LogUniform>(1.0, 1000.0);
    if (!Kprior)
        Kprior = make_shared<Uniform>(0.0, 100.0);
    if (!eprior)
        eprior = make_shared<Uniform>(0, 1);
    if (!phiprior)
        phiprior = make_shared<Uniform>(0, 2*M_PI);
    if (!wprior)
        wprior = make_shared<Uniform>(0, 2*M_PI);
}

void RVConditionalPrior::set_default_priors(const RVData &data)
{
    Pprior = make_shared<LogUniform>(1.0, max(1.1, data.get_timespan()));
    Kprior = make_shared<Uniform>(0.0, data.get_RV_span());
}

void RVConditionalPrior::use_hyperpriors()
{
    hyperpriors = true;
    if (!log_muP_prior)
        /// By default, Cauchy prior centered on log(365 days), scale=1
        /// for log(muP), with muP in days, truncated to (~-15.1, ~26.9)
        log_muP_prior = make_shared<TruncatedCauchy>(log(365), 1., log(365)-21, log(365)+21);

    /// By default, uniform prior for wP
    if (!wP_prior)
        wP_prior = make_shared<Uniform>(0.1, 3);

    /// By default, Cauchy prior centered on log(1), scale=1
    /// for log(muK), with muK in m/s, truncated to (-21, 21)
    /// NOTE: we actually sample on muK itself, just the prior is for log(muK)
    if (!log_muK_prior)
        log_muK_prior = make_shared<TruncatedCauchy>(0., 1., 0.-21, 0.+21);

    Pprior = make_shared<Laplace>();
    Kprior = make_shared<Exponential>();
}

void RVConditionalPrior::from_prior(RNG& rng)
{
    if(hyperpriors)
    {
        center = log_muP_prior->generate(rng);
        width = wP_prior->generate(rng);
        muK = exp(log_muK_prior->generate(rng));
    }
}

double RVConditionalPrior::perturb_hyperparameters(RNG& rng)
{
    double logH = 0.;

    if(hyperpriors)
    {
        int which = rng.rand_int(3);

        if(which == 0)
            log_muP_prior->perturb(center, rng);
        else if(which == 1)
            wP_prior->perturb(width, rng);
        else
        {
            muK = log(muK);
            log_muK_prior->perturb(muK, rng);
            muK = exp(muK);
        }
    }

    return logH;
}

double RVConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
    if(hyperpriors)
    {
        Pprior->setpars(center, width);
        Kprior->setpars(muK);
    }

    return Pprior->log_pdf(vec[0]) + 
           Kprior->log_pdf(vec[1]) + 
           phiprior->log_pdf(vec[2]) + 
           eprior->log_pdf(vec[3]) + 
           wprior->log_pdf(vec[4]);
}

void RVConditionalPrior::from_uniform(std::vector<double>& vec) const
{
    if(hyperpriors)
    {
        Pprior->setpars(center, width);
        Kprior->setpars(muK);
    }

    vec[0] = Pprior->cdf_inverse(vec[0]);
    vec[1] = Kprior->cdf_inverse(vec[1]);
    vec[2] = phiprior->cdf_inverse(vec[2]);
    vec[3] = eprior->cdf_inverse(vec[3]);
    vec[4] = wprior->cdf_inverse(vec[4]);
}

void RVConditionalPrior::to_uniform(std::vector<double>& vec) const
{
    if(hyperpriors)
    {
        Pprior->setpars(center, width);
        Kprior->setpars(muK);
    }

    vec[0] = Pprior->cdf(vec[0]);
    vec[1] = Kprior->cdf(vec[1]);
    vec[2] = phiprior->cdf(vec[2]);
    vec[3] = eprior->cdf(vec[3]);
    vec[4] = wprior->cdf(vec[4]);
}

void RVConditionalPrior::print(std::ostream& out) const
{
    if(hyperpriors)
        out<<center<<' '<<width<<' '<<muK<<' ';
}

/*****************************************************************************/

TRANSITConditionalPrior::TRANSITConditionalPrior()
{
    if (!Pprior)
        Pprior = make_shared<LogUniform>(1.0, 1000.0);
    if (!t0prior)
        t0prior = make_shared<Gaussian>(0.0, 1.0);
    if (!RPprior)
        RPprior = make_shared<Uniform>(0.0, 1.0);
    if (!aprior)
        aprior = make_shared<Uniform>(1, 10);
    if (!incprior)
        incprior = make_shared<Uniform>(0, M_PI);
    if (!eprior)
        eprior = make_shared<Uniform>(0, 1);
    if (!wprior)
        wprior = make_shared<Uniform>(0, 2*M_PI);
}

void TRANSITConditionalPrior::set_default_priors(const PHOTdata &data)
{
    Pprior = make_shared<LogUniform>(0.1, max(0.2, data.get_timespan()));
}


void TRANSITConditionalPrior::from_prior(RNG& rng)
{}

double TRANSITConditionalPrior::perturb_hyperparameters(RNG& rng)
{
    return 0.0;
}

double TRANSITConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
    return Pprior->log_pdf(vec[0]) + 
           t0prior->log_pdf(vec[1]) + 
           RPprior->log_pdf(vec[2]) + 
           aprior->log_pdf(vec[3]) + 
           incprior->log_pdf(vec[4]) + 
           eprior->log_pdf(vec[5]) + 
           wprior->log_pdf(vec[6]);
}

void TRANSITConditionalPrior::from_uniform(std::vector<double>& vec) const
{
    vec[0] = Pprior->cdf_inverse(vec[0]);
    vec[1] = t0prior->cdf_inverse(vec[1]);
    vec[2] = RPprior->cdf_inverse(vec[2]);
    vec[3] = aprior->cdf_inverse(vec[3]);
    vec[4] = incprior->cdf_inverse(vec[4]);
    vec[5] = eprior->cdf_inverse(vec[5]);
    vec[6] = wprior->cdf_inverse(vec[6]);
}

void TRANSITConditionalPrior::to_uniform(std::vector<double>& vec) const
{
    vec[0] = Pprior->cdf(vec[0]);
    vec[1] = t0prior->cdf(vec[1]);
    vec[2] = RPprior->cdf(vec[2]);
    vec[3] = aprior->cdf(vec[3]);
    vec[4] = incprior->cdf(vec[4]);
    vec[5] = eprior->cdf(vec[5]);
    vec[6] = wprior->cdf(vec[6]);
}

void TRANSITConditionalPrior::print(std::ostream& out) const
{}



using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;

void bind_RVConditionalPrior(nb::module_ &m) {
    nb::class_<RVConditionalPrior>(m, "RVConditionalPrior")
        .def(nb::init<>())
        .def_prop_rw("Pprior",
            [](RVConditionalPrior &c) { return c.Pprior; },
            [](RVConditionalPrior &c, distribution &d) { c.Pprior = d; },
            "Prior for the orbital period(s)")
        .def_prop_rw("Kprior",
            [](RVConditionalPrior &c) { return c.Kprior; },
            [](RVConditionalPrior &c, distribution &d) { c.Kprior = d; },
            "Prior for the semi-amplitude(s)")
        .def_prop_rw("eprior",
            [](RVConditionalPrior &c) { return c.eprior; },
            [](RVConditionalPrior &c, distribution &d) { c.eprior = d; },
            "Prior for the orbital eccentricities(s)")
        .def_prop_rw("wprior",
            [](RVConditionalPrior &c) { return c.wprior; },
            [](RVConditionalPrior &c, distribution &d) { c.wprior = d; },
            "Prior for the argument(s) of periastron")
        .def_prop_rw("Pprior",
            [](RVConditionalPrior &c) { return c.Pprior; },
            [](RVConditionalPrior &c, distribution &d) { c.Pprior = d; },
            "Prior for the orbital period(s)");
    
    nb::class_<TRANSITConditionalPrior>(m, "TRANSITConditionalPrior")
        .def(nb::init<>())
        .def_prop_rw("Pprior",
            [](TRANSITConditionalPrior &c) { return c.Pprior; },
            [](TRANSITConditionalPrior &c, distribution &d) { c.Pprior = d; },
            "Prior for the orbital period(s)")
        .def_prop_rw("t0prior",
            [](TRANSITConditionalPrior &c) { return c.t0prior; },
            [](TRANSITConditionalPrior &c, distribution &d) { c.t0prior = d; },
            "Prior for the time(s) of inferior conjunction")
        .def_prop_rw("RPprior",
            [](TRANSITConditionalPrior &c) { return c.RPprior; },
            [](TRANSITConditionalPrior &c, distribution &d) { c.RPprior = d; },
            "Prior for the planet(s) radius")
        .def_prop_rw("aprior",
            [](TRANSITConditionalPrior &c) { return c.aprior; },
            [](TRANSITConditionalPrior &c, distribution &d) { c.aprior = d; },
            "Prior for the planet(s) semi-major axis")
        .def_prop_rw("incprior",
            [](TRANSITConditionalPrior &c) { return c.incprior; },
            [](TRANSITConditionalPrior &c, distribution &d) { c.incprior = d; },
            "Prior for the inclinations(s)")
        .def_prop_rw("eprior",
            [](TRANSITConditionalPrior &c) { return c.eprior; },
            [](TRANSITConditionalPrior &c, distribution &d) { c.eprior = d; },
            "Prior for the orbital eccentricities(s)")
        .def_prop_rw("wprior",
            [](TRANSITConditionalPrior &c) { return c.wprior; },
            [](TRANSITConditionalPrior &c, distribution &d) { c.wprior = d; },
            "Prior for the argument(s) of periastron");
}

// NB_MODULE(ConditionalPrior, m) {
// }