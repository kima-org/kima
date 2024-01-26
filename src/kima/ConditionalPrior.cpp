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


/*****************************************************************************/


GAIAConditionalPrior::GAIAConditionalPrior():thiele_innes(false)
{
    
    if (!Pprior)
        Pprior = make_shared<LogUniform>(1., 1e5);
    if (!eprior)
        eprior = make_shared<Uniform>(0, 1);
    if (!phiprior)
        phiprior = make_shared<Uniform>(0, 2*M_PI);
    if(thiele_innes)
    {
        if (!Aprior)
            Aprior = make_shared<Gaussian>(0, 0.5);
        if (!Bprior)
            Bprior = make_shared<Gaussian>(0, 0.5);
        if (!Fprior)
            Fprior = make_shared<Gaussian>(0, 0.5);
        if (!Gprior)
            Gprior = make_shared<Gaussian>(0, 0.5);
    }
    else
    {
        if (!a0prior)
            a0prior = make_shared<ModifiedLogUniform>(0.01, 10);
        if (!omegaprior)
            omegaprior = make_shared<Uniform>(0, 2*M_PI);
        if (!cosiprior)
            cosiprior = make_shared<Uniform>(-1, 1);
        if (!Omegaprior)
            Omegaprior = make_shared<Uniform>(0, 2*M_PI);
    }
    
    
}


void GAIAConditionalPrior::set_default_priors(const GAIAData &data)
{
    Pprior = make_shared<LogUniform>(1.0, max(1.1, data.get_timespan()));
}

void GAIAConditionalPrior::use_thiele_innes()
{
    thiele_innes = true;
    if (!Aprior)
        Aprior = make_shared<Gaussian>(0, 0.5);
    if (!Bprior)
        Bprior = make_shared<Gaussian>(0, 0.5);
    if (!Fprior)
        Fprior = make_shared<Gaussian>(0, 0.5);
    if (!Gprior)
        Gprior = make_shared<Gaussian>(0, 0.5);
    Xprior = Aprior;
}

void GAIAConditionalPrior::from_prior(RNG& rng)//needed?
{
    
}

double GAIAConditionalPrior::perturb_hyperparameters(RNG& rng)
{
    return 0.0;
}

// vec[0] = period
// vec[1] = amplitude
// vec[2] = phase
// vec[3] = ecc
// vec[4] = viewing angle

double GAIAConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
    if(thiele_innes)
    {
        return Pprior->log_pdf(vec[0]) + 
                phiprior->log_pdf(vec[1]) + 
                eprior->log_pdf(vec[2]) + 
                Aprior->log_pdf(vec[3]) + 
                Bprior->log_pdf(vec[4]) +
                Fprior->log_pdf(vec[5]) + 
                Gprior->log_pdf(vec[6]);
    }
    else
    {
        return Pprior->log_pdf(vec[0]) + 
                phiprior->log_pdf(vec[1]) + 
                eprior->log_pdf(vec[2]) + 
                a0prior->log_pdf(vec[3]) + 
                omegaprior->log_pdf(vec[4]) +
                cosiprior->log_pdf(vec[5]) + 
                Omegaprior->log_pdf(vec[6]);
    }
}

void GAIAConditionalPrior::from_uniform(std::vector<double>& vec) const
{
    if(thiele_innes)
    {
        vec[0] = Pprior->cdf_inverse(vec[0]);
        vec[1] = phiprior->cdf_inverse(vec[1]);
        vec[2] = eprior->cdf_inverse(vec[2]);
        vec[3] = Aprior->cdf_inverse(vec[3]);
        vec[4] = Bprior->cdf_inverse(vec[4]);
        vec[5] = Fprior->cdf_inverse(vec[5]);
        vec[6] = Gprior->cdf_inverse(vec[6]);
    }
    else
    {
        vec[0] = Pprior->cdf_inverse(vec[0]);
        vec[1] = phiprior->cdf_inverse(vec[1]);
        vec[2] = eprior->cdf_inverse(vec[2]);
        vec[3] = a0prior->cdf_inverse(vec[3]);
        vec[4] = omegaprior->cdf_inverse(vec[4]);
        vec[5] = cosiprior->cdf_inverse(vec[5]);
        vec[6] = Omegaprior->cdf_inverse(vec[6]);
    }
}

void GAIAConditionalPrior::to_uniform(std::vector<double>& vec) const
{
    if(thiele_innes)
    {
        vec[0] = Pprior->cdf(vec[0]);
        vec[1] = phiprior->cdf(vec[1]);
        vec[2] = eprior->cdf(vec[2]);
        vec[3] = Aprior->cdf(vec[3]);
        vec[4] = Bprior->cdf(vec[4]);
        vec[5] = Fprior->cdf(vec[5]);
        vec[6] = Gprior->cdf(vec[6]);
    }
    else
    {
        vec[0] = Pprior->cdf(vec[0]);
        vec[1] = phiprior->cdf(vec[1]);
        vec[2] = eprior->cdf(vec[2]);
        vec[3] = a0prior->cdf(vec[3]);
        vec[4] = omegaprior->cdf(vec[4]);
        vec[5] = cosiprior->cdf(vec[5]);
        vec[6] = Omegaprior->cdf(vec[6]);
    }
}

void GAIAConditionalPrior::print(std::ostream& out) const //needed?
{
    
}


/*****************************************************************************/


RVGAIAConditionalPrior::RVGAIAConditionalPrior()
{
    
    if (!Pprior)
        Pprior = make_shared<LogUniform>(1., 1e5);
    if (!eprior)
        eprior = make_shared<Uniform>(0, 1);
    if (!phiprior)
        phiprior = make_shared<Uniform>(0, 2*M_PI);
    if (!Mprior)
        Mprior = make_shared<ModifiedLogUniform>(0.01, 10);
    if (!omegaprior)
        omegaprior = make_shared<Uniform>(0, 2*M_PI);
    if (!cosiprior)
        cosiprior = make_shared<Uniform>(-1, 1);
    if (!Omegaprior)
        Omegaprior = make_shared<Uniform>(0, M_PI);
    
}


void RVGAIAConditionalPrior::set_default_priors(const GAIAData &GAIAdata, RVData &RVdata)
{
    double tmin1, tmin2, tmax1, tmax2;
    tmin1 = RVdata.get_t_min();
    tmax1 = RVdata.get_t_max();
    tmin2 = GAIAdata.get_t_min();
    tmax2 = GAIAdata.get_t_max();
    double tspan = max(tmax1,tmax2) - min(tmin1,tmin2);
    Pprior = make_shared<LogUniform>(1.0, max(1.1, tspan));
}

void RVGAIAConditionalPrior::from_prior(RNG& rng)//needed?
{
    
}

double RVGAIAConditionalPrior::perturb_hyperparameters(RNG& rng)
{
    return 0.0;
}

double RVGAIAConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
    
    return Pprior->log_pdf(vec[0]) + 
            phiprior->log_pdf(vec[1]) + 
            eprior->log_pdf(vec[2]) + 
            Mprior->log_pdf(vec[3]) + 
            omegaprior->log_pdf(vec[4]) +
            cosiprior->log_pdf(vec[5]) + 
            Omegaprior->log_pdf(vec[6]);

}

void RVGAIAConditionalPrior::from_uniform(std::vector<double>& vec) const
{
    
    vec[0] = Pprior->cdf_inverse(vec[0]);
    vec[1] = phiprior->cdf_inverse(vec[1]);
    vec[2] = eprior->cdf_inverse(vec[2]);
    vec[3] = Mprior->cdf_inverse(vec[3]);
    vec[4] = omegaprior->cdf_inverse(vec[4]);
    vec[5] = cosiprior->cdf_inverse(vec[5]);
    vec[6] = Omegaprior->cdf_inverse(vec[6]);
    
}

void RVGAIAConditionalPrior::to_uniform(std::vector<double>& vec) const
{
    vec[0] = Pprior->cdf(vec[0]);
    vec[1] = phiprior->cdf(vec[1]);
    vec[2] = eprior->cdf(vec[2]);
    vec[3] = Mprior->cdf(vec[3]);
    vec[4] = omegaprior->cdf(vec[4]);
    vec[5] = cosiprior->cdf(vec[5]);
    vec[6] = Omegaprior->cdf(vec[6]);
}

void RVGAIAConditionalPrior::print(std::ostream& out) const //needed?
{
    
}


/*****************************************************************************/



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
            "Prior for the orbital eccentricity(ies)")
        .def_prop_rw("wprior",
            [](RVConditionalPrior &c) { return c.wprior; },
            [](RVConditionalPrior &c, distribution &d) { c.wprior = d; },
            "Prior for the argument(s) of periastron")
        .def_prop_rw("phiprior",
            [](RVConditionalPrior &c) { return c.phiprior; },
            [](RVConditionalPrior &c, distribution &d) { c.phiprior = d; },
            "Prior for the mean anomaly(ies)");
    
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
            "Prior for the orbital eccentricity(ies)")
        .def_prop_rw("wprior",
            [](TRANSITConditionalPrior &c) { return c.wprior; },
            [](TRANSITConditionalPrior &c, distribution &d) { c.wprior = d; },
            "Prior for the argument(s) of periastron");
}
           
void bind_GAIAConditionalPrior(nb::module_ &m) {            
    nb::class_<GAIAConditionalPrior>(m, "GAIAConditionalPrior")
        .def(nb::init<>())
//         .def("use_thiele_innes", []() { return GAIAConditionalPrior::use_thiele_innes(); });
        .def_rw("thiele_innes", &GAIAConditionalPrior::thiele_innes,
                "use a Student-t distribution for the likelihood (instead of Gaussian)")
        .def_prop_rw("Pprior",
            [](GAIAConditionalPrior &c) { return c.Pprior; },
            [](GAIAConditionalPrior &c, distribution &d) { c.Pprior = d; },
            "Prior for the orbital period(s)")
        .def_prop_rw("eprior",
            [](GAIAConditionalPrior &c) { return c.eprior; },
            [](GAIAConditionalPrior &c, distribution &d) { c.eprior = d; },
            "Prior for the orbital eccentricity(ies)")
        .def_prop_rw("a0prior",
            [](GAIAConditionalPrior &c) { return c.a0prior; },
            [](GAIAConditionalPrior &c, distribution &d) { c.a0prior = d; },
            "Prior for the photocentre semi-major-axis(es) (mas)")
        .def_prop_rw("omegaprior",
            [](GAIAConditionalPrior &c) { return c.omegaprior; },
            [](GAIAConditionalPrior &c, distribution &d) { c.omegaprior = d; },
            "Prior for the argument(s) of periastron")
        .def_prop_rw("phiprior",
            [](GAIAConditionalPrior &c) { return c.phiprior; },
            [](GAIAConditionalPrior &c, distribution &d) { c.phiprior = d; },
            "Prior for the mean anomaly(ies)")
        .def_prop_rw("Omegaprior",
            [](GAIAConditionalPrior &c) { return c.Omegaprior; },
            [](GAIAConditionalPrior &c, distribution &d) { c.Omegaprior = d; },
            "Prior for the longitude(s) of ascending node")
        .def_prop_rw("cosiprior",
            [](GAIAConditionalPrior &c) { return c.cosiprior; },
            [](GAIAConditionalPrior &c, distribution &d) { c.cosiprior = d; },
            "Prior for cosine(s) of the orbital inclination")
        .def_prop_rw("Aprior",
            [](GAIAConditionalPrior &c) { return c.Aprior; },
            [](GAIAConditionalPrior &c, distribution &d) { c.Aprior = d; },
            "Prior thiele_innes parameter(s) A")
        .def_prop_rw("Bprior",
            [](GAIAConditionalPrior &c) { return c.Bprior; },
            [](GAIAConditionalPrior &c, distribution &d) { c.Bprior = d; },
            "Prior thiele_innes parameter(s) B")
        .def_prop_rw("Fprior",
            [](GAIAConditionalPrior &c) { return c.Fprior; },
            [](GAIAConditionalPrior &c, distribution &d) { c.Fprior = d; },
            "Prior thiele_innes parameter(s) F")
        .def_prop_rw("Gprior",
            [](GAIAConditionalPrior &c) { return c.Gprior; },
            [](GAIAConditionalPrior &c, distribution &d) { c.Gprior = d; },
            "Prior thiele_innes parameter(s) G");
}

void bind_RVGAIAConditionalPrior(nb::module_ &m) {            
    nb::class_<RVGAIAConditionalPrior>(m, "RVGAIAConditionalPrior")
        .def(nb::init<>())
        .def_prop_rw("Pprior",
            [](RVGAIAConditionalPrior &c) { return c.Pprior; },
            [](RVGAIAConditionalPrior &c, distribution &d) { c.Pprior = d; },
            "Prior for the orbital period(s)")
        .def_prop_rw("eprior",
            [](RVGAIAConditionalPrior &c) { return c.eprior; },
            [](RVGAIAConditionalPrior &c, distribution &d) { c.eprior = d; },
            "Prior for the orbital eccentricity(ies)")
        .def_prop_rw("Mprior",
            [](RVGAIAConditionalPrior &c) { return c.Mprior; },
            [](RVGAIAConditionalPrior &c, distribution &d) { c.Mprior = d; },
            "Prior for the mass(es) (Solar mass)")
        .def_prop_rw("omegaprior",
            [](RVGAIAConditionalPrior &c) { return c.omegaprior; },
            [](RVGAIAConditionalPrior &c, distribution &d) { c.omegaprior = d; },
            "Prior for the argument(s) of periastron")
        .def_prop_rw("phiprior",
            [](RVGAIAConditionalPrior &c) { return c.phiprior; },
            [](RVGAIAConditionalPrior &c, distribution &d) { c.phiprior = d; },
            "Prior for the mean anomaly(ies)")
        .def_prop_rw("Omegaprior",
            [](RVGAIAConditionalPrior &c) { return c.Omegaprior; },
            [](RVGAIAConditionalPrior &c, distribution &d) { c.Omegaprior = d; },
            "Prior for the longitude(s) of ascending node")
        .def_prop_rw("cosiprior",
            [](RVGAIAConditionalPrior &c) { return c.cosiprior; },
            [](RVGAIAConditionalPrior &c, distribution &d) { c.cosiprior = d; },
            "Prior for cosine(s) of the orbital inclination");
}

// NB_MODULE(ConditionalPrior, m) {
// }