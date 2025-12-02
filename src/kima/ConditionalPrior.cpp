#include "ConditionalPrior.h"

using namespace std;
using namespace DNest4;

/// KeplerianConditionalPrior

KeplerianConditionalPrior::KeplerianConditionalPrior():hyperpriors(false)
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

    // if (!Pprior)
    //     Pprior = make_shared<LogUniform>(1.0, 1000.0);
    // if (!Kprior)
    //     Kprior = make_shared<Uniform>(0.0, 100.0);
    // if (!eprior)
    //     eprior = make_shared<Uniform>(0, 1);
    // if (!phiprior)
    //     phiprior = make_shared<Uniform>(0, 2*M_PI);
    // if (!wprior)
    //     wprior = make_shared<Uniform>(0, 2*M_PI);
}

void KeplerianConditionalPrior::set_default_priors(const RVData &data)
{
    auto defaults = DefaultPriors(data);
    if (!Pprior) Pprior = defaults.get("Pprior");
    if (!Kprior) Kprior = defaults.get("Kprior");
    if (!eprior) eprior = defaults.get("eprior");
    if (!phiprior) phiprior = defaults.get("phiprior");
    if (!wprior) wprior = defaults.get("wprior");
}

void KeplerianConditionalPrior::use_hyperpriors()
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

void KeplerianConditionalPrior::from_prior(RNG& rng)
{
    if(hyperpriors)
    {
        center = log_muP_prior->generate(rng);
        width = wP_prior->generate(rng);
        muK = exp(log_muK_prior->generate(rng));
    }
}

double KeplerianConditionalPrior::perturb_hyperparameters(RNG& rng)
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

double KeplerianConditionalPrior::log_pdf(const std::vector<double>& vec) const
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

void KeplerianConditionalPrior::from_uniform(std::vector<double>& vec) const
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

void KeplerianConditionalPrior::to_uniform(std::vector<double>& vec) const
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

void KeplerianConditionalPrior::print(std::ostream& out) const
{
    if(hyperpriors)
        out<<center<<' '<<width<<' '<<muK<<' ';
}

/*****************************************************************************/
// TRANSITConditionalPrior

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
// GAIAConditionalPrior

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
            Omegaprior = make_shared<Uniform>(0, M_PI);
    }
    
    
}

void GAIAConditionalPrior::set_default_priors(const GAIAdata &data)
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
// RVGAIAConditionalPrior

RVGAIAConditionalPrior::RVGAIAConditionalPrior()
{
    if (!Pprior)
        Pprior = make_shared<LogUniform>(1., 1e5);
    if (!eprior)
        eprior = make_shared<Uniform>(0, 1);
    if (!phiprior)
        phiprior = make_shared<Uniform>(0, 2*M_PI);
    if (!a0prior)
        a0prior = make_shared<ModifiedLogUniform>(0.01, 1);
    if (!omegaprior)
        omegaprior = make_shared<Uniform>(0, 2*M_PI);
    if (!cosiprior)
        cosiprior = make_shared<Uniform>(0, 1);
    if (!Omegaprior)
        Omegaprior = make_shared<Uniform>(0, 2*M_PI);
    
}

void RVGAIAConditionalPrior::set_default_priors(const GAIAdata &GAIA_data, RVData &RV_data)
{
    double tmin1, tmin2, tmax1, tmax2;
    tmin1 = RV_data.get_t_min();
    tmax1 = RV_data.get_t_max();
    tmin2 = GAIA_data.get_t_min();
    tmax2 = GAIA_data.get_t_max();
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
            a0prior->log_pdf(vec[3]) + 
            omegaprior->log_pdf(vec[4]) +
            cosiprior->log_pdf(vec[5]) + 
            Omegaprior->log_pdf(vec[6]);

}

void RVGAIAConditionalPrior::from_uniform(std::vector<double>& vec) const
{
    
    vec[0] = Pprior->cdf_inverse(vec[0]);
    vec[1] = phiprior->cdf_inverse(vec[1]);
    vec[2] = eprior->cdf_inverse(vec[2]);
    vec[3] = a0prior->cdf_inverse(vec[3]);
    vec[4] = omegaprior->cdf_inverse(vec[4]);
    vec[5] = cosiprior->cdf_inverse(vec[5]);
    vec[6] = Omegaprior->cdf_inverse(vec[6]);
    
}

void RVGAIAConditionalPrior::to_uniform(std::vector<double>& vec) const
{
    vec[0] = Pprior->cdf(vec[0]);
    vec[1] = phiprior->cdf(vec[1]);
    vec[2] = eprior->cdf(vec[2]);
    vec[3] = a0prior->cdf(vec[3]);
    vec[4] = omegaprior->cdf(vec[4]);
    vec[5] = cosiprior->cdf(vec[5]);
    vec[6] = Omegaprior->cdf(vec[6]);
}

void RVGAIAConditionalPrior::print(std::ostream& out) const //needed?
{
    
}


/*****************************************************************************/
// ETVConditionalPrior

ETVConditionalPrior::ETVConditionalPrior()
{
    

    if (!Pprior)
        Pprior = make_shared<LogUniform>(1., 1e5);
    if (!Kprior)
        Kprior = make_shared<ModifiedLogUniform>(1., 1e3);
    if (!eprior)
        eprior = make_shared<Uniform>(0, 1);
    if (!phiprior)
        phiprior = make_shared<Uniform>(0, 2*M_PI);
    if (!wprior)
        wprior = make_shared<Uniform>(0, 2*M_PI);
}

void ETVConditionalPrior::set_default_priors(const ETVData &ETVdata)
{
    
}

void ETVConditionalPrior::from_prior(RNG& rng)
{

}

double ETVConditionalPrior::perturb_hyperparameters(RNG& rng)
{
    return 0.0;
}

// vec[0] = period
// vec[1] = amplitude
// vec[2] = phase
// vec[3] = ecc
// vec[4] = viewing angle

double ETVConditionalPrior::log_pdf(const std::vector<double>& vec) const
{

    return Pprior->log_pdf(vec[0]) + 
           Kprior->log_pdf(vec[1]) + 
           phiprior->log_pdf(vec[2]) + 
           eprior->log_pdf(vec[3]) + 
           wprior->log_pdf(vec[4]);
}

void ETVConditionalPrior::from_uniform(std::vector<double>& vec) const
{

    vec[0] = Pprior->cdf_inverse(vec[0]);
    vec[1] = Kprior->cdf_inverse(vec[1]);
    vec[2] = phiprior->cdf_inverse(vec[2]);
    vec[3] = eprior->cdf_inverse(vec[3]);
    vec[4] = wprior->cdf_inverse(vec[4]);
}

void ETVConditionalPrior::to_uniform(std::vector<double>& vec) const
{

    vec[0] = Pprior->cdf(vec[0]);
    vec[1] = Kprior->cdf(vec[1]);
    vec[2] = phiprior->cdf(vec[2]);
    vec[3] = eprior->cdf(vec[3]);
    vec[4] = wprior->cdf(vec[4]);
}

void ETVConditionalPrior::print(std::ostream& out) const
{

}


/*****************************************************************************/
// ApodizedKeplerianConditionalPrior

ApodizedKeplerianConditionalPrior::ApodizedKeplerianConditionalPrior():hyperpriors(false)
{
}

void ApodizedKeplerianConditionalPrior::set_default_priors(const RVData &data)
{
    auto defaults = DefaultPriors(data);
    if (!Pprior) Pprior = defaults.get("Pprior");
    if (!Kprior) Kprior = defaults.get("Kprior");
    if (!eprior) eprior = defaults.get("eprior");
    if (!phiprior) phiprior = defaults.get("phiprior");
    if (!wprior) wprior = defaults.get("wprior");
    if (!tauprior) tauprior = defaults.get("tauprior");
    if (!t0prior) t0prior = defaults.get("t0prior");
    if (!sprior) sprior = defaults.get("sprior");
}

void ApodizedKeplerianConditionalPrior::use_hyperpriors()
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

void ApodizedKeplerianConditionalPrior::from_prior(RNG& rng)
{
    if(hyperpriors)
    {
        center = log_muP_prior->generate(rng);
        width = wP_prior->generate(rng);
        muK = exp(log_muK_prior->generate(rng));
    }
}

double ApodizedKeplerianConditionalPrior::perturb_hyperparameters(RNG& rng)
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

double ApodizedKeplerianConditionalPrior::log_pdf(const std::vector<double>& vec) const
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
           wprior->log_pdf(vec[4]) +
           tauprior->log_pdf(vec[5]) +
           t0prior->log_pdf(vec[6]) +
           sprior->log_pdf(vec[7]);
}

void ApodizedKeplerianConditionalPrior::from_uniform(std::vector<double>& vec) const
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
    vec[5] = tauprior->cdf_inverse(vec[5]);
    vec[6] = t0prior->cdf_inverse(vec[6]);
    vec[7] = sprior->cdf_inverse(vec[7]);
}

void ApodizedKeplerianConditionalPrior::to_uniform(std::vector<double>& vec) const
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
    vec[5] = tauprior->cdf(vec[5]);
    vec[6] = t0prior->cdf(vec[6]);
    vec[7] = sprior->cdf(vec[7]);
}

void ApodizedKeplerianConditionalPrior::print(std::ostream& out) const
{
    if(hyperpriors)
        out << center << ' ' << width << ' ' << muK << ' ';
}


/*****************************************************************************/
// RVHGPMConditionalPrior

RVHGPMConditionalPrior::RVHGPMConditionalPrior()
{}

void RVHGPMConditionalPrior::set_default_priors(const RVData &rv_data)
{
    auto defaults = DefaultPriors(rv_data);
    if (!Pprior) Pprior = defaults.get("Pprior");
    if (!Kprior) Kprior = defaults.get("Kprior");
    if (!eprior) eprior = defaults.get("eprior");
    if (!phiprior) phiprior = defaults.get("phiprior");
    if (!wprior) wprior = defaults.get("wprior");
    // TOOD: move to defaults
    if (!iprior) iprior = defaults.get("iprior");
    if (!Omegaprior) Omegaprior = defaults.get("Ωprior");
}

void RVHGPMConditionalPrior::from_prior(RNG& rng)
{}

double RVHGPMConditionalPrior::perturb_hyperparameters(RNG& rng)
{
    return 0.0; // logH
}

double RVHGPMConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
    return Pprior->log_pdf(vec[0]) +
           Kprior->log_pdf(vec[1]) +
           phiprior->log_pdf(vec[2]) +
           eprior->log_pdf(vec[3]) +
           wprior->log_pdf(vec[4]) +
           iprior->log_pdf(vec[5]) +
           Omegaprior->log_pdf(vec[6]);
}

void RVHGPMConditionalPrior::from_uniform(std::vector<double>& vec) const
{
    vec[0] = Pprior->cdf_inverse(vec[0]);
    vec[1] = Kprior->cdf_inverse(vec[1]);
    vec[2] = phiprior->cdf_inverse(vec[2]);
    vec[3] = eprior->cdf_inverse(vec[3]);
    vec[4] = wprior->cdf_inverse(vec[4]);
    vec[5] = iprior->cdf_inverse(vec[5]);
    vec[6] = Omegaprior->cdf_inverse(vec[6]);
}

void RVHGPMConditionalPrior::to_uniform(std::vector<double>& vec) const
{
    vec[0] = Pprior->cdf(vec[0]);
    vec[1] = Kprior->cdf(vec[1]);
    vec[2] = phiprior->cdf(vec[2]);
    vec[3] = eprior->cdf(vec[3]);
    vec[4] = wprior->cdf(vec[4]);
    vec[5] = iprior->cdf(vec[5]);
    vec[6] = Omegaprior->cdf(vec[6]);
}

void RVHGPMConditionalPrior::print(std::ostream& out) const
{}



using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;

void bind_KeplerianConditionalPrior(nb::module_ &m) {
    nb::class_<KeplerianConditionalPrior>(m, "KeplerianConditionalPrior")
        .def(nb::init<>())
        .def_prop_rw("Pprior",
            [](KeplerianConditionalPrior &c) { return c.Pprior; },
            [](KeplerianConditionalPrior &c, distribution &d) { c.Pprior = d; },
            "Prior for the orbital period(s)")
        .def_prop_rw("Kprior",
            [](KeplerianConditionalPrior &c) { return c.Kprior; },
            [](KeplerianConditionalPrior &c, distribution &d) { c.Kprior = d; },
            "Prior for the semi-amplitude(s)")
        .def_prop_rw("eprior",
            [](KeplerianConditionalPrior &c) { return c.eprior; },
            [](KeplerianConditionalPrior &c, distribution &d) { c.eprior = d; },
            "Prior for the orbital eccentricity(ies)")
        .def_prop_rw("wprior",
            [](KeplerianConditionalPrior &c) { return c.wprior; },
            [](KeplerianConditionalPrior &c, distribution &d) { c.wprior = d; },
            "Prior for the argument(s) of periastron")
        .def_prop_rw("phiprior",
            [](KeplerianConditionalPrior &c) { return c.phiprior; },
            [](KeplerianConditionalPrior &c, distribution &d) { c.phiprior = d; },
            "Prior for the mean anomaly(ies)");
    
    nb::class_<ApodizedKeplerianConditionalPrior>(m, "ApodizedKeplerianConditionalPrior")
        .def(nb::init<>())
        .def_prop_rw("Pprior",
            [](ApodizedKeplerianConditionalPrior &c) { return c.Pprior; },
            [](ApodizedKeplerianConditionalPrior &c, distribution &d) { c.Pprior = d; },
            "Prior for the orbital period(s)")
        .def_prop_rw("Kprior",
            [](ApodizedKeplerianConditionalPrior &c) { return c.Kprior; },
            [](ApodizedKeplerianConditionalPrior &c, distribution &d) { c.Kprior = d; },
            "Prior for the semi-amplitude(s)")
        .def_prop_rw("eprior",
            [](ApodizedKeplerianConditionalPrior &c) { return c.eprior; },
            [](ApodizedKeplerianConditionalPrior &c, distribution &d) { c.eprior = d; },
            "Prior for the orbital eccentricity(ies)")
        .def_prop_rw("wprior",
            [](ApodizedKeplerianConditionalPrior &c) { return c.wprior; },
            [](ApodizedKeplerianConditionalPrior &c, distribution &d) { c.wprior = d; },
            "Prior for the argument(s) of periastron")
        .def_prop_rw("phiprior",
            [](ApodizedKeplerianConditionalPrior &c) { return c.phiprior; },
            [](ApodizedKeplerianConditionalPrior &c, distribution &d) { c.phiprior = d; },
            "Prior for the mean anomaly(ies)")
        .def_prop_rw("tauprior",
            [](ApodizedKeplerianConditionalPrior &c) { return c.tauprior; },
            [](ApodizedKeplerianConditionalPrior &c, distribution &d) { c.tauprior = d; },
            "Prior for the apodization width(s) τ")
        .def_prop_rw("t0prior",
            [](ApodizedKeplerianConditionalPrior &c) { return c.t0prior; },
            [](ApodizedKeplerianConditionalPrior &c, distribution &d) { c.t0prior = d; },
            "Prior for the center of the apodizing window(s) t0")
        .def_prop_rw("sprior",
            [](ApodizedKeplerianConditionalPrior &c) { return c.sprior; },
            [](ApodizedKeplerianConditionalPrior &c, distribution &d) { c.sprior = d; },
            "Prior for the shape(s) of the apodizing function (if plateau)");
        

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
        .def_prop_rw("a0prior",
            [](RVGAIAConditionalPrior &c) { return c.a0prior; },
            [](RVGAIAConditionalPrior &c, distribution &d) { c.a0prior = d; },
            "Prior for the angular photocentre semi-major axis(es) (mas)")
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

void bind_ETVConditionalPrior(nb::module_ &m) {
    nb::class_<ETVConditionalPrior>(m, "ETVConditionalPrior")
        .def(nb::init<>())
        .def_prop_rw("Pprior",
            [](ETVConditionalPrior &c) { return c.Pprior; },
            [](ETVConditionalPrior &c, distribution &d) { c.Pprior = d; },
            "Prior for the orbital period(s)")
        .def_prop_rw("Kprior",
            [](ETVConditionalPrior &c) { return c.Kprior; },
            [](ETVConditionalPrior &c, distribution &d) { c.Kprior = d; },
            "Prior for the semi-amplitude(s)")
        .def_prop_rw("eprior",
            [](ETVConditionalPrior &c) { return c.eprior; },
            [](ETVConditionalPrior &c, distribution &d) { c.eprior = d; },
            "Prior for the orbital eccentricity(ies)")
        .def_prop_rw("wprior",
            [](ETVConditionalPrior &c) { return c.wprior; },
            [](ETVConditionalPrior &c, distribution &d) { c.wprior = d; },
            "Prior for the argument(s) of periastron")
        .def_prop_rw("phiprior",
            [](ETVConditionalPrior &c) { return c.phiprior; },
            [](ETVConditionalPrior &c, distribution &d) { c.phiprior = d; },
            "Prior for the mean anomaly(ies)");
}

void bind_RVHGPMConditionalPrior(nb::module_ &m) {            
    nb::class_<RVHGPMConditionalPrior>(m, "RVHGPMConditionalPrior")
        .def(nb::init<>())
        .def_prop_rw("Pprior",
            [](RVHGPMConditionalPrior &c) { return c.Pprior; },
            [](RVHGPMConditionalPrior &c, distribution &d) { c.Pprior = d; },
            "Prior for the orbital period(s)")
        .def_prop_rw("Kprior",
            [](RVHGPMConditionalPrior &c) { return c.Kprior; },
            [](RVHGPMConditionalPrior &c, distribution &d) { c.Kprior = d; },
            "Prior for the semi-amplitude(s)")
        .def_prop_rw("eprior",
            [](RVHGPMConditionalPrior &c) { return c.eprior; },
            [](RVHGPMConditionalPrior &c, distribution &d) { c.eprior = d; },
            "Prior for the orbital eccentricity(ies)")
        .def_prop_rw("wprior",
            [](RVHGPMConditionalPrior &c) { return c.wprior; },
            [](RVHGPMConditionalPrior &c, distribution &d) { c.wprior = d; },
            "Prior for the argument(s) of periastron")
        .def_prop_rw("phiprior",
            [](RVHGPMConditionalPrior &c) { return c.phiprior; },
            [](RVHGPMConditionalPrior &c, distribution &d) { c.phiprior = d; },
            "Prior for the mean anomaly(ies)")
        .def_prop_rw("Omegaprior",
            [](RVHGPMConditionalPrior &c) { return c.Omegaprior; },
            [](RVHGPMConditionalPrior &c, distribution &d) { c.Omegaprior = d; },
            "Prior for the longitude(s) of ascending node")
        .def_prop_rw("iprior",
            [](RVHGPMConditionalPrior &c) { return c.iprior; },
            [](RVHGPMConditionalPrior &c, distribution &d) { c.iprior = d; },
            "Prior for the orbital inclination");
}


// NB_MODULE(ConditionalPrior, m) {
// }