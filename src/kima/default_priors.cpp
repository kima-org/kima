#include "default_priors.h"

DefaultPriors::DefaultPriors(const RVData &data) : data(data)
{
    default_mapping.insert({
        // 
        // general parameters
        // 
        {"beta_prior", make_prior<DNest4::Gaussian>(0, 1)},
        // systemic velocity
        {"Cprior", make_prior<DNest4::Uniform>(data.get_RV_min(), data.get_RV_max())},
        // between-instrument offsets
        {"offsets_prior", make_prior<DNest4::Uniform>( -data.get_RV_span(), data.get_RV_span() )},
        // jitter, per instrument
        {"Jprior", make_prior<DNest4::ModifiedLogUniform>(min(1.0, 0.1*data.get_max_RV_span()), data.get_max_RV_span())},
        // stellar jitter
        {"stellar_jitter_prior", make_prior<DNest4::Fixed>(0.0)},
        // coefficients of the trend
        {"slope_prior", make_prior<DNest4::Gaussian>( 0.0, pow(10, data.get_trend_magnitude(1)) )},
        {"quadr_prior", make_prior<DNest4::Gaussian>( 0.0, pow(10, data.get_trend_magnitude(2)) )},
        {"cubic_prior", make_prior<DNest4::Gaussian>( 0.0, pow(10, data.get_trend_magnitude(3)) )},
        // degrees of freedom of the Student t likelihood
        {"nu_prior", make_prior<DNest4::LogUniform>(2, 1000)},
        // orbital parameters
        {"Pprior", make_prior<DNest4::LogUniform>(1.0, max(1.1, data.get_timespan()))},
        {"Kprior", make_prior<DNest4::Uniform>(0.0, data.get_RV_span())},
        {"eprior", make_prior<DNest4::Uniform>(0.0, 1.0)},
        {"phiprior", make_prior<DNest4::UniformAngle>()},
        {"wprior", make_prior<DNest4::UniformAngle>()},
        // GP hyperparameters
        {"eta1_prior", make_prior<DNest4::LogUniform>( 0.1, data.get_max_RV_span() )},
        {"eta2_prior", make_prior<DNest4::LogUniform>(1, data.get_timespan())},
        {"eta3_prior", make_prior<DNest4::Uniform>(10, 40)},
        {"eta4_prior", make_prior<DNest4::Uniform>(0.2, 5.0)},
    });

    // some default priors depend on data having (at least) one indicator

    if (data.number_indicators > 0)
    {
        default_mapping.insert({
            // 
            // specific parameters in the RVFWHM model
            // systemic FWHM
            {"C2prior", make_prior<DNest4::Uniform>(data.get_actind_min(0), data.get_actind_max(0))},
            // between-instrument FWHM offsets
            {"offsets_fwhm_prior", make_prior<DNest4::Uniform>( -data.get_actind_span(0), data.get_actind_span(0) )},
            // jitter for the FWHM, per instrument
            {"J2prior", make_prior<DNest4::ModifiedLogUniform>(min(1.0, 0.1*data.get_actind_span(0)), data.get_actind_span(0))},
            // GP hyperparameters
            {"eta1_fwhm_prior", make_prior<DNest4::LogUniform>( 0.1, data.get_actind_span(0) )},
            {"eta2_fwhm_prior", make_prior<DNest4::LogUniform>(1, data.get_timespan())},
            {"eta3_fwhm_prior", make_prior<DNest4::Uniform>(10, 40)},
            {"eta4_fwhm_prior", make_prior<DNest4::Uniform>(0.2, 5.0)},
        });
    }
}

distribution DefaultPriors::get(std::string name)
{
    if (auto prior = default_mapping.find(name); prior != default_mapping.end())
    {
        return prior->second;
    }
    else
    {
        std::string msg = "kima: DefaultPriors: unknown prior: " + name;
        throw std::logic_error(msg);
    }
}


void DefaultPriors::print()
{
    for (auto prior : default_mapping)
    {
        std::cout << prior.first << ": " << *prior.second << std::endl;
    }
}