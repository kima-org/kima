#include "RVHGPMmodel.h"

#define TIMING false
#define DEBUG false

const double halflog2pi = 0.5*log(2.*M_PI);


void RVHGPMmodel::initialize_from_data(RVData& data)
{
    offsets.resize(data.number_instruments - 1);
    jitters.resize(data.number_instruments);
    betas.resize(data.number_indicators);
    individual_offset_prior.resize(data.number_instruments - 1);

    // resize RV model vector
    mu.resize(data.N());
    // resize astrometric model vector 
    // (pm_ra_hip, pm_dec_hip, pm_ra_gaia, pm_dec_gaia, pm_ra_hg, pm_dec_hg)
    mu_pm.resize(6);

    // set default conditional priors that depend on data
    auto conditional = planets.get_conditional_prior();
    conditional->set_default_priors(data);
}


void RVHGPMmodel::set_known_object(size_t n)
{
    known_object = true;
    n_known_object = n;
    // planet_perturb_prob = 0.5;
    // jitKO_perturb_prob = 0.3;

    KO_Pprior.resize(n); KO_Kprior.resize(n); KO_eprior.resize(n); KO_phiprior.resize(n); KO_wprior.resize(n);
    KO_iprior.resize(n); KO_Omegaprior.resize(n);

    KO_P.resize(n); KO_K.resize(n); KO_e.resize(n); KO_phi.resize(n); KO_w.resize(n);
    KO_i.resize(n); KO_Omega.resize(n);
}

void RVHGPMmodel::set_transiting_planet(size_t n)
{
    transiting_planet = true;
    n_transiting_planet = n;
    // planet_perturb_prob = 0.5;
    // jitKO_perturb_prob = 0.3;

    TR_Pprior.resize(n); TR_Kprior.resize(n); TR_eprior.resize(n); TR_Tcprior.resize(n); TR_wprior.resize(n);

    TR_P.resize(n); TR_K.resize(n); TR_e.resize(n); TR_Tc.resize(n); TR_w.resize(n);
}

void RVHGPMmodel::set_apodized_keplerians(size_t n)
{
    apodized_keplerians = true;
    n_apodized_keplerians = n;

    AK_Pprior.resize(n); AK_Kprior.resize(n); AK_eprior.resize(n); AK_phiprior.resize(n); AK_wprior.resize(n); 
    AK_tauprior.resize(n); AK_t0prior.resize(n);

    AK_P.resize(n); AK_K.resize(n); AK_e.resize(n); AK_phi.resize(n); AK_w.resize(n);
    AK_tau.resize(n); AK_t0.resize(n);
}


/* set default priors if the user didn't change them */
void RVHGPMmodel::setPriors()  // BUG: should be done by only one thread!
{
    #if DEBUG
    std::cout << std::endl << "setPriors started... " << std::endl;
    #endif

    auto defaults = DefaultPriors(data);

    beta_prior = defaults.get("beta_prior");

    if (!Cprior) 
        Cprior = defaults.get("Cprior");

    // TODO: change these priors
    if (!pm_ra_bary_prior)
        pm_ra_bary_prior = make_prior<Gaussian>(pm_data.pm_ra_hg, pm_data.sig_hg_ra);
    if (!pm_dec_bary_prior)
        pm_dec_bary_prior = make_prior<Gaussian>(pm_data.pm_dec_hg, pm_data.sig_hg_dec);
    if (!parallax_prior)
        parallax_prior = make_prior<Gaussian>(pm_data.parallax_gaia, pm_data.parallax_gaia_error);


    if (!Jprior)
        Jprior = defaults.get("Jprior");

    if (data._multi && !stellar_jitter_prior)
        stellar_jitter_prior = defaults.get("stellar_jitter_prior");

    if (jitter_propto_indicator && !jitter_slope_prior)
        jitter_slope_prior = make_prior<Uniform>(0, 50);

    if (trend)
    {
        if (degree == 0)
            throw std::logic_error("trend=true but degree=0");
        if (degree > 3)
            throw std::range_error("can't go higher than 3rd degree trends");
        if (degree >= 1 && !slope_prior)
            slope_prior = defaults.get("slope_prior");
        if (degree >= 2 && !quadr_prior)
            quadr_prior = defaults.get("quadr_prior");
        if (degree == 3 && !cubic_prior)
            cubic_prior = defaults.get("cubic_prior");
    }

    if (data._multi && !offsets_prior)
        offsets_prior = defaults.get("offsets_prior");

    for (size_t j = 0; j < data.number_instruments - 1; j++)
    {
        // if individual_offset_prior is not (re)defined, assign it offsets_prior
        if (!individual_offset_prior[j])
            individual_offset_prior[j] = offsets_prior;
    }

    // KO mode!
    if (known_object) { 
        for (size_t i = 0; i < n_known_object; i++)
        {
            if (!KO_Pprior[i] || !KO_Kprior[i] || !KO_eprior[i] || !KO_phiprior[i] || !KO_wprior[i])
            {
                std::string p = "KO_Pprior, KO_Kprior, KO_eprior, KO_phiprior, KO_wprior, KO_iprior, KO_Omegaprior";
                std::string msg = "When known_object=true, must set explicit priors for each of " + p;
                throw std::logic_error(msg);
            }
        }
    }

    if (transiting_planet)
    {
        for (size_t i = 0; i < n_transiting_planet; i++)
        {
            if (!TR_Pprior[i] || !TR_Kprior[i] || !TR_eprior[i] || !TR_Tcprior[i] || !TR_wprior[i])
            {
                std::string msg = "When transiting_planet=true, must set priors for each of TR_Pprior, TR_Kprior, TR_eprior, TR_Tcprior, TR_wprior";
                throw std::logic_error(msg);
            }
        }
    }

    if (apodized_keplerians)
    {
        for (size_t i = 0; i < n_apodized_keplerians; i++)
        {
            if (!AK_Pprior[i] || !AK_Kprior[i] || !AK_eprior[i] || !AK_phiprior[i] || !AK_wprior[i] || !AK_tauprior[i] || !AK_t0prior[i])
            {
                std::string msg = "When apodized_keplerians=true, must set priors for each of AK_Pprior, AK_Kprior, AK_eprior, AK_phiprior, AK_wprior, AK_tauprior, AK_t0prior";
                throw std::logic_error(msg);
            }
        }
    }

    if (studentt && !nu_prior)
            nu_prior = defaults.get("nu_prior");

    #if DEBUG
    std::cout << "setPriors done" << std::endl;
    #endif

}


void RVHGPMmodel::from_prior(RNG& rng)
{
    // preliminaries
    setPriors();
    save_setup();

    planets.from_prior(rng);
    planets.consolidate_diff();
    // if (remove_label_switching_degeneracy && planets.components_changed())
    //     solve_label_switching();

    background = Cprior->generate(rng);

    pm_ra_bary = pm_ra_bary_prior->generate(rng);
    pm_dec_bary = pm_dec_bary_prior->generate(rng);
    parallax = parallax_prior->generate(rng);

    if(data._multi)
    {
        for (int i = 0; i < offsets.size(); i++)
            offsets[i] = individual_offset_prior[i]->generate(rng);
        for (int i = 0; i < jitters.size(); i++)
            jitters[i] = Jprior->generate(rng);
        stellar_jitter = stellar_jitter_prior->generate(rng);
    }
    else
    {
        jitter = Jprior->generate(rng);
    }

    if (jitter_propto_indicator)
        jitter_propto_indicator_slope = jitter_slope_prior->generate(rng);

    if(trend)
    {
        if (degree >= 1) slope = slope_prior->generate(rng);
        if (degree >= 2) quadr = quadr_prior->generate(rng);
        if (degree == 3) cubic = cubic_prior->generate(rng);
    }

    if (indicator_correlations)
    {
        for (int i = 0; i < data.number_indicators; i++)
            betas[i] = beta_prior->generate(rng);
    }

    if (known_object) { // KO mode!
        for (int i = 0; i < n_known_object; i++)
        {
            KO_P[i] = KO_Pprior[i]->generate(rng);
            KO_K[i] = KO_Kprior[i]->generate(rng);
            KO_e[i] = KO_eprior[i]->generate(rng);
            KO_phi[i] = KO_phiprior[i]->generate(rng);
            KO_w[i] = KO_wprior[i]->generate(rng);
            KO_i[i] = KO_iprior[i]->generate(rng);
            KO_Omega[i] = KO_Omegaprior[i]->generate(rng);
        }
    }

    if (transiting_planet) {
        for (int i = 0; i < n_transiting_planet; i++)
        {
            TR_P[i] = TR_Pprior[i]->generate(rng);
            TR_K[i] = TR_Kprior[i]->generate(rng);
            TR_e[i] = TR_eprior[i]->generate(rng);
            TR_Tc[i] = TR_Tcprior[i]->generate(rng);
            TR_w[i] = TR_wprior[i]->generate(rng);
        }
    }

    if (apodized_keplerians) {
        for (int i = 0; i < n_apodized_keplerians; i++)
        {
            AK_P[i] = AK_Pprior[i]->generate(rng);
            AK_K[i] = AK_Kprior[i]->generate(rng);
            AK_e[i] = AK_eprior[i]->generate(rng);
            AK_phi[i] = AK_phiprior[i]->generate(rng);
            AK_w[i] = AK_wprior[i]->generate(rng);
            AK_tau[i] = AK_tauprior[i]->generate(rng);
            AK_t0[i] = AK_t0prior[i]->generate(rng);
        }
    }

    if (studentt)
        nu = nu_prior->generate(rng);

    calculate_mu();

    #if DEBUG
    std::cout << std::endl << "from_prior done" << std::endl;
    #endif
}

/**
 * @brief Calculate the full RV model
 * 
*/
void RVHGPMmodel::calculate_mu()
{
    size_t N = data.N();

    // Update or from scratch?
    // bool update = (planets.get_added().size() < planets.get_components().size()) &&
    //         (staleness <= 10);
    bool update = false;

    // Get the components
    const vector< vector<double> >& components = (update)?(planets.get_added()):
                (planets.get_components());
    // at this point, components has:
    //  if updating: only the added planets' parameters
    //  if from scratch: all the planets' parameters

    // Zero the signal
    mu_pm.assign(mu_pm.size(), 0.0);
    mu_pm[0] += pm_ra_bary; // barycenter
    mu_pm[1] += pm_dec_bary;
    mu_pm[2] += pm_ra_bary;
    mu_pm[3] += pm_dec_bary;
    mu_pm[4] += pm_ra_bary;
    mu_pm[5] += pm_dec_bary;

    if(!update) // not updating, means recalculate everything
    {
        mu.assign(mu.size(), background);

        staleness = 0;
        if(trend)
        {
            double tmid = data.get_t_middle();
            for(size_t i=0; i<N; i++)
            {
                mu[i] += slope * (data.t[i] - tmid) +
                         quadr * pow(data.t[i] - tmid, 2) +
                         cubic * pow(data.t[i] - tmid, 3);
            }
        }

        if(data._multi)
        {
            for(size_t j=0; j<offsets.size(); j++)
            {
                for(size_t i=0; i<N; i++)
                {
                    if (data.obsi[i] == j+1) { mu[i] += offsets[j]; }
                }
            }
        }

        if(indicator_correlations)
        {
            for (size_t j = 0; j < data.number_indicators; j++)
            {
                double mean = data.get_actind_mean(j);
                for (size_t i = 0; i < N; i++)
                    mu[i] += betas[j] * (data.actind[j][i] - mean);
            }
        }

        if (known_object) { // KO mode!
            add_known_object();
        }

        if (transiting_planet) {
            add_transiting_planet();
        }

        if (apodized_keplerians) {
            add_apodized_keplerians();
        }
    }

    else // just updating (adding) planets
        staleness++;


    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif


    double P, K, phi, ecc, omega, inc, Omega;
    for (size_t j = 0; j < components.size(); j++)
    {
        P = components[j][0];
        K = components[j][1];
        phi = components[j][2];
        ecc = components[j][3];
        omega = components[j][4];
        inc = components[j][5];
        Omega = components[j][6];

        auto [v, pm] = brandt::keplerian_rvpm(data.t, 
                                              {pm_data.epoch_ra_hip, pm_data.epoch_dec_hip, pm_data.epoch_ra_gaia, pm_data.epoch_dec_gaia},
                                              parallax, 
                                              P, K, ecc, omega, phi, data.M0_epoch, inc, Omega);
        // std::cout << pm.size() << " : " << pm[0][0] << " " << pm[0][1] << " " << pm[0][2] << " " << pm[0][3] << std::endl;
        // auto v = brandt::keplerian(data.t, P, K, ecc, omega, phi, data.M0_epoch);
        for (size_t i = 0; i < N; i++)
            mu[i] += v[i];
        
        mu_pm[0] += pm[0][0]; // pm RA Hipparcos
        mu_pm[1] += pm[1][0]; // pm Dec Hipparcos
        mu_pm[2] += pm[0][1]; // pm RA Gaia
        mu_pm[3] += pm[1][1]; // pm Dec Gaia
        mu_pm[4] += pm[2][0]; // pm RA Hipparcos-Gaia
        mu_pm[5] += pm[2][1]; // pm Dec Hipparcos-Gaia
    }

    #if DEBUG
    std::cout << std::endl << "calculate_mu done" << std::endl;
    #endif

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Model eval took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif

}


void RVHGPMmodel::remove_known_object()
{
    for (int j = 0; j < n_known_object; j++) {
        auto v = brandt::keplerian(data.t, KO_P[j], KO_K[j], KO_e[j], KO_w[j], KO_phi[j], data.M0_epoch);
        for (size_t i = 0; i < data.N(); i++) {
            mu[i] -= v[i];
        }
    }
}

void RVHGPMmodel::add_known_object()
{
    for (int j = 0; j < n_known_object; j++) {
        auto v = brandt::keplerian(data.t, KO_P[j], KO_K[j], KO_e[j], KO_w[j], KO_phi[j], data.M0_epoch);
        for (size_t i = 0; i < data.N(); i++) {
            mu[i] += v[i];
        }
    }
}

void RVHGPMmodel::remove_transiting_planet()
{
    for (int j = 0; j < n_transiting_planet; j++) {
        double ecc = TR_e[j];
        double f = M_PI/2 - TR_w[j];  // true anomaly at conjunction
        double E = 2.0 * atan(tan(f/2) * sqrt((1-ecc)/(1+ecc)));  // eccentric anomaly at conjunction
        double M = E - ecc * sin(E);  // mean anomaly at conjunction
        auto v = brandt::keplerian(data.t, TR_P[j], TR_K[j], TR_e[j], TR_w[j], M, TR_Tc[j]);
        for (size_t i = 0; i < data.N(); i++)
        {
            mu[i] -= v[i];
        }
    }
}

void RVHGPMmodel::add_transiting_planet()
{
    for (int j = 0; j < n_transiting_planet; j++) {
        double ecc = TR_e[j];
        double f = M_PI/2 - TR_w[j];  // true anomaly at conjunction
        double E = 2.0 * atan(tan(f/2) * sqrt((1-ecc)/(1+ecc)));  // eccentric anomaly at conjunction
        double M = E - ecc * sin(E);  // mean anomaly at conjunction
        auto v = brandt::keplerian(data.t, TR_P[j], TR_K[j], TR_e[j], TR_w[j], M, TR_Tc[j]);
        for (size_t i = 0; i < data.N(); i++)
        {
            mu[i] += v[i];
        }
    }
}

void RVHGPMmodel::remove_apodized_keplerians()
{
    for (int j = 0; j < n_apodized_keplerians; j++) {
        auto v = brandt::keplerian(data.t, AK_P[j], AK_K[j], AK_e[j], AK_w[j], AK_phi[j], data.M0_epoch);
        auto apod = gaussian(data.t, AK_t0[j], AK_tau[j]);
        for (size_t i = 0; i < data.N(); i++) {
            mu[i] -= apod[i] * v[i];
        }
    }
}

void RVHGPMmodel::add_apodized_keplerians()
{
    for (int j = 0; j < n_apodized_keplerians; j++) {
        auto v = brandt::keplerian(data.t, AK_P[j], AK_K[j], AK_e[j], AK_w[j], AK_phi[j], data.M0_epoch);
        auto apod = gaussian(data.t, AK_t0[j], AK_tau[j]);
        for (size_t i = 0; i < data.N(); i++) {
            mu[i] += apod[i] * v[i];
        }
    }
}


void RVHGPMmodel::solve_label_switching(RNG& rng)
{
    if (npmax <= 1) // nothing to do
        return;

    auto components = planets.get_components();
    auto K = components.size();

    if (K <= 1) // nothing to do
        return;

    if (K == 2 && rng.rand() <= 1.0 / K)
    {
        // cout << "swapping!" << endl;
        std::swap(components[0][0], components[1][0]);
        std::swap(components[0][1], components[1][1]);
        std::swap(components[0][2], components[1][2]);
        std::swap(components[0][3], components[1][3]);
        std::swap(components[0][4], components[1][4]);
        planets.set_components(components);
    }

//     cout << staleness << endl;
//     cout << "P: " << components[0][0] << '\t' << components[1][0] << endl;
//     return;


//     auto conditional = planets.get_conditional_prior();

//     // map periods to the hypertriangle
//     vector<double> store_Pnew(K);
//     //double x_im1 = 0.0;

//     double P1 = components[0][0];
//     double x1 = conditional->Pprior->cdf(P1);
//     double X1 = 1.0 - pow(1.0 - x1, 1.0 / K);
//     double P2 = components[1][0];
//     double x2 = conditional->Pprior->cdf(P2);
//     double X2 = 1.0 - pow(1.0 - x2, 1.0) * (1.0 - X1);
//     components[0][0] = conditional->Pprior->cdf_inverse(X1);
//     components[1][0] = conditional->Pprior->cdf_inverse(X2);
//     // cout << "P1: " << P1 << '\t' << "--> " << X1 << endl;
//     // cout << "P2: " << P2 << '\t' << "--> " << X2 << endl;

//     // for (size_t i = 0; i < K; i++)
//     // {
//     //     double P = components[i][0];
//     //     double x = conditional->Pprior->cdf(P);
//     //     double xnew = 1.0 - pow(1 - x, 1.0/(K+1.0-i-1.0)) * (1.0 - x_im1);
//     //     double Pnew = conditional->Pprior->cdf_inverse(xnew);
//     //     components[i][0] = Pnew;
//     //     store_Pnew[i] = Pnew;
//     //     x_im1 = xnew;
//     // }

//     // auto indices = argsort(store_Pnew);
//     // for (size_t i = 0; i < K - 1; i++)
//     // {
//     //     if (indices[i] > indices[i+1])
//     //     {
//     //         std::swap(components[i][1], components[i+1][1]);
//     //         std::swap(components[i][2], components[i+1][2]);
//     //         std::swap(components[i][3], components[i+1][3]);
//     //         std::swap(components[i][4], components[i+1][4]);
//     //     }
//     // }

//     planets.set_components(components);
//     cout << "out of solve_label_switching" << endl;
}

int RVHGPMmodel::is_stable() const
{
    // Get the components
    const vector< vector<double> >& components = planets.get_components();
    if (components.size() == 0 && !known_object)
        return 0;
    
    int stable_planets = 0;
    int stable_known_object = 0;
    // int stable_transiting_planet = 0;

    if (components.size() != 0) {
        if (known_object) {
            vector<vector<double>> all_components;
            all_components.insert(all_components.end(), components.begin(), components.end());
            all_components.resize(components.size() + n_known_object);
            size_t i = 0;
            for (size_t j = components.size(); j < components.size() + n_known_object; j++) {
                all_components[j] = {KO_P[i], KO_K[i], KO_phi[i], KO_e[i], KO_w[i]};
                i++;
            }
            stable_planets = AMD::AMD_stable(all_components, star_mass);
        }
        else {
            stable_planets = AMD::AMD_stable(components, star_mass);
        }
        return stable_planets;
    }

    if (known_object) {
        vector<vector<double>> ko_components;
        ko_components.resize(n_known_object);
        for (int j = 0; j < n_known_object; j++) {
            ko_components[j] = {KO_P[j], KO_K[j], KO_phi[j], KO_e[j], KO_w[j]};
        }
        stable_known_object = AMD::AMD_stable(ko_components, star_mass);
        return stable_known_object;
    }

    // if (transiting_planet) {
    //     vector<vector<double>> tr_components;
    //     tr_components.resize(n_transiting_planet);
    //     for (int j = 0; j < n_transiting_planet; j++) {
    //         tr_components[j] = {TR_P[j], TR_K[j], Tc, TR_e[j], TR_w[j]};
    //     }
    //     stable_transiting_planet = AMD::AMD_stable(tr_components, star_mass);
    // }

    // should never get here
    return 0;
}


double RVHGPMmodel::perturb(RNG& rng)
{
    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    auto actind = data.get_actind();
    double logH = 0.;
    double tmid = data.get_t_middle();

    if (remove_label_switching_degeneracy)
    {
        solve_label_switching(rng);
    }

    if(npmax > 0 && rng.rand() <= planet_perturb_prob) // perturb planet parameters
    {
        logH += planets.perturb(rng, false);
        planets.consolidate_diff();
        // if (remove_label_switching_degeneracy && planets.components_changed())
        //     solve_label_switching();

        calculate_mu();
    }
    else if(rng.rand() <= jitKO_perturb_prob) // perturb jitter(s) + known_object
    {
        if(data._multi)
        {
            logH += stellar_jitter_prior->perturb(stellar_jitter, rng);
            for (int i = 0; i < jitters.size(); i++)
                logH += Jprior->perturb(jitters[i], rng);
        }
        else
        {
            logH += Jprior->perturb(jitter, rng);
        }

        if (jitter_propto_indicator)
            logH += jitter_slope_prior->perturb(jitter_propto_indicator_slope, rng);


        if (studentt)
            logH += nu_prior->perturb(nu, rng);


        if (known_object)
        {
            remove_known_object();

            for (int i = 0; i < n_known_object; i++)
            {
                logH += KO_Pprior[i]->perturb(KO_P[i], rng);
                logH += KO_Kprior[i]->perturb(KO_K[i], rng);
                logH += KO_eprior[i]->perturb(KO_e[i], rng);
                logH += KO_phiprior[i]->perturb(KO_phi[i], rng);
                logH += KO_wprior[i]->perturb(KO_w[i], rng);
                logH += KO_iprior[i]->perturb(KO_i[i], rng);
                logH += KO_Omegaprior[i]->perturb(KO_Omega[i], rng);
            }

            add_known_object();
        }
    
        if (transiting_planet)
        {
            remove_transiting_planet();

            for (int i = 0; i < n_transiting_planet; i++)
            {
                logH += TR_Pprior[i]->perturb(TR_P[i], rng);
                logH += TR_Kprior[i]->perturb(TR_K[i], rng);
                logH += TR_eprior[i]->perturb(TR_e[i], rng);
                logH += TR_Tcprior[i]->perturb(TR_Tc[i], rng);
                logH += TR_wprior[i]->perturb(TR_w[i], rng);
            }

            add_transiting_planet();
        }

        if (apodized_keplerians)
        {
            remove_apodized_keplerians();

            for (int i = 0; i < n_apodized_keplerians; i++)
            {
                logH += AK_Pprior[i]->perturb(AK_P[i], rng);
                logH += AK_Kprior[i]->perturb(AK_K[i], rng);
                logH += AK_eprior[i]->perturb(AK_e[i], rng);
                logH += AK_phiprior[i]->perturb(AK_phi[i], rng);
                logH += AK_wprior[i]->perturb(AK_w[i], rng);
                logH += AK_tauprior[i]->perturb(AK_tau[i], rng);
                logH += AK_t0prior[i]->perturb(AK_t0[i], rng);
            }

            add_apodized_keplerians();
        }

    }
    else
    {
        // propose new vsys
        logH += Cprior->perturb(background, rng);

        // propose new instrument offsets
        if (data._multi)
        {
            for (size_t j = 0; j < offsets.size(); j++)
            {
                logH += individual_offset_prior[j]->perturb(offsets[j], rng);
            }
        }

        // propose new slope
        if(trend) {
            if (degree >= 1) logH += slope_prior->perturb(slope, rng);
            if (degree >= 2) logH += quadr_prior->perturb(quadr, rng);
            if (degree == 3) logH += cubic_prior->perturb(cubic, rng);
        }

        // propose new indicator correlations
        if(indicator_correlations){
            for (size_t j = 0; j < data.number_indicators; j++)
            {
                logH += beta_prior->perturb(betas[j], rng);
            }
        }

        logH += pm_ra_bary_prior->perturb(pm_ra_bary, rng);
        logH += pm_dec_bary_prior->perturb(pm_dec_bary, rng);
        logH += parallax_prior->perturb(parallax, rng);

        calculate_mu();
    }


    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Perturb took ";
    cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
    cout << " μs" << std::endl;
    #endif

    #if DEBUG
    std::cout << std::endl << "perturb finished" << std::endl;
    #endif

    return logH;
}

/**
 * Calculate the log-likelihood for the current values of the parameters.
 * 
 * @return double the log-likelihood
*/
double RVHGPMmodel::log_likelihood() const
{
    size_t N = data.N();
    const auto& y = data.get_y();
    const auto& sig = data.get_sig();
    const auto& obsi = data.get_obsi();
    const auto& normalized_actind = data.get_normalized_actind();

    double logL = 0.;

    if (enforce_stability){
        int stable = is_stable();
        if (stable != 0)
            return -std::numeric_limits<double>::infinity();
    }


    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    if (studentt){
        // The following code calculates the log likelihood in the case of a t-Student model
        
        // constant which only depends on nu
        double c_nu = std::lgamma(0.5*(nu + 1.)) - std::lgamma(0.5*nu) - 0.5*log(M_PI*nu);
        
        double var, jit;
        for(size_t i=0; i<N; i++)
        {
            if(data._multi) 
            {
                jit = jitters[obsi[i]-1];
                var = sig[i]*sig[i] + jit*jit + stellar_jitter*stellar_jitter;
            }
            else
            {
                var = sig[i]*sig[i] + jitter*jitter;
            }

            if (jitter_propto_indicator)
                var += pow(jitter_propto_indicator_slope * normalized_actind[jitter_propto_indicator_index][i], 2);

            logL += c_nu - 0.5*log(var) - 0.5*(nu + 1.)*log(1. + pow(y[i] - mu[i], 2)/var/nu);
        }

        auto like_hip = DNest4::BivariateGaussian(mu_pm[0], mu_pm[1], pm_data.sig_hip_ra, pm_data.sig_hip_dec, pm_data.rho_hip);
        auto like_gaia = DNest4::BivariateGaussian(mu_pm[2], mu_pm[3], pm_data.sig_gaia_ra, pm_data.sig_gaia_dec, pm_data.rho_gaia);
        auto like_hg = DNest4::BivariateGaussian(mu_pm[4], mu_pm[5], pm_data.sig_hg_ra, pm_data.sig_hg_dec, pm_data.rho_hg);
        logL += like_hip.log_pdf(pm_data.pm_ra_hip, pm_data.pm_dec_hip);
        logL += like_gaia.log_pdf(pm_data.pm_ra_gaia, pm_data.pm_dec_gaia);
        logL += like_hg.log_pdf(pm_data.pm_ra_hg, pm_data.pm_dec_hg);

    }

    else{
        // The following code calculates the log likelihood
        // in the case of a Gaussian likelihood
        double var, jit;
        for (size_t i = 0; i < N; i++)
        {
            if (data._multi)
            {
                jit = jitters[obsi[i] - 1];
                var = sig[i] * sig[i] + jit * jit + stellar_jitter * stellar_jitter;
            }
            else {
                var = sig[i] * sig[i] + jitter * jitter;
            }

            if (jitter_propto_indicator)
                var += pow(jitter_propto_indicator_slope * normalized_actind[jitter_propto_indicator_index][i], 2);

            logL += - halflog2pi - 0.5*log(var) - 0.5*(pow(y[i] - mu[i], 2)/var);
        }

        auto like_hip = DNest4::BivariateGaussian(mu_pm[0], mu_pm[1], pm_data.sig_hip_ra, pm_data.sig_hip_dec, pm_data.rho_hip);
        auto like_gaia = DNest4::BivariateGaussian(mu_pm[2], mu_pm[3], pm_data.sig_gaia_ra, pm_data.sig_gaia_dec, pm_data.rho_gaia);
        auto like_hg = DNest4::BivariateGaussian(mu_pm[4], mu_pm[5], pm_data.sig_hg_ra, pm_data.sig_hg_dec, pm_data.rho_hg);
        logL += like_hip.log_pdf(pm_data.pm_ra_hip, pm_data.pm_dec_hip);
        logL += like_gaia.log_pdf(pm_data.pm_ra_gaia, pm_data.pm_dec_gaia);
        logL += like_hg.log_pdf(pm_data.pm_ra_hg, pm_data.pm_dec_hg);
    }

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Likelihood took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif

    if(std::isnan(logL) || std::isinf(logL))
    {
        logL = std::numeric_limits<double>::infinity();
    }

    #if DEBUG
    std::cout << std::endl << "log_likelihood finished";
    std::cout << "--> logL: " << logL << std::endl;
    #endif

    return logL;
}


void RVHGPMmodel::print(std::ostream& out) const
{
    #if DEBUG
    std::cout << std::endl << "print started... " << std::endl;
    #endif

    // output precision
    out.setf(ios::fixed,ios::floatfield);
    out.precision(8);

    if (data._multi)
    {
        out << stellar_jitter << '\t';
        for (int j = 0; j < jitters.size(); j++)
            out << jitters[j] << '\t';
    }
    else
        out << jitter << '\t';

    if (jitter_propto_indicator)
        out << jitter_propto_indicator_slope << '\t';

    if(trend)
    {
        out.precision(15);
        if (degree >= 1) out << slope << '\t';
        if (degree >= 2) out << quadr << '\t';
        if (degree == 3) out << cubic << '\t';
        out.precision(8);
    }
    
    if (data._multi){
        for(int j=0; j<offsets.size(); j++){
            out << offsets[j] << '\t';
        }
    }

    if(indicator_correlations){
        for (int j = 0; j < data.number_indicators; j++)
        {
            out << betas[j] << '\t';
        }
    }
    
    if(known_object){ // KO mode!
        for (auto P: KO_P)     out << P << "\t";
        for (auto K: KO_K)     out << K << "\t";
        for (auto phi: KO_phi) out << phi << "\t";
        for (auto e: KO_e)     out << e << "\t";
        for (auto w: KO_w)     out << w << "\t";
        for (auto i: KO_i)     out << i << "\t";
        for (auto W: KO_Omega) out << W << "\t";
    }

    if(transiting_planet){
        for (auto P: TR_P)   out << P  << "\t";
        for (auto K: TR_K)   out << K  << "\t";
        for (auto Tc: TR_Tc) out << Tc << "\t";
        for (auto e: TR_e)   out << e  << "\t";
        for (auto w: TR_w)   out << w  << "\t";
    }

    if (apodized_keplerians)
    {
        for (auto P: AK_P)     out << P   << "\t";
        for (auto K: AK_K)     out << K   << "\t";
        for (auto phi: AK_phi) out << phi << "\t";
        for (auto e: AK_e)     out << e   << "\t";
        for (auto w: AK_w)     out << w   << "\t";
        for (auto tau: AK_tau) out << tau << "\t";
        for (auto t0: AK_t0)   out << t0  << "\t";
    }

    planets.print(out);

    out << staleness << '\t';

    if (studentt)
        out << nu << '\t';

    out << pm_ra_bary << '\t' << pm_dec_bary << '\t' << parallax << '\t';

    out << background;

    #if DEBUG
    std::cout << "print finished" << std::endl;
    #endif
}


string RVHGPMmodel::description() const
{
    string desc;
    string sep = "   ";

    if (data._multi)
    {
        desc += "stellar_jitter" + sep;
        for(int j=0; j<jitters.size(); j++)
           desc += "jitter" + std::to_string(j+1) + sep;
    }
    else
        desc += "jitter" + sep;

    if (jitter_propto_indicator)
        desc += "jitter_slope" + sep;

    if(trend)
    {
        if (degree >= 1) desc += "slope" + sep;
        if (degree >= 2) desc += "quadr" + sep;
        if (degree == 3) desc += "cubic" + sep;
    }


    if (data._multi){
        for(unsigned j=0; j<offsets.size(); j++)
            desc += "offset" + std::to_string(j+1) + sep;
    }

    if (indicator_correlations)
    {
        for (int j = 0; j < data.number_indicators; j++)
        {
            desc += "beta" + std::to_string(j + 1) + sep;
        }
    }

    if(known_object) { // KO mode!
        for (int i = 0; i < n_known_object; i++) desc += "KO_P" + std::to_string(i) + sep;
        for (int i = 0; i < n_known_object; i++) desc += "KO_K" + std::to_string(i) + sep;
        for (int i = 0; i < n_known_object; i++) desc += "KO_phi" + std::to_string(i) + sep;
        for (int i = 0; i < n_known_object; i++) desc += "KO_ecc" + std::to_string(i) + sep;
        for (int i = 0; i < n_known_object; i++) desc += "KO_w" + std::to_string(i) + sep;
        for (int i = 0; i < n_known_object; i++) desc += "KO_i" + std::to_string(i) + sep;
        for (int i = 0; i < n_known_object; i++) desc += "KO_Omega" + std::to_string(i) + sep;
    }

    if(transiting_planet) {
        for (int i = 0; i < n_transiting_planet; i++) desc += "TR_P" + std::to_string(i) + sep;
        for (int i = 0; i < n_transiting_planet; i++) desc += "TR_K" + std::to_string(i) + sep;
        for (int i = 0; i < n_transiting_planet; i++) desc += "TR_Tc" + std::to_string(i) + sep;
        for (int i = 0; i < n_transiting_planet; i++) desc += "TR_ecc" + std::to_string(i) + sep;
        for (int i = 0; i < n_transiting_planet; i++) desc += "TR_w" + std::to_string(i) + sep;
    }

    if (apodized_keplerians) {
        for (int i = 0; i < n_apodized_keplerians; i++) desc += "AK_P" + std::to_string(i) + sep;
        for (int i = 0; i < n_apodized_keplerians; i++) desc += "AK_K" + std::to_string(i) + sep;
        for (int i = 0; i < n_apodized_keplerians; i++) desc += "AK_phi" + std::to_string(i) + sep;
        for (int i = 0; i < n_apodized_keplerians; i++) desc += "AK_ecc" + std::to_string(i) + sep;
        for (int i = 0; i < n_apodized_keplerians; i++) desc += "AK_w" + std::to_string(i) + sep;
        for (int i = 0; i < n_apodized_keplerians; i++) desc += "AK_tau" + std::to_string(i) + sep;
        for (int i = 0; i < n_apodized_keplerians; i++) desc += "AK_t0" + std::to_string(i) + sep;
    }

    desc += "ndim" + sep + "maxNp" + sep;
    if(false) // hyperpriors
        desc += "muP" + sep + "wP" + sep + "muK";

    desc += "Np" + sep;

    int maxpl = planets.get_max_num_components();
    if (maxpl > 0) {
        for(int i = 0; i < maxpl; i++) desc += "P" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "K" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "phi" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "ecc" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "w" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "i" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "W" + std::to_string(i) + sep;
    }

    desc += "staleness" + sep;
    if (studentt)
        desc += "nu" + sep;

    desc += "pm_ra_bary" + sep + "pm_dec_bary" + sep + "plx" + sep;

    desc += "vsys";

    return desc;
}

/**
 * Save the options of the current model in a INI file.
 * 
*/
void RVHGPMmodel::save_setup() {
	std::fstream fout("kima_model_setup.txt", std::ios::out);
    fout << std::boolalpha << std::fixed;

    fout << "; " << timestamp() << endl << endl;

    fout << "[kima]" << endl;

    fout << "model: " << "RVHGPMmodel" << endl << endl;
    fout << "fix: " << fix << endl;
    fout << "npmax: " << npmax << endl << endl;

    fout << "hyperpriors: " << false << endl;
    fout << "trend: " << trend << endl;
    fout << "degree: " << degree << endl;
    fout << "multi_instrument: " << data._multi << endl;
    fout << "known_object: " << known_object << endl;
    fout << "n_known_object: " << n_known_object << endl;
    fout << "transiting_planet: " << transiting_planet << endl;
    fout << "n_transiting_planet: " << n_transiting_planet << endl;
    fout << "apodized_keplerians: " << apodized_keplerians << endl;
    fout << "n_apodized_keplerians: " << n_apodized_keplerians << endl;
    fout << "studentt: " << studentt << endl;
    fout << "indicator_correlations: " << indicator_correlations << endl;
    fout << "jitter_propto_indicator: " << jitter_propto_indicator << endl;
    fout << "jitter_propto_indicator_index: " << jitter_propto_indicator_index << endl;
    fout << endl;

    fout << endl;

    fout << "[data]" << endl;
    fout << "file: " << data._datafile << endl;
    fout << "units: " << data._units << endl;
    fout << "skip: " << data._skip << endl;
    fout << "multi: " << data._multi << endl;

    fout << "files: ";
    for (auto f: data._datafiles)
        fout << f << ",";
    fout << endl;

    fout << "indicators: ";
    for (auto n: data._indicator_names)
        fout << n << ",";
    fout << endl;

    fout << "M0_epoch: " << data.M0_epoch << endl;

    fout << endl;

    fout << "[priors.general]" << endl;
    fout << "Cprior: " << *Cprior << endl;
    fout << "Jprior: " << *Jprior << endl;
    if (data._multi)
        fout << "stellar_jitter_prior: " << *stellar_jitter_prior << endl;
    if (jitter_propto_indicator)
        fout << "jitter_slope_prior: " << *jitter_slope_prior << endl;

    if (trend){
        if (degree >= 1) fout << "slope_prior: " << *slope_prior << endl;
        if (degree >= 2) fout << "quadr_prior: " << *quadr_prior << endl;
        if (degree == 3) fout << "cubic_prior: " << *cubic_prior << endl;
    }

    if (data._multi) {
        fout << "offsets_prior: " << *offsets_prior << endl;
        int i = 0;
        for (auto &p : individual_offset_prior) {
            fout << "individual_offset_prior[" << i << "]: " << *p << endl;
            i++;
        }
    }

    if (indicator_correlations)
        fout << "beta_prior: " << *beta_prior << endl;

    if (studentt)
        fout << "nu_prior: " << *nu_prior << endl;

    fout << "pm_ra_bary_prior: " << *pm_ra_bary_prior << endl;
    fout << "pm_dec_bary_prior: " << *pm_dec_bary_prior << endl;
    fout << "parallax_prior: " << *parallax_prior << endl;

    if (planets.get_max_num_components()>0){
        auto conditional = planets.get_conditional_prior();

        fout << endl << "[priors.planets]" << endl;
        fout << "Pprior: " << *conditional->Pprior << endl;
        fout << "Kprior: " << *conditional->Kprior << endl;
        fout << "eprior: " << *conditional->eprior << endl;
        fout << "phiprior: " << *conditional->phiprior << endl;
        fout << "wprior: " << *conditional->wprior << endl;
        fout << "iprior: " << *conditional->iprior << endl;
        fout << "Omegaprior: " << *conditional->Omegaprior << endl;
    }

    if (known_object) {
        fout << endl << "[priors.known_object]" << endl;
        for (int i = 0; i < n_known_object; i++)
        {
            fout << "Pprior_" << i << ": " << *KO_Pprior[i] << endl;
            fout << "Kprior_" << i << ": " << *KO_Kprior[i] << endl;
            fout << "eprior_" << i << ": " << *KO_eprior[i] << endl;
            fout << "phiprior_" << i << ": " << *KO_phiprior[i] << endl;
            fout << "wprior_" << i << ": " << *KO_wprior[i] << endl;
        }
    }

    if (transiting_planet) {
        fout << endl << "[priors.transiting_planet]" << endl;
        for (int i = 0; i < n_transiting_planet; i++)
        {
            fout << "Pprior_" << i << ": " << *TR_Pprior[i] << endl;
            fout << "Kprior_" << i << ": " << *TR_Kprior[i] << endl;
            fout << "eprior_" << i << ": " << *TR_eprior[i] << endl;
            fout << "Tcprior_" << i << ": " << *TR_Tcprior[i] << endl;
            fout << "wprior_" << i << ": " << *TR_wprior[i] << endl;
        }
    }

    if (apodized_keplerians) {
        fout << endl << "[priors.apodized_keplerians]" << endl;
        for (int i = 0; i < n_apodized_keplerians; i++)
        {
            fout << "Pprior_" << i << ": " << *AK_Pprior[i] << endl;
            fout << "Kprior_" << i << ": " << *AK_Kprior[i] << endl;
            fout << "eprior_" << i << ": " << *AK_eprior[i] << endl;
            fout << "phiprior_" << i << ": " << *AK_phiprior[i] << endl;
            fout << "wprior_" << i << ": " << *AK_wprior[i] << endl;
            fout << "tauprior_" << i << ": " << *AK_tauprior[i] << endl;
            fout << "t0prior_" << i << ": " << *AK_t0prior[i] << endl;
        }
    }

    fout << endl;
	fout.close();

    #if DEBUG
    std::cout << std::endl << "kima_model_setup.txt saved" << std::endl;
    #endif
}


using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;

auto RVHGPMMODEL_DOC = R"D(
Model radial velocities and Hipparcos-Gaia proper motion "anomaly" with a 
sum-of-Keplerians where the number of Keplerians can be free.

Args:
    fix (bool):
        whether the number of Keplerians should be fixed
    npmax (int):
        maximum number of Keplerians
    data (RVData):
        the RV data
    pm_data (HGPMdata):
        the Hipparcos-Gaia proper motion data
)D";

class RVHGPMmodel_publicist : public RVHGPMmodel
{
    public:
        using RVHGPMmodel::fix;
        using RVHGPMmodel::npmax;
        using RVHGPMmodel::data;
        using RVHGPMmodel::pm_data;
        //
        using RVHGPMmodel::trend;
        using RVHGPMmodel::degree;
        using RVHGPMmodel::studentt;
        using RVHGPMmodel::star_mass;
        using RVHGPMmodel::enforce_stability;
        using RVHGPMmodel::remove_label_switching_degeneracy;
        using RVHGPMmodel::indicator_correlations;
        using RVHGPMmodel::jitter_propto_indicator;
        using RVHGPMmodel::jitter_propto_indicator_index;
};


NB_MODULE(RVHGPMmodel, m) {
    // bind ConditionalPrior so it can be returned
    bind_RVHGPMConditionalPrior(m);

    nb::class_<RVHGPMmodel>(m, "RVHGPMmodel", "")
        .def(nb::init<bool&, int&, RVData&, HGPMdata&>(), "fix"_a, "npmax"_a, "data"_a, "pm_data"_a, RVHGPMMODEL_DOC)
        //
        .def_rw("directory", &RVHGPMmodel::directory,
                "directory where the model ran")
        // 
        .def_rw("fix", &RVHGPMmodel_publicist::fix, "whether the number of Keplerians is fixed")
        .def_rw("npmax", &RVHGPMmodel_publicist::npmax, "maximum number of Keplerians")
        .def_ro("data", &RVHGPMmodel_publicist::data, "the RV data")
        .def_ro("pm_data", &RVHGPMmodel_publicist::pm_data, "the H-G proper motion data")

        //
        .def_rw("trend", &RVHGPMmodel_publicist::trend,
                "whether the model includes a polynomial trend")
        .def_rw("degree", &RVHGPMmodel_publicist::degree,
                "degree of the polynomial trend")

        .def_rw("studentt", &RVHGPMmodel_publicist::studentt,
                "use a Student-t distribution for the likelihood (instead of Gaussian)")

        // KO mode
        .def("set_known_object", &RVHGPMmodel::set_known_object)
        .def_prop_ro("known_object", [](RVHGPMmodel &m) { return m.get_known_object(); },
                     "whether the model includes (better) known extra Keplerian curve(s)")
        .def_prop_ro("n_known_object", [](RVHGPMmodel &m) { return m.get_n_known_object(); },
                     "how many known objects")


        // transiting planets
        .def("set_transiting_planet", &RVHGPMmodel::set_transiting_planet)
        .def_prop_ro("transiting_planet", [](RVHGPMmodel &m) { return m.get_transiting_planet(); },
                     "whether the model includes transiting planet(s)")
        .def_prop_ro("n_transiting_planet", [](RVHGPMmodel &m) { return m.get_n_transiting_planet(); },
                     "how many transiting planets")


        // apodized Keplerians
        .def("set_apodized_keplerians", &RVHGPMmodel::set_apodized_keplerians)
        .def_prop_ro("apodized_keplerians", [](RVHGPMmodel &m) { return m.get_apodized_keplerians(); },
                     "whether the model includes apodized Keplerian(s)")
        .def_prop_ro("n_apodized_keplerians", [](RVHGPMmodel &m) { return m.get_n_apodized_keplerians(); },
                     "how many apodized Keplerians")


        //
        .def_rw("star_mass", &RVHGPMmodel_publicist::star_mass,
                "stellar mass [Msun]")
        .def_rw("enforce_stability", &RVHGPMmodel_publicist::enforce_stability, 
                "whether to enforce AMD-stability")
        
        //
        .def_rw("remove_label_switching_degeneracy", &RVHGPMmodel_publicist::remove_label_switching_degeneracy, 
                "docs")
        
        //
        .def_rw("indicator_correlations", &RVHGPMmodel_publicist::indicator_correlations, 
                "include in the model linear correlations with indicators")
        
        // 
        .def_rw("jitter_propto_indicator", &RVHGPMmodel_publicist::jitter_propto_indicator, 
                "docs")
        .def_rw("jitter_propto_indicator_index", &RVHGPMmodel_publicist::jitter_propto_indicator_index, 
                "docs")

        // // to un/pickle RVHGPMmodel
        // .def("__getstate__", [](const RVHGPMmodel &m)
        //     {
        //         return std::make_tuple(m.fix, m.d._datafile, d._datafiles, d._units, d._skip, d._indicator_names, d._multi);
        //     })
        // .def("__setstate__", [](RVHGPMmodel &d, const _state_type &state)
        //     {
        //         bool _multi = std::get<5>(state);
        //         if (_multi) {
        //             new (&d) RVHGPMmodel(std::get<1>(state), std::get<2>(state), std::get<3>(state), 0, " ", std::get<4>(state));
        //             //              filename,           units,              skip   
        //         } else {
        //             new (&d) RVHGPMmodel(std::get<0>(state), std::get<2>(state), std::get<3>(state), 0, " ", std::get<4>(state));
        //             //              filenames,          units,              skip   
        //         }
        //     })
        // //


        // priors
        .def_prop_rw("Cprior",
            [](RVHGPMmodel &m) { return m.Cprior; },
            [](RVHGPMmodel &m, distribution &d) { m.Cprior = d; },
            "Prior for the systemic velocity")

        .def_prop_rw("Jprior",
            [](RVHGPMmodel &m) { return m.Jprior; },
            [](RVHGPMmodel &m, distribution &d) { m.Jprior = d; },
            "Prior for the extra white noise (jitter)")
        .def_prop_rw("stellar_jitter_prior",
            [](RVHGPMmodel &m) { return m.stellar_jitter_prior; },
            [](RVHGPMmodel &m, distribution &d) { m.stellar_jitter_prior = d; },
            "Prior for the stellar jitter (common to all instruments)")

        .def_prop_rw("slope_prior",
            [](RVHGPMmodel &m) { return m.slope_prior; },
            [](RVHGPMmodel &m, distribution &d) { m.slope_prior = d; },
            "Prior for the slope")
        .def_prop_rw("quadr_prior",
            [](RVHGPMmodel &m) { return m.quadr_prior; },
            [](RVHGPMmodel &m, distribution &d) { m.quadr_prior = d; },
            "Prior for the quadratic coefficient of the trend")
        .def_prop_rw("cubic_prior",
            [](RVHGPMmodel &m) { return m.cubic_prior; },
            [](RVHGPMmodel &m, distribution &d) { m.cubic_prior = d; },
            "Prior for the cubic coefficient of the trend")

        .def_prop_rw("offsets_prior",
            [](RVHGPMmodel &m) { return m.offsets_prior; },
            [](RVHGPMmodel &m, distribution &d) { m.offsets_prior = d; },
            "Common prior for the between-instrument offsets")
        .def_prop_rw("individual_offset_prior",
            [](RVHGPMmodel &m) { return m.individual_offset_prior; },
            [](RVHGPMmodel &m, std::vector<distribution>& vd) { m.individual_offset_prior = vd; },
            "Common prior for the between-instrument offsets")

        .def_prop_rw("beta_prior",
            [](RVHGPMmodel &m) { return m.beta_prior; },
            [](RVHGPMmodel &m, distribution &d) { m.beta_prior = d; },
            "(Common) prior for the activity indicator coefficients")

        .def_prop_rw("nu_prior",
            [](RVHGPMmodel &m) { return m.nu_prior; },
            [](RVHGPMmodel &m, distribution &d) { m.nu_prior = d; },
            "Prior for the degrees of freedom of the Student-t likelihood")


        .def_prop_rw("pm_ra_bary_prior",
            [](RVHGPMmodel &m) { return m.pm_ra_bary_prior; },
            [](RVHGPMmodel &m, distribution &d) { m.pm_ra_bary_prior = d; },
            "Prior for the proper motion of the system's barycenter (RA)")
        .def_prop_rw("pm_dec_bary_prior",
            [](RVHGPMmodel &m) { return m.pm_dec_bary_prior; },
            [](RVHGPMmodel &m, distribution &d) { m.pm_dec_bary_prior = d; },
            "Prior for the proper motion of the system's barycenter (Dec)")
        .def_prop_rw("parallax_prior",
            [](RVHGPMmodel &m) { return m.parallax_prior; },
            [](RVHGPMmodel &m, distribution &d) { m.parallax_prior = d; },
            "Prior for the parallax of the system's barycenter")


        // known object priors
        // ? should these setters check if known_object is true?
        .def_prop_rw("KO_Pprior",
                     [](RVHGPMmodel &m) { return m.KO_Pprior; },
                     [](RVHGPMmodel &m, std::vector<distribution>& vd) { m.KO_Pprior = vd; },
                     "Prior for KO orbital period")
        .def_prop_rw("KO_Kprior",
                     [](RVHGPMmodel &m) { return m.KO_Kprior; },
                     [](RVHGPMmodel &m, std::vector<distribution>& vd) { m.KO_Kprior = vd; },
                     "Prior for KO semi-amplitude")
        .def_prop_rw("KO_eprior",
                     [](RVHGPMmodel &m) { return m.KO_eprior; },
                     [](RVHGPMmodel &m, std::vector<distribution>& vd) { m.KO_eprior = vd; },
                     "Prior for KO eccentricity")
        .def_prop_rw("KO_wprior",
                     [](RVHGPMmodel &m) { return m.KO_wprior; },
                     [](RVHGPMmodel &m, std::vector<distribution>& vd) { m.KO_wprior = vd; },
                     "Prior for KO argument of periastron")
        .def_prop_rw("KO_phiprior",
                     [](RVHGPMmodel &m) { return m.KO_phiprior; },
                     [](RVHGPMmodel &m, std::vector<distribution>& vd) { m.KO_phiprior = vd; },
                     "Prior for KO mean anomaly(ies)")

        // transiting planet priors
        // ? should these setters check if transiting_planet is true?
        .def_prop_rw("TR_Pprior",
                     [](RVHGPMmodel &m) { return m.TR_Pprior; },
                     [](RVHGPMmodel &m, std::vector<distribution>& vd) { m.TR_Pprior = vd; },
                     "Prior for TR orbital period")
        .def_prop_rw("TR_Kprior",
                     [](RVHGPMmodel &m) { return m.TR_Kprior; },
                     [](RVHGPMmodel &m, std::vector<distribution>& vd) { m.TR_Kprior = vd; },
                     "Prior for TR semi-amplitude")
        .def_prop_rw("TR_eprior",
                     [](RVHGPMmodel &m) { return m.TR_eprior; },
                     [](RVHGPMmodel &m, std::vector<distribution>& vd) { m.TR_eprior = vd; },
                     "Prior for TR eccentricity")
        .def_prop_rw("TR_wprior",
                     [](RVHGPMmodel &m) { return m.TR_wprior; },
                     [](RVHGPMmodel &m, std::vector<distribution>& vd) { m.TR_wprior = vd; },
                     "Prior for TR argument of periastron")
        .def_prop_rw("TR_Tcprior",
                     [](RVHGPMmodel &m) { return m.TR_Tcprior; },
                     [](RVHGPMmodel &m, std::vector<distribution>& vd) { m.TR_Tcprior = vd; },
                     "Prior for TR mean anomaly(ies)")

        // apodized Keplerian priors
        .def_prop_rw("AK_Pprior",
                     [](RVHGPMmodel &m) { return m.AK_Pprior; },
                     [](RVHGPMmodel &m, std::vector<distribution>& vd) { m.AK_Pprior = vd; },
                     "Prior for AK orbital period")
        .def_prop_rw("AK_Kprior",
                     [](RVHGPMmodel &m) { return m.AK_Kprior; },
                     [](RVHGPMmodel &m, std::vector<distribution>& vd) { m.AK_Kprior = vd; },
                     "Prior for AK semi-amplitude")
        .def_prop_rw("AK_eprior",
                     [](RVHGPMmodel &m) { return m.AK_eprior; },
                     [](RVHGPMmodel &m, std::vector<distribution>& vd) { m.AK_eprior = vd; },
                     "Prior for AK eccentricity")
        .def_prop_rw("AK_wprior",
                     [](RVHGPMmodel &m) { return m.AK_wprior; },
                     [](RVHGPMmodel &m, std::vector<distribution>& vd) { m.AK_wprior = vd; },
                     "Prior for AK argument of periastron")
        .def_prop_rw("AK_phiprior",
                     [](RVHGPMmodel &m) { return m.AK_phiprior; },
                     [](RVHGPMmodel &m, std::vector<distribution>& vd) { m.AK_phiprior = vd; },
                     "Prior for AK mean anomaly(ies)")
        .def_prop_rw("AK_tauprior",
                     [](RVHGPMmodel &m) { return m.AK_tauprior; },
                     [](RVHGPMmodel &m, std::vector<distribution>& vd) { m.AK_tauprior = vd; },
                     "Prior for AK apodization widths τ (days)")
        .def_prop_rw("AK_t0prior",
                     [](RVHGPMmodel &m) { return m.AK_t0prior; },
                     [](RVHGPMmodel &m, std::vector<distribution>& vd) { m.AK_t0prior = vd; },
                     "Prior for AK center of apodizing windows (days)")


        .def("set_loguniform_prior_Np", &RVHGPMmodel::set_loguniform_prior_Np)

        // conditional object
        .def_prop_rw("conditional",
                     [](RVHGPMmodel &m) { return m.get_conditional_prior(); },
                     [](RVHGPMmodel &m, RVHGPMConditionalPrior& c) { /* does nothing */ });
}