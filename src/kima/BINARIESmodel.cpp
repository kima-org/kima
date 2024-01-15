#include "BINARIESmodel.h"

using namespace std;
// using namespace Eigen;
using namespace DNest4;
using namespace nijenhuis;
using namespace brandt;
using namespace postKep;

#define TIMING false

const double halflog2pi = 0.5*log(2.*M_PI);

void BINARIESmodel::initialize_from_data(RVData& data)
{
    offsets.resize(data.number_instruments - 1);
    jitters.resize(data.number_instruments);
    offsets_2.resize(data.number_instruments - 1);
    jitters_2.resize(data.number_instruments);
    // betas.resize(data.number_indicators);
    individual_offset_prior.resize(data.number_instruments - 1);
    
    //individual_offset_prior_2.resize(data.number_instruments - 1); maybe????

    // resize RV model vector
    mu.resize(data.N());
    mu_2.resize(data.N());

    // set default conditional priors that depend on data
    auto conditional = planets.get_conditional_prior();
    conditional->set_default_priors(data);
}


/* set default priors if the user didn't change them */
void BINARIESmodel::setPriors()  // BUG: should be done by only one thread!
{

    //betaprior = make_prior<Gaussian>(0, 1);
    
    if (!Cprior)
        Cprior = make_prior<Uniform>(data.get_RV_min(), data.get_RV_max());

    if (!Jprior)
        Jprior = make_prior<ModifiedLogUniform>(min(1.0, 0.1*data.get_max_RV_span()), data.get_max_RV_span());

    // if (!slope_prior)
    //     slope_prior = make_prior<Uniform>( -data.topslope(), data.topslope() );

    if (trend){
        if (degree == 0)
            throw std::logic_error("trend=true but degree=0");
        if (degree > 3)
            throw std::range_error("can't go higher than 3rd degree trends");
        if (degree >= 1 && !slope_prior)
            slope_prior = make_prior<Gaussian>( 0.0, pow(10, data.get_trend_magnitude(1)) );
        if (degree >= 2 && !quadr_prior)
            quadr_prior = make_prior<Gaussian>( 0.0, pow(10, data.get_trend_magnitude(2)) );
        if (degree == 3 && !cubic_prior)
            cubic_prior = make_prior<Gaussian>( 0.0, pow(10, data.get_trend_magnitude(3)) );
    }

    // if offsets_prior is not (re)defined, assume a default
    if (data._multi && !offsets_prior)
        offsets_prior = make_prior<Uniform>( -data.get_RV_span(), data.get_RV_span() );

    for (size_t j = 0; j < data.number_instruments - 1; j++)
    {
        // if individual_offset_prior is not (re)defined, assume a offsets_prior
        if (!individual_offset_prior[j])
            individual_offset_prior[j] = offsets_prior;
    }


    if (known_object) { // KO mode!
        // if (n_known_object == 0) cout << "Warning: `known_object` is true, but `n_known_object` is set to 0";
        for (int i = 0; i < n_known_object; i++){
            if (!KO_Pprior[i] || !KO_Kprior[i] || !KO_eprior[i] || !KO_phiprior[i] || !KO_wprior[i] || !KO_wdotprior[i])
                throw std::logic_error("When using BINARIESmodel, please set priors on the binary parameters (KO_Pprior, KO_Kprior, KO_eprior, KO_phiprior, KO_wprior, KO_wdotprior)");
            if (double_lined && !KO_qprior[i])
                throw std::logic_error("When double_lined=true, please set prior for KO_qprior");
        }
    }

    if (studentt)
        nu_prior = make_prior<LogUniform>(2, 1000);

}


void BINARIESmodel::from_prior(RNG& rng)
{
    // preliminaries
    setPriors();
    save_setup();

    planets.from_prior(rng);
    planets.consolidate_diff();

    bkg = Cprior->generate(rng);
    bkg2 = Cprior->generate(rng);

    if(data._multi)
    {
        for(int i=0; i<offsets.size(); i++)
            offsets[i] = individual_offset_prior[i]->generate(rng);
        for(int i=0; i<jitters.size(); i++)
            jitters[i] = Jprior->generate(rng);
    }
    else
    {
        extra_sigma = Jprior->generate(rng);
        extra_sigma_2 = Jprior->generate(rng);
    }


    if(trend)
    {
        if (degree >= 1) slope = slope_prior->generate(rng);
        if (degree >= 2) quadr = quadr_prior->generate(rng);
        if (degree == 3) cubic = cubic_prior->generate(rng);
    }


//     if (data.indicator_correlations)
//     {
//         for (unsigned i=0; i<data.number_indicators; i++)
//             betas[i] = betaprior->generate(rng);
//     }

    if (known_object) { // KO mode!
        KO_P.resize(n_known_object);
        KO_K.resize(n_known_object);
        if (double_lined)
            KO_q.resize(n_known_object);
        KO_e.resize(n_known_object);
        KO_phi.resize(n_known_object);
        KO_w.resize(n_known_object);
        KO_wdot.resize(n_known_object);

        for (int i=0; i<n_known_object; i++){
            KO_P[i] = KO_Pprior[i]->generate(rng);
            KO_K[i] = KO_Kprior[i]->generate(rng);
            if (double_lined)
                KO_q[i] = KO_qprior[i]->generate(rng);
            KO_e[i] = KO_eprior[i]->generate(rng);
            KO_phi[i] = KO_phiprior[i]->generate(rng);
            KO_w[i] = KO_wprior[i]->generate(rng);
            KO_wdot[i] = KO_wdotprior[i]->generate(rng);
        }
    }

    if (studentt)
        nu = nu_prior->generate(rng);


    calculate_mu();
    if (double_lined)
        calculate_mu_2();

}


/**
 * @brief Calculate the full RV model
 * 
*/
void BINARIESmodel::calculate_mu()
{
    size_t N = data.N();

    // Update or from scratch?
    bool update = (planets.get_added().size() < planets.get_components().size()) &&
            (staleness <= 10);

    // Get the components
    const vector< vector<double> >& components = (update)?(planets.get_added()):
                (planets.get_components());
    // at this point, components has:
    //  if updating: only the added planets' parameters
    //  if from scratch: all the planets' parameters

    // Zero the signal
    if(!update) // not updating, means recalculate everything
    {
        mu.assign(mu.size(), bkg);
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

        //if(data.indicator_correlations)
        //{
            //for(size_t i=0; i<N; i++)
            //{
                //for(size_t j = 0; j < data.number_indicators; j++)
                   //mu[i] += betas[j] * actind[j][i];
            //}   
        //}

        if (known_object) { // KO mode!
            add_known_object();
        }
    }
    else // just updating (adding) planets
        staleness++;


    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif


    double f, v, ti;
    double P, K, phi, ecc, omega, omegadot, omega_t, Tp, P_anom;
    for(size_t j=0; j<components.size(); j++)
    {
        //if(hyperpriors)
            //P = exp(components[j][0]);
        //else
            //P = components[j][0];
            
        P = components[j][0];
        K = components[j][1];
        phi = components[j][2];
        ecc = components[j][3];
        omega = components[j][4];
        
        
        auto v = brandt::keplerian(data.t, P, K, ecc, omega, phi, data.M0_epoch);
        for(size_t i=0; i<N; i++)
            mu[i] += v[i];

//         for(size_t i=0; i<N; i++)
//         {
//             ti = data.t[i];
//             P_anom = postKep::period_correction(P, 0);
//             Tp = data.M0_epoch - (P_anom * phi) / (2. * M_PI);
//             omega_t = postKep::change_omega(omega, 0, ti, Tp);
//             f = nijenhuis::true_anomaly(ti, P_anom, ecc, Tp);
//             // f = brandt::true_anomaly(ti, P, ecc, Tp);
//             v = K * (cos(f + omega_t) + ecc * cos(omega_t));
//             mu[i] += v;
//         }
    }



    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Model eval took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif
    

}

void BINARIESmodel::calculate_mu_2()
{
    size_t N = data.N();

    // Update or from scratch?
    bool update = (planets.get_added().size() < planets.get_components().size()) &&
            (staleness <= 10);

    // Get the components
    const vector< vector<double> >& components = (update)?(planets.get_added()):
                (planets.get_components());
    // at this point, components has:
    //  if updating: only the added planets' parameters
    //  if from scratch: all the planets' parameters

    // Zero the signal
    if(!update) // not updating, means recalculate everything
    {
        mu_2.assign(mu_2.size(), bkg2);
        staleness = 0;
        if(trend)
        {
            double tmid = data.get_t_middle();
            for(size_t i=0; i<N; i++)
            {
                mu_2[i] += slope*(data.t[i]-tmid) + 
                            quadr*pow(data.t[i]-tmid, 2) + 
                            cubic*pow(data.t[i]-tmid, 3);
            }
        }

        if(data._multi)
        {
            for(size_t j=0; j<offsets_2.size(); j++)
            {
                for(size_t i=0; i<N; i++)
                {
                    if (data.obsi[i] == j+1) { mu_2[i] += offsets_2[j]; }
                }
            }
        }

//         if(data.indicator_correlations)
//         {
//             for(size_t i=0; i<N; i++)
//             {
//                 for(size_t j = 0; j < data.number_indicators; j++)
//                    mu_2[i] += betas[j] * actind[j][i];
//             }   
//         }

        if (known_object) { // KO mode!
            add_known_object_secondary();
        }
    }
    else // just updating (adding) planets
        staleness++;


    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif


    double f, v, ti;
    double P, K, phi, ecc, omega, omegadot, omega_t, Tp, P_anom;
    for(size_t j=0; j<components.size(); j++)
    {
//         if(hyperpriors)
//             P = exp(components[j][0]);
//         else
//             P = components[j][0];

        P = components[j][0];
        K = components[j][1];
        phi = components[j][2];
        ecc = components[j][3];
        omega = components[j][4];
        
        auto v = brandt::keplerian(data.t, P, K, ecc, omega, phi, data.M0_epoch);
        for(size_t i=0; i<N; i++)
            mu_2[i] += v[i];

//         for(size_t i=0; i<t.size(); i++)
//         {
//             ti = t[i];
//             P_anom = postKep::period_correction(P, omegadot);
//             Tp = data.M0_epoch - (P_anom * phi) / (2. * M_PI);
//             omega_t = postKep::change_omega(omega, omegadot, ti, Tp);
//             f = nijenhuis::true_anomaly(ti, P_anom, ecc, Tp);
//             // f = brandt::true_anomaly(ti, P, ecc, Tp);
//             v = K * (cos(f + omega_t) + ecc * cos(omega_t));
//             mu_2[i] += v;
//         }
    }


    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Model eval took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif
    

}

void BINARIESmodel::remove_known_object()
{   
    double f, v, delta_v, ti, Tp, w_t, P_anom;
    // cout << "in remove_known_obj: " << KO_P[1] << endl;
    for(int j=0; j<n_known_object; j++)
    {
        P_anom = postKep::period_correction(KO_P[j], KO_wdot[j]);
        auto v = postKep::keplerian_prec(data.t, P_anom, KO_K[j], KO_e[j], KO_w[j], KO_wdot[j], KO_phi[j], data.M0_epoch);
        for(size_t i=0; i<data.t.size(); i++)
        {
            mu[i] -= v[i];
//             delta_v = postKep::post_Newtonian(KO_K[j],f,KO_e[j],w_t,P_anom,star_mass,binary_mass,star_radius,relativistic_correction,tidal_correction);
        }
//         for(size_t i=0; i<data.t.size(); i++)
//         {
//             ti = data.t[i];
//             P_anom = postKep::period_correction(KO_P[j], KO_wdot[j]);
//             Tp = data.M0_epoch-(P_anom*KO_phi[j])/(2.*M_PI);
//             w_t = postKep::change_omega(KO_w[j], KO_wdot[j], ti, Tp);
//             f = nijenhuis::true_anomaly(ti, P_anom, KO_e[j], Tp);
//             v = KO_K[j] * (cos(f+w_t) + KO_e[j]*cos(w_t));
//             delta_v = postKep::post_Newtonian(KO_K[j],f,KO_e[j],w_t,P_anom,star_mass,binary_mass,star_radius,relativistic_correction,tidal_correction);
//             v += delta_v;
//             mu[i] -= v;
//         }
    }
}

void BINARIESmodel::remove_known_object_secondary()
{
    
    double f, v, delta_v, ti, Tp, w, w_t, P_anom, K2;
    // cout << "in remove_known_obj: " << KO_P[1] << endl;
    for(int j=0; j<n_known_object; j++)
    {
        P_anom = postKep::period_correction(KO_P[j], KO_wdot[j]);
        K2 = KO_K[j]/KO_q[j];
        w = KO_w[j] - M_PI;
        auto v = postKep::keplerian_prec(data.t, P_anom, K2, KO_e[j], w, KO_wdot[j], KO_phi[j], data.M0_epoch);
        for(size_t i=0; i<data.t.size(); i++)
        {
            mu_2[i] -= v[i];
        }        
//         for(size_t i=0; i<data.t.size(); i++)
//         {
//             ti = data.t[i];
//             P_anom = postKep::period_correction(KO_P[j], KO_wdot[j]);
//             Tp = data.M0_epoch-(P_anom*KO_phi[j])/(2.*M_PI);
//             w_t = postKep::change_omega(KO_w[j], KO_wdot[j], ti, Tp)-M_PI;
//             f = nijenhuis::true_anomaly(ti, P_anom, KO_e[j], Tp);
//             K2 = KO_K[j]/KO_q[j];
//             v = K2 * (cos(f+w_t) + KO_e[j]*cos(w_t));
//             delta_v = postKep::post_Newtonian(K2,f,KO_e[j],w_t,P_anom,binary_mass,star_mass,binary_radius,relativistic_correction,tidal_correction);
//             v += delta_v;
//             mu_2[i] -= v;
//         }
    }
}

void BINARIESmodel::add_known_object()
{
    
    double f, v, delta_v, ti, Tp, w_t, P_anom;
    for(int j=0; j<n_known_object; j++)
    {
        P_anom = postKep::period_correction(KO_P[j], KO_wdot[j]);
        auto v = postKep::keplerian_prec(data.t, P_anom, KO_K[j], KO_e[j], KO_w[j], KO_wdot[j], KO_phi[j], data.M0_epoch);
        for(size_t i=0; i<data.t.size(); i++)
        {
            mu[i] += v[i];
        }        
//         for(size_t i=0; i<data.t.size(); i++)
//         {
//             ti = data.t[i];
//             P_anom = postKep::period_correction(KO_P[j], KO_wdot[j]);
//             Tp = data.M0_epoch-(P_anom*KO_phi[j])/(2.*M_PI);
//             w_t = postKep::change_omega(KO_w[j], KO_wdot[j], ti, Tp);
//             f = nijenhuis::true_anomaly(ti, P_anom, KO_e[j], Tp);
//             v = KO_K[j] * (cos(f+w_t) + KO_e[j]*cos(w_t));
//             delta_v = postKep::post_Newtonian(KO_K[j],f,KO_e[j],w_t,P_anom,star_mass,binary_mass,star_radius,relativistic_correction,tidal_correction);
//             v += delta_v;
//             mu[i] += v;
//         }
    }
}

void BINARIESmodel::add_known_object_secondary()
{
    
    double f, v, delta_v, ti, Tp, w, w_t, P_anom, K2;
    for(int j=0; j<n_known_object; j++)
    {
        P_anom = postKep::period_correction(KO_P[j], KO_wdot[j]);
        K2 = KO_K[j]/KO_q[j];
        w = KO_w[j] - M_PI;
        auto v = postKep::keplerian_prec(data.t, P_anom, K2, KO_e[j], w, KO_wdot[j], KO_phi[j], data.M0_epoch);
        for(size_t i=0; i<data.t.size(); i++)
        {
            mu_2[i] += v[i];
        }        
//         for(size_t i=0; i<data.t.size(); i++)
//         {
//             ti = data.t[i];
//             P_anom = postKep::period_correction(KO_P[j], KO_wdot[j]);
//             Tp = data.M0_epoch-(P_anom*KO_phi[j])/(2.*M_PI);
//             w_t = postKep::change_omega(KO_w[j], KO_wdot[j], ti, Tp)-M_PI;
//             f = nijenhuis::true_anomaly(ti, P_anom, KO_e[j], Tp);
//             K2 = KO_K[j]/KO_q[j];
//             v = K2 * (cos(f+w_t) + KO_e[j]*cos(w_t));
//             delta_v = postKep::post_Newtonian(K2,f,KO_e[j],w_t,P_anom,binary_mass,star_mass,binary_radius,relativistic_correction,tidal_correction);
//             v += delta_v;
//             mu_2[i] += v;
//         }
    }
}

int BINARIESmodel::is_stable() const
{
    // Get the components
    const vector< vector<double> >& components = planets.get_components();
    if (components.size() == 0 && !known_object)
        return 0;
    
    int stable_planets = 0;
    int stable_known_object = 0;

    if (components.size() != 0)
        stable_planets = AMD::AMD_stable(components, star_mass);

    if (known_object) {
        vector<vector<double>> ko_components;
        ko_components.resize(n_known_object);
        for (int j = 0; j < n_known_object; j++) {
            ko_components[j] = {KO_P[j], KO_K[j], KO_phi[j], KO_e[j], KO_w[j]};
        }
        
        stable_known_object = AMD::AMD_stable(ko_components, star_mass);
    }

    return stable_planets + stable_known_object;
}


double BINARIESmodel::perturb(RNG& rng)
{
    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    auto actind = data.get_actind();
    double logH = 0.;
    double tmid = data.get_t_middle();

    

    if(rng.rand() <= 0.75) // perturb planet parameters
    {
        logH += planets.perturb(rng);
        planets.consolidate_diff();
        calculate_mu();
        if (double_lined)
            calculate_mu_2();
    }
    else if(rng.rand() <= 0.5) // perturb jitter(s) + known_object
    {
        if(data._multi)
        {
            for(int i=0; i<jitters.size(); i++)
                Jprior->perturb(jitters[i], rng);
            if (double_lined) {
                for(int i=0; i<jitters_2.size(); i++)
                    Jprior->perturb(jitters_2[i], rng);
            }
        }
        else
        {
            Jprior->perturb(extra_sigma, rng);
            if (double_lined)
                Jprior->perturb(extra_sigma_2, rng);
        }

        if (studentt)
            nu_prior->perturb(nu, rng);


        if (known_object)
        {
            remove_known_object();
            if (double_lined)
                remove_known_object_secondary();

            for (int i=0; i<n_known_object; i++){
                KO_Pprior[i]->perturb(KO_P[i], rng);
                KO_Kprior[i]->perturb(KO_K[i], rng);
                KO_eprior[i]->perturb(KO_e[i], rng);
                KO_phiprior[i]->perturb(KO_phi[i], rng);
                KO_wprior[i]->perturb(KO_w[i], rng);
                KO_wdotprior[i]->perturb(KO_wdot[i], rng);
                if (double_lined)
                    KO_qprior[i]->perturb(KO_q[i],rng);
            }

            add_known_object();
            if (double_lined)
                add_known_object_secondary();
        }
    
    }
    else
    {
        for(size_t i=0; i<mu.size(); i++)
        {
            mu[i] -= bkg;
            if(trend) {
                mu[i] -= slope*(data.t[i]-tmid) + quadr*pow(data.t[i]-tmid, 2) + cubic*pow(data.t[i]-tmid, 3);
            }
            if(data._multi) {
                for(size_t j=0; j<offsets.size(); j++){
                    if (data.obsi[i] == j+1) { mu[i] -= offsets[j]; }
                }
            }

//             if(data.indicator_correlations) {
//                 for(size_t j = 0; j < data.number_indicators; j++){
//                     mu[i] -= betas[j] * actind[j][i];
//                 }
//             }
            
            if (double_lined)
            {
                mu_2[i] -= bkg2;
                if(trend) {
                    mu_2[i] -= slope*(data.t[i]-tmid) + quadr*pow(data.t[i]-tmid, 2) + cubic*pow(data.t[i]-tmid, 3);
                }
                if(data._multi) {
                    for(size_t j=0; j<offsets_2.size(); j++){
                        if (data.obsi[i] == j+1) { mu_2[i] -= offsets_2[j]; }
                    }
                }

//                 if(data.indicator_correlations) {
//                     for(size_t j = 0; j < data.number_indicators; j++){
//                         mu_2[i] -= betas[j] * actind[j][i];
//                     }
//                 }
            }
        }

        // propose new vsys
        Cprior->perturb(bkg, rng);
        Cprior->perturb(bkg2, rng);

        // propose new instrument offsets
        if (data._multi){
            for(unsigned j=0; j<offsets.size(); j++){
                individual_offset_prior[j]->perturb(offsets[j], rng);
            }
            if (double_lined){
                for(unsigned j=0; j<offsets.size(); j++){
                    individual_offset_prior[j]->perturb(offsets_2[j], rng);
                }
            }
        }

        // propose new slope
        if(trend) {
            if (degree >= 1) slope_prior->perturb(slope, rng);
            if (degree >= 2) quadr_prior->perturb(quadr, rng);
            if (degree == 3) cubic_prior->perturb(cubic, rng);
        }

        // propose new indicator correlations
//         if(data.indicator_correlations){
//             for(size_t j = 0; j < data.number_indicators; j++){
//                 betaprior->perturb(betas[j], rng);
//             }
//         }

        for(size_t i=0; i<mu.size(); i++)
        {
            mu[i] += bkg;
            if(trend) {
                mu[i] += slope*(data.t[i]-tmid) + quadr*pow(data.t[i]-tmid, 2) + cubic*pow(data.t[i]-tmid, 3);
            }

            if(data._multi) {
                for(size_t j=0; j<offsets.size(); j++){
                    if (data.obsi[i] == j+1) { mu[i] += offsets[j]; }
                }
            }

//             if(data.indicator_correlations) {
//                 for(size_t j = 0; j < data.number_indicators; j++){
//                     mu[i] += betas[j]*actind[j][i];
//                 }
//             }
            if (double_lined)
            {
                mu_2[i] += bkg2;
                if(trend) {
                    mu_2[i] += slope*(data.t[i]-tmid) + quadr*pow(data.t[i]-tmid, 2) + cubic*pow(data.t[i]-tmid, 3);
                }
                if(data._multi) {
                    for(size_t j=0; j<offsets_2.size(); j++){
                        if (data.obsi[i] == j+1) { mu_2[i] += offsets_2[j]; }
                    }
                }

//                 if(data.indicator_correlations) {
//                     for(size_t j = 0; j < data.number_indicators; j++){
//                         mu_2[i] += betas[j]*actind[j][i];
//                     }
//                 }
            }
        }

    }


    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Perturb took ";
    cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
    cout << " μs" << std::endl;
    #endif

    return logH;
}

/**
 * Calculate the log-likelihood for the current values of the parameters.
 * 
 * @return double the log-likelihood
*/
double BINARIESmodel::log_likelihood() const
{
    size_t N = data.N();
    const auto& y = data.get_y();
    const auto& sig = data.get_sig();
    const auto& obsi = data.get_obsi();
    const auto& y_2 = data.get_y2();
    const auto& sig_2 = data.get_sig2();


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
        // The following code calculates the log likelihood 
        // in the case of a t-Student model
        double var, var_2, jit, jit2;
        for(size_t i=0; i<N; i++)
        {
            if(data._multi)
            {
                jit = jitters[obsi[i]-1];
                var = sig[i]*sig[i] + jit*jit;
                if (double_lined){
                    jit2 = jitters_2[obsi[i]-1];
                    var_2 = sig_2[i]*sig_2[i] + jit2*jit2;
                }
            }
            else
                var = sig[i]*sig[i] + extra_sigma*extra_sigma;
                if (double_lined)
                    var_2 = sig_2[i]*sig_2[i] + extra_sigma_2*extra_sigma_2;

            logL += std::lgamma(0.5*(nu + 1.)) - std::lgamma(0.5*nu)
                    - 0.5*log(M_PI*nu) - 0.5*log(var)
                    - 0.5*(nu + 1.)*log(1. + pow(y[i] - mu[i], 2)/var/nu);
            if (double_lined)
            {
                logL += std::lgamma(0.5*(nu + 1.)) - std::lgamma(0.5*nu)
                        - 0.5*log(M_PI*nu) - 0.5*log(var_2)
                        - 0.5*(nu + 1.)*log(1. + pow(y_2[i] - mu_2[i], 2)/var_2/nu);
            }
        }

    }

    else{
        // The following code calculates the log likelihood
        // in the case of a Gaussian likelihood
        double var, var_2, jit, jit2;
        for(size_t i=0; i<N; i++)
        {
            if(data._multi)
            {
                jit = jitters[obsi[i]-1];
                var = sig[i]*sig[i] + jit*jit;
                if (double_lined){
                    jit2 = jitters_2[obsi[i]-1];
                    var_2 = sig_2[i]*sig_2[i] + jit2*jit2;
                }
            }
            else
                var = sig[i]*sig[i] + extra_sigma*extra_sigma;
                if (double_lined)
                    var_2 = sig_2[i]*sig_2[i] + extra_sigma_2*extra_sigma_2;

            logL += - halflog2pi - 0.5*log(var)
                    - 0.5*(pow(y[i] - mu[i], 2)/var);
            if (double_lined)
            {
                logL += - halflog2pi - 0.5*log(var_2)
                        - 0.5*(pow(y_2[i]-mu_2[i],2)/var_2);
            }
        }
    }

    

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Likelihood took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif

    if(std::isnan(logL) || std::isinf(logL))
    {
        logL = std::numeric_limits<double>::infinity();
    }
    return logL;
}


void BINARIESmodel::print(std::ostream& out) const
{
    // output precision
    out.setf(ios::fixed,ios::floatfield);
    out.precision(8);

    if (data._multi)
    {
        for(int j=0; j<jitters.size(); j++)
            out<<jitters[j]<<'\t';
        if (double_lined) {
            for(int j=0; j<jitters_2.size(); j++)
                out<<jitters_2[j]<<'\t';
        }
    }
    else
    {
        out<<extra_sigma<<'\t';
        if (double_lined)
            out<<extra_sigma_2<<'\t';
    }

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
            out<<offsets[j]<<'\t';
        }
        if (double_lined) {
            for(int j=0; j<offsets_2.size(); j++){
                out<<offsets_2[j]<<'\t';
            }
        }
    }

//     if(data.indicator_correlations){
//         for(int j=0; j<data.number_indicators; j++){
//             out<<betas[j]<<'\t';
//         }
//     }

    if(known_object){ // KO mode!
        for (auto P: KO_P) out << P << "\t";
        for (auto K: KO_K) out << K << "\t";
        if (double_lined)
            for (auto q: KO_q) out << q << "\t";
        for (auto phi: KO_phi) out << phi << "\t";
        for (auto e: KO_e) out << e << "\t";
        for (auto w: KO_w) out << w << "\t";
        for (auto wdot: KO_wdot) out <<wdot << "\t";
    }

    planets.print(out);

    out << ' ' << staleness << ' ';

    if (studentt)
        out << '\t' << nu << '\t';
    if (double_lined)
        out << bkg2 << '\t';
    out << bkg;
}

string BINARIESmodel::description() const
{
    string desc;
    string sep = "   ";

    if (data._multi)
    {
        for(int j=0; j<jitters.size(); j++)
            desc += "jitter" + std::to_string(j+1) + sep;
        if (double_lined) {
            for(int j=0; j<jitters_2.size(); j++)
                desc += "jitter_sec" + std::to_string(j+1) + sep;
        }
    }
    else
    {
        desc += "extra_sigma   ";
        if (double_lined)
            desc += "extra_sigma_sec   ";
    }

    if(trend)
    {
        if (degree >= 1) desc += "slope" + sep;
        if (degree >= 2) desc += "quadr" + sep;
        if (degree == 3) desc += "cubic" + sep;
    }


    if (data._multi){
        for(unsigned j=0; j<offsets.size(); j++)
            desc += "offset" + std::to_string(j+1) + sep;
        if (double_lined){
            for(unsigned j=0; j<offsets_2.size(); j++)
                desc += "offset_sec" + std::to_string(j+1) + sep;
        }
    }

    if(data.indicator_correlations){
        for(int j=0; j<data.number_indicators; j++){
            desc += "beta" + std::to_string(j+1) + sep;
        }
    }

    if(known_object) { // KO mode!
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_P" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_K" + std::to_string(i) + sep;
        if (double_lined)
        {
            for(int i=0; i<n_known_object; i++) 
                desc += "KO_q" + std::to_string(i) + sep;
        }
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_phi" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_ecc" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_w" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_wdot" + std::to_string(i) + sep;
    }

    desc += "ndim" + sep + "maxNp" + sep;

    desc += "Np" + sep;

    int maxpl = planets.get_max_num_components();
    if (maxpl > 0) {
        for(int i = 0; i < maxpl; i++) desc += "P" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "K" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "phi" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "ecc" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "w" + std::to_string(i) + sep;
    }

    desc += "staleness" + sep;
    if (studentt)
        desc += "nu" + sep;
    if (double_lined)
        desc += "vsys_sec" + sep;
    desc += "vsys";

    return desc;
}

/**
 * Save the options of the current model in a INI file.
 * 
*/
void BINARIESmodel::save_setup() {
    std::fstream fout("kima_model_setup.txt", std::ios::out);
    fout << std::boolalpha;

    time_t rawtime;
    time (&rawtime);
    fout << ";" << ctime(&rawtime) << endl;

    fout << "[kima]" << endl;

    fout << "model: " << "RV_binaries_model" << endl << endl;
    fout << "fix: " << fix << endl;
    fout << "npmax: " << npmax << endl << endl;

    fout << "trend: " << trend << endl;
    fout << "degree: " << degree << endl;
    fout << "multi_instrument: " << data._multi << endl;
    fout << "known_object: " << known_object << endl;
    fout << "n_known_object: " << n_known_object << endl;
    fout << "studentt: " << studentt << endl;
    fout << "relativistic_correction: " << relativistic_correction <<endl;
    fout << "tidal_correction: " << tidal_correction <<endl;
    fout << "double_lined: " << double_lined <<endl;
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

    fout.precision(15);
    fout << "M0_epoch: " << data.M0_epoch << endl;
    fout.precision(6);

    fout << endl;

    fout << "[priors.general]" << endl;
    fout << "Cprior: " << *Cprior << endl;
    fout << "Jprior: " << *Jprior << endl;
    if (trend){
        if (degree >= 1) fout << "slope_prior: " << *slope_prior << endl;
        if (degree >= 2) fout << "quadr_prior: " << *quadr_prior << endl;
        if (degree == 3) fout << "cubic_prior: " << *cubic_prior << endl;
    }
    if (data._multi)
        fout << "offsets_prior: " << *offsets_prior << endl;
    if (studentt)
        fout << "nu_prior: " << *nu_prior << endl;


    if (planets.get_max_num_components()>0){
        auto conditional = planets.get_conditional_prior();

        fout << endl << "[priors.planets]" << endl;
        fout << "Pprior: " << *conditional->Pprior << endl;
        fout << "Kprior: " << *conditional->Kprior << endl;
        fout << "eprior: " << *conditional->eprior << endl;
        fout << "phiprior: " << *conditional->phiprior << endl;
        fout << "wprior: " << *conditional->wprior << endl;
    }

    if (known_object) {
        fout << endl << "[priors.known_object]" << endl;
        for(int i=0; i<n_known_object; i++){
            fout << "Pprior_" << i << ": " << *KO_Pprior[i] << endl;
            fout << "Kprior_" << i << ": " << *KO_Kprior[i] << endl;
            if (double_lined)
                fout << "qprior_" << i << ": " << *KO_qprior[i] << endl;
            fout << "eprior_" << i << ": " << *KO_eprior[i] << endl;
            fout << "phiprior_" << i << ": " << *KO_phiprior[i] << endl;
            fout << "wprior_" << i << ": " << *KO_wprior[i] << endl;
            fout << "wdotprior_" << i << ": " << *KO_wdotprior[i] << endl;
        }
    }

    fout << endl;
    fout.close();
}

using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;

auto BINARIESMODEL_DOC = R"D(
Implements a sum-of-Keplerians model where the number of Keplerians can be free.
This model assumes white, uncorrelated noise. This modules is tailored for the analysis
of stellar binaries through the known object mode (without it this defaults to the RVmodel)
The binary can have one set of RVs or two (one on each star). This model adds apsidal 
precession as a free parameter and accounts for GR and Tidal effects on the radial velocities.

Args:
    fix (bool, default=True): whether the number of Keplerians should be fixed
    npmax (int, default=0): maximum number of Keplerians
    data (RVData): the RV data
)D";

class BINARIESmodel_publicist : public BINARIESmodel
{
    public:
        using BINARIESmodel::trend;
        using BINARIESmodel::degree;
        using BINARIESmodel::studentt;
        using BINARIESmodel::known_object;
        using BINARIESmodel::n_known_object;
        using BINARIESmodel::star_mass;
        using BINARIESmodel::binary_mass;
        using BINARIESmodel::star_radius;
        using BINARIESmodel::binary_radius;
        using BINARIESmodel::enforce_stability;
        using BINARIESmodel::relativistic_correction;
        using BINARIESmodel::tidal_correction;
        using BINARIESmodel::double_lined;
};


NB_MODULE(BINARIESmodel, m) {
    nb::class_<BINARIESmodel>(m, "BINARIESmodel")
        .def(nb::init<bool&, int&, RVData&>(), "fix"_a, "npmax"_a, "data"_a, BINARIESMODEL_DOC)
        //
        .def_rw("trend", &BINARIESmodel_publicist::trend,
                "whether the model includes a polynomial trend")
        .def_rw("degree", &BINARIESmodel_publicist::degree,
                "degree of the polynomial trend")
        //
        .def_rw("studentt", &BINARIESmodel_publicist::studentt,
                "use a Student-t distribution for the likelihood (instead of Gaussian)")
        //
        .def_rw("known_object", &BINARIESmodel_publicist::known_object,
                "whether to include (better) known extra Keplerian curve(s), should be true for BINARIESmodel")
        .def_rw("n_known_object", &BINARIESmodel_publicist::n_known_object,
                "how many known objects, should be >=1 for BINARIESmodel")
        //
        .def_rw("star_mass", &BINARIESmodel_publicist::star_mass,
                "primary mass [Msun]")
        .def_rw("star_radius", &BINARIESmodel_publicist::star_radius,
                        "primary radius [Rsun]")
        .def_rw("binary_mass", &BINARIESmodel_publicist::binary_mass,
                "secondary mass [Msun]")
        .def_rw("binary_radius", &BINARIESmodel_publicist::binary_radius,
                        "secondary radius [Rsun]")
        //
        .def_rw("enforce_stability", &BINARIESmodel_publicist::enforce_stability, 
                "whether to enforce AMD-stability")
        
        //
        .def_rw("relativistic_correction", &BINARIESmodel_publicist::relativistic_correction,
                "whether to perform the GR correction")
        .def_rw("tidal_correction", &BINARIESmodel_publicist::tidal_correction,
                "whether to perform the tidal correction")
                
        //
        .def_rw("double_lined", &BINARIESmodel_publicist::double_lined,
                "whether the data is for a double lined binary, and has two sets of RVs for each epoch")

        // priors
        .def_prop_rw("Cprior",
            [](BINARIESmodel &m) { return m.Cprior; },
            [](BINARIESmodel &m, distribution &d) { m.Cprior = d; },
            "Prior for the systemic velocity")
        .def_prop_rw("Jprior",
            [](BINARIESmodel &m) { return m.Jprior; },
            [](BINARIESmodel &m, distribution &d) { m.Jprior = d; },
            "Prior for the extra white noise (jitter)")
        .def_prop_rw("slope_prior",
            [](BINARIESmodel &m) { return m.slope_prior; },
            [](BINARIESmodel &m, distribution &d) { m.slope_prior = d; },
            "Prior for the slope")
        .def_prop_rw("quadr_prior",
            [](BINARIESmodel &m) { return m.quadr_prior; },
            [](BINARIESmodel &m, distribution &d) { m.quadr_prior = d; },
            "Prior for the quadratic coefficient of the trend")
        .def_prop_rw("cubic_prior",
            [](BINARIESmodel &m) { return m.cubic_prior; },
            [](BINARIESmodel &m, distribution &d) { m.cubic_prior = d; },
            "Prior for the cubic coefficient of the trend")
        .def_prop_rw("offsets_prior",
            [](BINARIESmodel &m) { return m.offsets_prior; },
            [](BINARIESmodel &m, distribution &d) { m.offsets_prior = d; },
            "Common prior for the between-instrument offsets")
        .def_prop_rw("nu_prior",
            [](BINARIESmodel &m) { return m.nu_prior; },
            [](BINARIESmodel &m, distribution &d) { m.nu_prior = d; },
            "Prior for the degrees of freedom of the Student-t likelihood")
        
        // known object priors
        // ? should these setters check if known_object/sb2 is true?
        .def_prop_rw("KO_Pprior",
                     [](BINARIESmodel &m) { return m.KO_Pprior; },
                     [](BINARIESmodel &m, std::vector<distribution>& vd) { m.KO_Pprior = vd; })
        .def_prop_rw("KO_Kprior",
                     [](BINARIESmodel &m) { return m.KO_Kprior; },
                     [](BINARIESmodel &m, std::vector<distribution>& vd) { m.KO_Kprior = vd; })
        .def_prop_rw("KO_eprior",
                     [](BINARIESmodel &m) { return m.KO_eprior; },
                     [](BINARIESmodel &m, std::vector<distribution>& vd) { m.KO_eprior = vd; })
        .def_prop_rw("KO_wprior",
                     [](BINARIESmodel &m) { return m.KO_wprior; },
                     [](BINARIESmodel &m, std::vector<distribution>& vd) { m.KO_wprior = vd; })
        .def_prop_rw("KO_phiprior",
                     [](BINARIESmodel &m) { return m.KO_phiprior; },
                     [](BINARIESmodel &m, std::vector<distribution>& vd) { m.KO_phiprior = vd; })
        .def_prop_rw("KO_wdotprior",
                     [](BINARIESmodel &m) { return m.KO_wdotprior; },
                     [](BINARIESmodel &m, std::vector<distribution>& vd) { m.KO_wdotprior = vd; })
        .def_prop_rw("KO_qprior",
                     [](BINARIESmodel &m) { return m.KO_qprior; },
                     [](BINARIESmodel &m, std::vector<distribution>& vd) { m.KO_qprior = vd; })
                     
            
            
        // conditional object
        .def_prop_rw("conditional",
                     [](BINARIESmodel &m) { return m.get_conditional_prior(); },
                     [](BINARIESmodel &m, RVConditionalPrior& c) { /* does nothing */ });
}