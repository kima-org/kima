#include "RVGAIAmodel.h"

using namespace std;
// using namespace Eigen;
using namespace DNest4;
using namespace nijenhuis;
using namespace brandt;
using namespace MassConv;

#define TIMING false

const double halflog2pi = 0.5*log(2.*M_PI);


void RVGAIAmodel::initialize_from_data(GAIAData& GAIAdata, RVData& RVdata)
{   
    offsets.resize(RVdata.number_instruments - 1);
    jitters.resize(RVdata.number_instruments);
    betas.resize(RVdata.number_indicators);
    individual_offset_prior.resize(RVdata.number_instruments - 1);
    
    // resize GAIA model vector
    mu_GAIA.resize(GAIAdata.N());
    
    // resize RV model vector
    mu_RV.resize(RVdata.N());

    // set default conditional priors that depend on data
    auto conditional = planets.get_conditional_prior();
    conditional->set_default_priors(GAIAdata, RVdata);
}

void RVGAIAmodel::set_known_object(size_t n)
{
    known_object = true;
    n_known_object = n;

    KO_Pprior.resize(n);
    KO_Mprior.resize(n);
    KO_eprior.resize(n);
    KO_phiprior.resize(n);
    KO_omegaprior.resize(n);
    KO_cosiprior.resize(n);
    KO_Omegaprior.resize(n);
}

/* set default priors if the user didn't change them */

void RVGAIAmodel::setPriors()  // BUG: should be done by only one thread!
{   
    
    betaprior = make_prior<Gaussian>(0, 1);

    if (!Cprior)
        Cprior = make_prior<Uniform>(RVdata.get_RV_min(), RVdata.get_RV_max());

    if (!J_RV_prior)
        J_RV_prior = make_prior<ModifiedLogUniform>(
            min(1.0, 0.1*RVdata.get_max_RV_span()), 
            RVdata.get_max_RV_span()
        );
    
    if (!J_GAIA_prior)
        J_GAIA_prior = make_prior<ModifiedLogUniform>(0.1,100.);
    
    if (trend){
        if (degree == 0)
            throw std::logic_error("trend=true but degree=0");
        if (degree > 3)
            throw std::range_error("can't go higher than 3rd degree trends");
        if (degree >= 1 && !slope_prior)
            slope_prior = make_prior<Gaussian>( 0.0, pow(10, RVdata.get_trend_magnitude(1)) );
        if (degree >= 2 && !quadr_prior)
            quadr_prior = make_prior<Gaussian>( 0.0, pow(10, RVdata.get_trend_magnitude(2)) );
        if (degree == 3 && !cubic_prior)
            cubic_prior = make_prior<Gaussian>( 0.0, pow(10, RVdata.get_trend_magnitude(3)) );
    }

    // if offsets_prior is not (re)defined, assume a default
    if (RVdata._multi && !offsets_prior)
        offsets_prior = make_prior<Uniform>( -RVdata.get_RV_span(), RVdata.get_RV_span() );

    for (size_t j = 0; j < RVdata.number_instruments - 1; j++)
    {
        // if individual_offset_prior is not (re)defined, assume a offsets_prior
        if (!individual_offset_prior[j])
            individual_offset_prior[j] = offsets_prior;
    }
    
    if (!da_prior)
        da_prior = make_prior<Gaussian>(0.0,pow(10,0));
    if (!dd_prior)
        dd_prior = make_prior<Gaussian>(0.0,pow(10,0));
    if (!mua_prior)
        mua_prior = make_prior<Gaussian>(0.0,pow(10,1));
    if (!mud_prior)
        mud_prior = make_prior<Gaussian>(0.0,pow(10,1));
    if (!plx_prior)
        plx_prior = make_prior<LogUniform>(1.,100.);
        
    if (known_object) { // KO mode!
        // if (n_known_object == 0) cout << "Warning: `known_object` is true, but `n_known_object` is set to 0";
        for (int i = 0; i < n_known_object; i++){
            if (!KO_Pprior[i] || !KO_Mprior[i] || !KO_eprior[i] || !KO_phiprior[i] || !KO_omegaprior[i] || !KO_cosiprior[i] || !KO_Omegaprior[i])
                throw std::logic_error("When known_object=true, please set priors for each (KO_Pprior, KO_Kprior, KO_eprior, KO_phiprior, KO_wprior, KO_cosiprior, KO_Omprior)");
        }
    }

    if (studentt)
    {
        nu_GAIA_prior = make_prior<LogUniform>(2, 1000);
        nu_RV_prior = make_prior<LogUniform>(2, 1000);
    }

}


void RVGAIAmodel::from_prior(RNG& rng)
{
    // preliminaries
    setPriors();
    save_setup();

    planets.from_prior(rng);
    planets.consolidate_diff();
    
    
    jitter_GAIA = J_GAIA_prior->generate(rng);
    
    background = Cprior->generate(rng);
    
    da = da_prior->generate(rng);
    dd = dd_prior->generate(rng);
    mua = mua_prior->generate(rng);
    mud = mud_prior->generate(rng);
    plx = plx_prior->generate(rng);
    
    if(RVdata._multi)
    {
        for(int i=0; i<offsets.size(); i++)
            offsets[i] = individual_offset_prior[i]->generate(rng);
        for(int i=0; i<jitters.size(); i++)
            jitters[i] = J_RV_prior->generate(rng);
    }
    else
    {
        jitter_RV = J_RV_prior->generate(rng);
    }
    
    if(trend)
    {
        if (degree >= 1) slope = slope_prior->generate(rng);
        if (degree >= 2) quadr = quadr_prior->generate(rng);
        if (degree == 3) cubic = cubic_prior->generate(rng);
    }
    
    if (indicator_correlations)
    {
        for (unsigned i=0; i<RVdata.number_indicators; i++)
            betas[i] = betaprior->generate(rng);
    }
    
    if (known_object) { // KO mode!
        KO_P.resize(n_known_object);
        KO_M.resize(n_known_object);
        KO_e.resize(n_known_object);
        KO_phi.resize(n_known_object);
        KO_omega.resize(n_known_object);
        KO_cosi.resize(n_known_object);
        KO_Omega.resize(n_known_object);

        for (int i=0; i<n_known_object; i++){
            KO_P[i] = KO_Pprior[i]->generate(rng);
            KO_M[i] = KO_Mprior[i]->generate(rng);
            KO_e[i] = KO_eprior[i]->generate(rng);
            KO_phi[i] = KO_phiprior[i]->generate(rng);
            KO_omega[i] = KO_omegaprior[i]->generate(rng);
            KO_cosi[i] = KO_cosiprior[i]->generate(rng);
            KO_Omega[i] = KO_Omegaprior[i]->generate(rng);
        }
    }

    if (studentt)
    {
        nu_GAIA = nu_GAIA_prior->generate(rng);
        nu_RV = nu_RV_prior->generate(rng);
    }

    calculate_mu();

}

/**
 * @brief Calculate the full ET model
 * 
*/
void RVGAIAmodel::calculate_mu()
{

    // Get the epochs from the data
    size_t N_RV = RVdata.N();
    size_t N_GAIA = GAIAdata.N();
//     const vector<double>& t = data.get_t();
//     const vector<double>& psi = data.get_psi();
//     const vector<double>& pf = data.get_pf();

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
        mu_GAIA.assign(mu_GAIA.size(), 0.);
        mu_RV.assign(mu_RV.size(), background);
        
        staleness = 0;
        
        for(size_t i=0; i<N_GAIA; i++)
        {
            mu_GAIA[i] += (da + mua * (GAIAdata.t[i]-GAIAdata.M0_epoch)) * sin(GAIAdata.psi[i]) + (dd + mud * (GAIAdata.t[i]-GAIAdata.M0_epoch)) * cos(GAIAdata.psi[i]) + plx*GAIAdata.pf[i];
        }
        
        if(trend)
        {
            double tmid = RVdata.get_t_middle();
            for(size_t i=0; i<N_RV; i++)
            {
                mu_RV[i] += slope * (RVdata.t[i] - tmid) +
                         quadr * pow(RVdata.t[i] - tmid, 2) +
                         cubic * pow(RVdata.t[i] - tmid, 3);
            }
        }

        if(RVdata._multi)
        {
            for(size_t j=0; j<offsets.size(); j++)
            {
                for(size_t i=0; i<N_RV; i++)
                {
                    if (RVdata.obsi[i] == j+1) { mu_RV[i] += offsets[j]; }
                }
            }
        }

        if(indicator_correlations)
        {
            for(size_t i=0; i<N_RV; i++)
            {
                for(size_t j = 0; j < RVdata.number_indicators; j++)
                   mu_RV[i] += betas[j] * RVdata.actind[j][i];
            }   
        }

        if (known_object) { // KO mode!
            add_known_object();
        }
    }
    else // just updating (adding) planets
        staleness++;


    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif


    double wk, ti;
    double P, M, phi, ecc, omega, Omega, cosi, Tp, a0, K;
    double A, B, F, G, X, Y;
    for(size_t j=0; j<components.size(); j++)
    {
        P = components[j][0];
        phi = components[j][1];
        ecc = components[j][2];
        M = components[j][3];
        omega = components[j][4];
        cosi = components[j][5];
        Omega = components[j][6];
        
        K = MassConv::SemiAmp(P,ecc,star_mass,M,cosi);
        a0 = MassConv::SemiPhotPl(P,star_mass,M,plx);
        
        A = a0*(cos(omega) * cos(Omega) - sin(omega) * sin(Omega) * cosi);
        B = a0*(cos(omega) * sin(Omega) + sin(omega) * cos(Omega) * cosi);
        F = -a0*(sin(omega) * cos(Omega) + cos(omega) * sin(Omega) * cosi);
        G = -a0*(sin(omega) * sin(Omega) - cos(omega) * cos(Omega) * cosi);
        
        auto wk = brandt::keplerian_gaia(GAIAdata.t,GAIAdata.psi, A, B, F, G, ecc, P, phi, GAIAdata.M0_epoch);
        for(size_t i=0; i<N_GAIA; i++)
            mu_GAIA[i] += wk[i];
        auto v = brandt::keplerian(RVdata.t, P, K, ecc, omega, phi, GAIAdata.M0_epoch);
        for(size_t i=0; i<N_RV; i++)
            mu_RV[i] += v[i];
            
    }

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Model eval took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif

}


void RVGAIAmodel::remove_known_object()
{
    double wk, a0, K;
    double A, B, F, G, X, Y;
    // cout << "in remove_known_obj: " << KO_P[1] << endl;
    for(int j=0; j<n_known_object; j++)
    {
        
        K = MassConv::SemiAmp(KO_P[j],KO_e[j],star_mass,KO_M[j],KO_cosi[j]);
        a0 = MassConv::SemiPhotPl(KO_P[j],star_mass,KO_M[j],plx);
        
        A = a0*(cos(KO_omega[j]) * cos(KO_Omega[j]) - sin(KO_omega[j]) * sin(KO_Omega[j]) * KO_cosi[j]);
        B = a0*(cos(KO_omega[j]) * sin(KO_Omega[j]) - sin(KO_omega[j]) * cos(KO_Omega[j]) * KO_cosi[j]);
        F = a0*(sin(KO_omega[j]) * cos(KO_Omega[j]) - cos(KO_omega[j]) * sin(KO_Omega[j]) * KO_cosi[j]);
        G = a0*(sin(KO_omega[j]) * sin(KO_Omega[j]) - cos(KO_omega[j]) * cos(KO_Omega[j]) * KO_cosi[j]);
        
        auto wk = brandt::keplerian_gaia(GAIAdata.t, GAIAdata.psi, A, B, F, G, KO_e[j], KO_P[j], KO_phi[j], GAIAdata.M0_epoch);
        for (size_t i = 0; i < GAIAdata.N(); i++) {
            mu_GAIA[i] -= wk[i];
        }
        
        auto v = brandt::keplerian(RVdata.t, KO_P[j], K, KO_e[j], KO_omega[j], KO_phi[j], GAIAdata.M0_epoch);
        for (size_t i = 0; i < RVdata.N(); i++) {
            mu_RV[i] -= v[i];
        }
        
        
//         for(size_t i=0; i<data.N(); i++)
//         {
//             ti = data.t[i];
//             
//             A = KO_a0[j]*(cos(KO_omega[j]) * cos(KO_Omega[j]) - sin(KO_omega[j]) * sin(KO_Omega[j]) * KO_cosi[j]);
//             B = KO_a0[j]*(cos(KO_omega[j]) * sin(KO_Omega[j]) - sin(KO_omega[j]) * cos(KO_Omega[j]) * KO_cosi[j]);
//             F = -KO_a0[j]*(sin(KO_omega[j]) * cos(KO_Omega[j]) - cos(KO_omega[j]) * sin(KO_Omega[j]) * KO_cosi[j]);
//             G = -KO_a0[j]*(sin(KO_omega[j]) * sin(KO_Omega[j]) - cos(KO_omega[j]) * cos(KO_Omega[j]) * KO_cosi[j]);
//             
//             Tp = data.M0_epoch-(KO_P[j]*KO_phi[j])/(2.*M_PI);
//             
//             tie(X,Y) = nijenhuis::ellip_rectang(ti, KO_P[j], KO_e[j], Tp);
//             
//             wk =(B*X + G*Y)*sin(data.psi[i]) + (A*X + F*Y)*cos(data.psi[i]);
//             mu[i] -= wk;
//         }
    }
}


void RVGAIAmodel::add_known_object()
{
    double wk, a0, K;
    double A, B, F, G, X, Y;
    for(int j=0; j<n_known_object; j++)
    {
        K = MassConv::SemiAmp(KO_P[j],KO_e[j],star_mass,KO_M[j],KO_cosi[j]);
        a0 = MassConv::SemiPhotPl(KO_P[j],star_mass,KO_M[j],plx);
        
        A = a0*(cos(KO_omega[j]) * cos(KO_Omega[j]) - sin(KO_omega[j]) * sin(KO_Omega[j]) * KO_cosi[j]);
        B = a0*(cos(KO_omega[j]) * sin(KO_Omega[j]) - sin(KO_omega[j]) * cos(KO_Omega[j]) * KO_cosi[j]);
        F = a0*(sin(KO_omega[j]) * cos(KO_Omega[j]) - cos(KO_omega[j]) * sin(KO_Omega[j]) * KO_cosi[j]);
        G = a0*(sin(KO_omega[j]) * sin(KO_Omega[j]) - cos(KO_omega[j]) * cos(KO_Omega[j]) * KO_cosi[j]);
        
        auto wk = brandt::keplerian_gaia(GAIAdata.t, GAIAdata.psi, A, B, F, G, KO_e[j], KO_P[j], KO_phi[j], GAIAdata.M0_epoch);
        for (size_t i = 0; i < GAIAdata.N(); i++) {
            mu_GAIA[i] += wk[i];
        }
        
        auto v = brandt::keplerian(RVdata.t, KO_P[j], K, KO_e[j], KO_omega[j], KO_phi[j], GAIAdata.M0_epoch);
        for (size_t i = 0; i < RVdata.N(); i++) {
            mu_RV[i] += v[i];
        }
//         
//         for(size_t i=0; i<data.N(); i++)
//         {
//             ti = data.t[i];
//             
//             A = KO_a0[j]*(cos(KO_omega[j]) * cos(KO_Omega[j]) - sin(KO_omega[j]) * sin(KO_Omega[j]) * KO_cosi[j]);
//             B = KO_a0[j]*(cos(KO_omega[j]) * sin(KO_Omega[j]) - sin(KO_omega[j]) * cos(KO_Omega[j]) * KO_cosi[j]);
//             F = -KO_a0[j]*(sin(KO_omega[j]) * cos(KO_Omega[j]) - cos(KO_omega[j]) * sin(KO_Omega[j]) * KO_cosi[j]);
//             G = -KO_a0[j]*(sin(KO_omega[j]) * sin(KO_Omega[j]) - cos(KO_omega[j]) * cos(KO_Omega[j]) * KO_cosi[j]);
//             
//             Tp = data.M0_epoch-(KO_P[j]*KO_phi[j])/(2.*M_PI);
//             
//             tie(X,Y) = nijenhuis::ellip_rectang(ti,  KO_P[j], KO_e[j], Tp);
//             
//             wk =(B*X + G*Y)*sin(data.psi[i]) + (A*X + F*Y)*cos(data.psi[i]);
//             mu[i] += wk;
//         }
    }
}

double RVGAIAmodel::perturb(RNG& rng)
{
    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif
    

    double logH = 0.;
    auto actind = RVdata.get_actind();
    double tmid = RVdata.get_t_middle();

    if(rng.rand() <= 0.75) // perturb planet parameters
    {
        logH += planets.perturb(rng);
        planets.consolidate_diff();
        calculate_mu();
    }
    else if(rng.rand() <= 0.25) // perturb jitter(s) + known_object
    {
        
        J_GAIA_prior->perturb(jitter_GAIA, rng);
        if(RVdata._multi)
        {
            for(int i=0; i<jitters.size(); i++)
                J_RV_prior->perturb(jitters[i], rng);
        }
        else
        {
            J_RV_prior->perturb(jitter_RV, rng);
        }
        
        if (studentt)
        {
            nu_GAIA_prior->perturb(nu_GAIA, rng);
            nu_RV_prior->perturb(nu_RV, rng);
        }


        if (known_object)
        {
            remove_known_object();

            for (int i=0; i<n_known_object; i++){
                KO_Pprior[i]->perturb(KO_P[i], rng);
                KO_Mprior[i]->perturb(KO_M[i], rng);
                KO_eprior[i]->perturb(KO_e[i], rng);
                KO_phiprior[i]->perturb(KO_phi[i], rng);
                KO_omegaprior[i]->perturb(KO_omega[i], rng);
                KO_cosiprior[i]->perturb(KO_cosi[i], rng);
                KO_Omegaprior[i]->perturb(KO_Omega[i], rng);
            }

            add_known_object();
        }
        
    }
    else //perturb background solution
    {
        //subtract astrometric solution
        for(size_t i=0; i<mu_GAIA.size(); i++)
        {
            mu_GAIA[i] += -(da + mua * (GAIAdata.t[i]-GAIAdata.M0_epoch)) * sin(GAIAdata.psi[i]) - (dd + mud * (GAIAdata.t[i]-GAIAdata.M0_epoch)) * cos(GAIAdata.psi[i]) - plx*GAIAdata.pf[i];
        }
        // propose new parameters
        da_prior->perturb(da, rng);
        dd_prior->perturb(dd, rng);
        mua_prior->perturb(mua, rng);
        mud_prior->perturb(mud, rng);
        plx_prior->perturb(plx, rng);

        //add astrometric solution back in
        for(size_t i=0; i<mu_GAIA.size(); i++)
        {
            mu_GAIA[i] += (da + mua * (GAIAdata.t[i]-GAIAdata.M0_epoch)) * sin(GAIAdata.psi[i]) + (dd + mud * (GAIAdata.t[i]-GAIAdata.M0_epoch)) * cos(GAIAdata.psi[i]) + plx*GAIAdata.pf[i];

        }
        
        for(size_t i=0; i<mu_RV.size(); i++)
        {
            //subtract vsys
            mu_RV[i] -= background;
            if(trend) {
                mu_RV[i] -= slope * (RVdata.t[i] - tmid) +
                            quadr * pow(RVdata.t[i] - tmid, 2) +
                            cubic * pow(RVdata.t[i] - tmid, 3);
            }
            if(RVdata._multi) {
                for(size_t j=0; j<offsets.size(); j++){
                    if (RVdata.obsi[i] == j+1) { mu_RV[i] -= offsets[j]; }
                }
            }
    
            if(indicator_correlations) {
                for(size_t j = 0; j < RVdata.number_indicators; j++){
                    mu_RV[i] -= betas[j] * actind[j][i];
                }
            }
        }
        
        // propose new vsys
        Cprior->perturb(background, rng);

        // propose new instrument offsets
        if (RVdata._multi){
            for(unsigned j=0; j<offsets.size(); j++){
                individual_offset_prior[j]->perturb(offsets[j], rng);
            }
        }

        // propose new slope
        if(trend) {
            if (degree >= 1) slope_prior->perturb(slope, rng);
            if (degree >= 2) quadr_prior->perturb(quadr, rng);
            if (degree == 3) cubic_prior->perturb(cubic, rng);
        }

        // propose new indicator correlations
        if(indicator_correlations){
            for(size_t j = 0; j < RVdata.number_indicators; j++){
                betaprior->perturb(betas[j], rng);
            }
        }
        
        for(size_t i=0; i<mu_RV.size(); i++)
        {
            mu_RV[i] += background;
            if(trend) {
                mu_RV[i] += slope * (RVdata.t[i] - tmid) +
                            quadr * pow(RVdata.t[i] - tmid, 2) +
                            cubic * pow(RVdata.t[i] - tmid, 3);
            }
            if(RVdata._multi) {
                for(size_t j=0; j<offsets.size(); j++){
                    if (RVdata.obsi[i] == j+1) { mu_RV[i] += offsets[j]; }
                }
            }

            if(indicator_correlations) {
                for(size_t j = 0; j < RVdata.number_indicators; j++){
                    mu_RV[i] += betas[j]*actind[j][i];
                }
            }
        }
        
    }


    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Perturb took ";
    cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6;
    cout << " ms" << std::endl;
    #endif

    return logH;
}

/**
 * Calculate the log-likelihood for the current values of the parameters.
 * 
 * @return double the log-likelihood
*/
double RVGAIAmodel::log_likelihood() const
{
    size_t N_GAIA = GAIAdata.N();
    size_t N_RV = RVdata.N();
    
    const auto& w = GAIAdata.get_w();
    const auto& wsig = GAIAdata.get_wsig();
    
    const auto& y = RVdata.get_y();
    const auto& sig = RVdata.get_sig();
    const auto& obsi = RVdata.get_obsi();

    double logL = 0.;


    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif
    
    double jit_GAIA = jitter_GAIA;

    if (studentt){
        // The following code calculates the log likelihood 
        // in the case of a t-Student model
        double var, jit;
        for(size_t i=0; i<N_GAIA; i++)
        {
            var = wsig[i]*wsig[i] +jit_GAIA*jit_GAIA;

            logL += std::lgamma(0.5*(nu_GAIA + 1.)) - std::lgamma(0.5*nu_GAIA)
                    - 0.5*log(M_PI*nu_GAIA) - 0.5*log(var)
                    - 0.5*(nu_GAIA + 1.)*log(1. + pow(w[i] - mu_GAIA[i], 2)/var/nu_GAIA);
        }
        
        for(size_t i=0; i<N_RV; i++)
        {
            if(RVdata._multi)
            {
                jit = jitters[obsi[i]-1];
                var = sig[i]*sig[i] + jit*jit;
            }
            else
                var = sig[i]*sig[i] + jitter_RV*jitter_RV;

            logL += std::lgamma(0.5*(nu_RV + 1.)) - std::lgamma(0.5*nu_RV)
                    - 0.5*log(M_PI*nu_RV) - 0.5*log(var)
                    - 0.5*(nu_RV + 1.)*log(1. + pow(y[i] - mu_RV[i], 2)/var/nu_RV);
        }

    }

    else{
        // The following code calculates the log likelihood
        // in the case of a Gaussian likelihood
        double var, jit;
        for(size_t i=0; i<N_GAIA; i++)
        {
            var = wsig[i]*wsig[i] + jit_GAIA*jit_GAIA;

            logL += - halflog2pi - 0.5*log(var)
                    - 0.5*(pow(w[i] - mu_GAIA[i], 2)/var);
        }
        
        for(size_t i=0; i<N_RV; i++)
        {
            if(RVdata._multi)
            {
                jit = jitters[obsi[i]-1];
                var = sig[i]*sig[i] + jit*jit;
            }
            else
                var = sig[i]*sig[i] + jitter_RV*jitter_RV;
    
            logL += - halflog2pi - 0.5*log(var)
                    - 0.5*(pow(y[i] - mu_RV[i], 2)/var);
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


void RVGAIAmodel::print(std::ostream& out) const
{
    // output precision
    out.setf(ios::fixed,ios::floatfield);
    out.precision(8);

    out<<jitter_GAIA<<'\t';
    
    if (RVdata._multi)
    {
        for(int j=0; j<jitters.size(); j++)
            out<<jitters[j]<<'\t';
    }
    else
        out<<jitter_RV<<'\t';

    if(trend)
    {
        out.precision(15);
        if (degree >= 1) out << slope << '\t';
        if (degree >= 2) out << quadr << '\t';
        if (degree == 3) out << cubic << '\t';
        out.precision(8);
    }
        
    if (RVdata._multi){
        for(int j=0; j<offsets.size(); j++){
            out<<offsets[j]<<'\t';
        }
    }

    if(indicator_correlations){
        for(int j=0; j<RVdata.number_indicators; j++){
            out<<betas[j]<<'\t';
        }
    }

    out.precision(24);
    
    out << da << '\t';
    out << dd << '\t';
    out << mua << '\t';
    out << mud << '\t';
    out << plx << '\t';
    
    out.precision(8);

    //auto data = get_data();

    if(known_object){ // KO mode!
        for (auto P: KO_P) out << P << "\t";
        for (auto M: KO_M) out << M << "\t";
        for (auto phi: KO_phi) out << phi << "\t";
        for (auto e: KO_e) out << e << "\t";
        for (auto w: KO_omega) out << w << "\t";
        for (auto cosi: KO_cosi) out << cosi << "\t";
        for (auto Om: KO_Omega) out << Om << "\t";
    }

    planets.print(out);

    out << ' ' << staleness << ' ';

    if (studentt)
    {
        out << '\t' << nu_GAIA << '\t';
        out << '\t' << nu_RV << '\t';
    }
        
    out << background;

}


string RVGAIAmodel::description() const
{
    string desc;
    string sep = "   ";

    desc += "jitter_GAIA"+sep;
    
    if (RVdata._multi)
    {
        for(int j=0; j<jitters.size(); j++)
           desc += "jitter_RV" + std::to_string(j+1) + sep;
    }
    else
        desc += "jitter_RV" + sep;

    if(trend)
    {
        if (degree >= 1) desc += "slope" + sep;
        if (degree >= 2) desc += "quadr" + sep;
        if (degree == 3) desc += "cubic" + sep;
    }


    if (RVdata._multi){
        for(unsigned j=0; j<offsets.size(); j++)
            desc += "offset" + std::to_string(j+1) + sep;
    }

    if(indicator_correlations){
        for(int j=0; j<RVdata.number_indicators; j++){
            desc += "beta" + std::to_string(j+1) + sep;
        }
    }

    desc += "da" + sep;
    desc += "dd" + sep;
    desc += "mua" + sep;
    desc += "mud" + sep;
    desc += "parallax" + sep;

    //auto data = get_data();

    if(known_object) { // KO mode!
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_P" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_M" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_phi" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_ecc" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_omega" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_cosi" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_Omega" + std::to_string(i) + sep;
    }

    desc += "ndim" + sep + "maxNp" + sep;

    desc += "Np" + sep;

    int maxpl = planets.get_max_num_components();
    if (maxpl > 0) {
        for(int i = 0; i < maxpl; i++) desc += "P" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "phi" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "ecc" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "M" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "w" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "cosi" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "W" + std::to_string(i) + sep;
    }

    desc += "staleness" + sep;
    if (studentt)
    {
        desc += "nu_GAIA" + sep;
        desc += "nu_RV" + sep;
    }
        
    desc += "vsys";
    
    return desc;
}

/**
 * Save the options of the current model in a INI file.
 * 
*/
void RVGAIAmodel::save_setup() {
    std::fstream fout("kima_model_setup.txt", std::ios::out);
    fout << std::boolalpha;

    time_t rawtime;
    time (&rawtime);
    fout << ";" << ctime(&rawtime) << endl;

    fout << "[kima]" << endl;

    fout << "model: " << "RVGAIAmodel" << endl << endl;
    fout << "fix: " << fix << endl;
    fout << "npmax: " << npmax << endl << endl;
    
    fout << "trend: " << trend << endl;
    fout << "degree: " << degree << endl;
    fout << "multi_instrument: " << RVdata._multi << endl;
    fout << "known_object: " << known_object << endl;
    fout << "n_known_object: " << n_known_object << endl;
    fout << "studentt: " << studentt << endl;
    fout << endl;

    fout << endl;

    fout << "[GAIAdata]" << endl;
    fout << "file: " << GAIAdata._datafile << endl;
    fout << "skip: " << GAIAdata._skip << endl;

    
    fout << "[data]" << endl;
    fout << "file: " << RVdata._datafile << endl;
    fout << "units: " << RVdata._units << endl;
    fout << "skip: " << RVdata._skip << endl;
    fout << "multi: " << RVdata._multi << endl;

    fout << "files: ";
    for (auto f: RVdata._datafiles)
        fout << f << ",";
    fout << endl;

    fout.precision(15);
    fout << "M0_epoch: " << GAIAdata.M0_epoch << endl;
    fout.precision(6);

    fout << endl;

    fout << "[priors.general]" << endl;
    fout << "Cprior: " << *Cprior << endl;
    fout << "J_GAIA_prior: " << *J_GAIA_prior << endl;
    fout << "J_RV_prior: " << *J_RV_prior << endl;
    
    if (trend){
        if (degree >= 1) fout << "slope_prior: " << *slope_prior << endl;
        if (degree >= 2) fout << "quadr_prior: " << *quadr_prior << endl;
        if (degree == 3) fout << "cubic_prior: " << *cubic_prior << endl;
    }

    if (RVdata._multi) {
        fout << "offsets_prior: " << *offsets_prior << endl;
        int i = 0;
        for (auto &p : individual_offset_prior) {
            fout << "individual_offset_prior[" << i << "]: " << *p << endl;
            i++;
        }
    }

    fout << "da_prior: " << *da_prior << endl;
    fout << "dd_prior: " << *dd_prior << endl;
    fout << "mua_prior: " << *mua_prior << endl;
    fout << "mud_prior: " << *mud_prior << endl;
    fout << "parallax_prior: " << *plx_prior << endl;

    if (studentt)
    {
        fout << "nu_GAIA_prior: " << *nu_GAIA_prior << endl;
        fout << "nu_RV_prior: " << *nu_RV_prior << endl;
    }

    if (planets.get_max_num_components()>0){
        auto conditional = planets.get_conditional_prior();

        fout << endl << "[priors.planets]" << endl;
        fout << "Pprior: " << *conditional->Pprior << endl;
        fout << "phiprior: " << *conditional->phiprior << endl;
        fout << "eprior: " << *conditional->eprior << endl;
        fout << "Mprior: " << *conditional->Mprior << endl;
        fout << "omegaprior: " << *conditional->omegaprior << endl;
        fout << "cosiprior: " << *conditional->cosiprior << endl;
        fout << "Omegaprior: " << *conditional->Omegaprior << endl;
    }

    if (known_object) {
        fout << endl << "[priors.known_object]" << endl;
        for(int i=0; i<n_known_object; i++){
            fout << "Pprior_" << i << ": " << *KO_Pprior[i] << endl;
            fout << "Mprior_" << i << ": " << *KO_Mprior[i] << endl;
            fout << "eprior_" << i << ": " << *KO_eprior[i] << endl;
            fout << "phiprior_" << i << ": " << *KO_phiprior[i] << endl;
            fout << "wprior_" << i << ": " << *KO_omegaprior[i] << endl;
            fout << "cosiprior_" << i << ": " << *KO_cosiprior[i] << endl;
            fout << "Wprior_" << i << ": " << *KO_Omegaprior[i] << endl;
        }
    }

    fout << endl;
    fout.close();
}

using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;

auto RVGAIAMODEL_DOC = R"D(
Combined analysis of Gaia epoch astrometry and radial velocity timeseries. Implements a sum-of-Keplerians model where the number of Keplerians can be free.
This model assumes white, uncorrelated noise. Planets are to be given Mass priors in Solar-Mass.

Args:
    fix (bool, default=True):
        whether the number of Keplerians should be fixed
    npmax (int, default=0):
        maximum number of Keplerians
    GAIAdata (GAIAdata):
        the astrometric data
    RVdata (RVData):
        the radial velocity data
)D";

class RVGAIAmodel_publicist : public RVGAIAmodel
{
    public:
        using RVGAIAmodel::studentt;
        using RVGAIAmodel::star_mass;
        using RVGAIAmodel::fix;
        using RVGAIAmodel::npmax;
        using RVGAIAmodel::known_object;
        using RVGAIAmodel::n_known_object;
        using RVGAIAmodel::trend;
        using RVGAIAmodel::degree;
        using RVGAIAmodel::indicator_correlations;
        using RVGAIAmodel::GAIAdata;
        using RVGAIAmodel::RVdata;
};


NB_MODULE(RVGAIAmodel, m) {
    // bind RVConditionalPrior so it can be returned
    bind_RVGAIAConditionalPrior(m);

    nb::class_<RVGAIAmodel>(m, "RVGAIAmodel")
        .def(nb::init<bool&, int&, GAIAData&, RVData&>(), "fix"_a, "npmax"_a, "GAIAData"_a, "RVData"_a, RVGAIAMODEL_DOC)
        //

        .def_rw("studentt", &RVGAIAmodel_publicist::studentt,
                "use a Student-t distribution for the likelihood (instead of Gaussian)")
        .def_rw("fix", &RVGAIAmodel_publicist::fix,
                "whether the number of Keplerians is fixed")
        .def_rw("npmax", &RVGAIAmodel_publicist::npmax,
                "maximum number of Keplerians")
        .def_ro("GAIAdata", &RVGAIAmodel_publicist::GAIAdata,
                "the data")
        .def_ro("RVdata", &RVGAIAmodel_publicist::RVdata,
                "the data")
                
        //
        .def_rw("trend", &RVGAIAmodel_publicist::trend,
                "whether the model includes a polynomial trend")
        .def_rw("degree", &RVGAIAmodel_publicist::degree,
                "degree of the polynomial trend")

        // KO mode
        .def("set_known_object", &RVGAIAmodel::set_known_object)
        .def_prop_ro("known_object", [](RVGAIAmodel &m) { return m.get_known_object(); },
                     "whether the model includes (better) known extra Keplerian curve(s)")
        .def_prop_ro("n_known_object", [](RVGAIAmodel &m) { return m.get_n_known_object(); },
                     "how many known objects")
                     
        //
        .def_rw("star_mass", &RVGAIAmodel_publicist::star_mass,
                "stellar mass [Msun]")
//         .def_rw("enforce_stability", &RVGAIAmodel_publicist::enforce_stability, 
//                 "whether to enforce AMD-stability")
        
        //
        .def_rw("indicator_correlations", &RVGAIAmodel_publicist::indicator_correlations, 
                "include in the model linear correlations with indicators")


        // priors
        .def_prop_rw("Cprior",
            [](RVGAIAmodel &m) { return m.Cprior; },
            [](RVGAIAmodel &m, distribution &d) { m.Cprior = d; },
            "Prior for the systemic velocity")
        .def_prop_rw("J_GAIA_prior",
            [](RVGAIAmodel &m) { return m.J_GAIA_prior; },
            [](RVGAIAmodel &m, distribution &d) { m.J_GAIA_prior = d; },
            "Prior for the extra white noise (jitter) for GAIA data")
        .def_prop_rw("J_RV_prior",
            [](RVGAIAmodel &m) { return m.J_RV_prior; },
            [](RVGAIAmodel &m, distribution &d) { m.J_RV_prior = d; },
            "Prior for the extra white noise (jitter) for RV data")
        .def_prop_rw("nu_GAIA_prior",
            [](RVGAIAmodel &m) { return m.nu_GAIA_prior; },
            [](RVGAIAmodel &m, distribution &d) { m.nu_GAIA_prior = d; },
            "Prior for the degrees of freedom of the Student-t likelihood for GAIA data")
        .def_prop_rw("nu_RV_prior",
            [](RVGAIAmodel &m) { return m.nu_RV_prior; },
            [](RVGAIAmodel &m, distribution &d) { m.nu_RV_prior = d; },
            "Prior for the degrees of freedom of the Student-t likelihood for RV data")
            
        .def_prop_rw("slope_prior",
            [](RVGAIAmodel &m) { return m.slope_prior; },
            [](RVGAIAmodel &m, distribution &d) { m.slope_prior = d; },
            "Prior for the slope")
        .def_prop_rw("quadr_prior",
            [](RVGAIAmodel &m) { return m.quadr_prior; },
            [](RVGAIAmodel &m, distribution &d) { m.quadr_prior = d; },
            "Prior for the quadratic coefficient of the trend")
        .def_prop_rw("cubic_prior",
            [](RVGAIAmodel &m) { return m.cubic_prior; },
            [](RVGAIAmodel &m, distribution &d) { m.cubic_prior = d; },
            "Prior for the cubic coefficient of the trend")

        .def_prop_rw("offsets_prior",
            [](RVGAIAmodel &m) { return m.offsets_prior; },
            [](RVGAIAmodel &m, distribution &d) { m.offsets_prior = d; },
            "Common prior for the between-instrument offsets")
        .def_prop_rw("individual_offset_prior",
            [](RVGAIAmodel &m) { return m.individual_offset_prior; },
            [](RVGAIAmodel &m, std::vector<distribution>& vd) { m.individual_offset_prior = vd; },
            "Common prior for the between-instrument offsets")
            
        .def_prop_rw("da_prior",
            [](RVGAIAmodel &m) { return m.da_prior; },
            [](RVGAIAmodel &m, distribution &d) { m.da_prior = d; },
            "Prior for the offset in right-ascension (mas)")
        .def_prop_rw("dd_prior",
            [](RVGAIAmodel &m) { return m.dd_prior; },
            [](RVGAIAmodel &m, distribution &d) { m.dd_prior = d; },
            "Prior for the the offset in declination (mas)")
        .def_prop_rw("mua_prior",
            [](RVGAIAmodel &m) { return m.mua_prior; },
            [](RVGAIAmodel &m, distribution &d) { m.mua_prior = d; },
            "Prior for the proper-motion in right-ascension")
        .def_prop_rw("mud_prior",
            [](RVGAIAmodel &m) { return m.mud_prior; },
            [](RVGAIAmodel &m, distribution &d) { m.mud_prior = d; },
            "Prior for the proper-motion in declination")
        .def_prop_rw("parallax_prior",
            [](RVGAIAmodel &m) { return m.plx_prior; },
            [](RVGAIAmodel &m, distribution &d) { m.plx_prior = d; },
            "Prior for the parallax")

        // known object priors
        // ? should these setters check if known_object is true?
        .def_prop_rw("KO_Pprior",
                     [](RVGAIAmodel &m) { return m.KO_Pprior; },
                     [](RVGAIAmodel &m, std::vector<distribution>& vd) { m.KO_Pprior = vd; },
                     "Prior for KO orbital period(s)")
        .def_prop_rw("KO_Mprior",
                     [](RVGAIAmodel &m) { return m.KO_Mprior; },
                     [](RVGAIAmodel &m, std::vector<distribution>& vd) { m.KO_Mprior = vd; },
                     "Prior for KO mass(es) (M_sun)")
        .def_prop_rw("KO_eprior",
                     [](RVGAIAmodel &m) { return m.KO_eprior; },
                     [](RVGAIAmodel &m, std::vector<distribution>& vd) { m.KO_eprior = vd; },
                     "Prior for KO eccentricity(ies)")
        .def_prop_rw("KO_omegaprior",
                     [](RVGAIAmodel &m) { return m.KO_omegaprior; },
                     [](RVGAIAmodel &m, std::vector<distribution>& vd) { m.KO_omegaprior = vd; },
                     "Prior for KO argument(s) of periastron")
        .def_prop_rw("KO_phiprior",
                     [](RVGAIAmodel &m) { return m.KO_phiprior; },
                     [](RVGAIAmodel &m, std::vector<distribution>& vd) { m.KO_phiprior = vd; },
                     "Prior for KO mean anomaly(ies)")
        .def_prop_rw("KO_cosiprior",
                     [](RVGAIAmodel &m) { return m.KO_cosiprior; },
                     [](RVGAIAmodel &m, std::vector<distribution>& vd) { m.KO_cosiprior = vd; },
                     "Prior for cosine of KO inclination(s)")
        .def_prop_rw("KO_Omegaprior",
                     [](RVGAIAmodel &m) { return m.KO_Omegaprior; },
                     [](RVGAIAmodel &m, std::vector<distribution>& vd) { m.KO_Omegaprior = vd; },
                     "Prior for KO longitude(s) of ascending node")

        // conditional object
        .def_prop_rw("conditional",
                     [](RVGAIAmodel &m) { return m.get_conditional_prior(); },
                     [](RVGAIAmodel &m, RVGAIAConditionalPrior& c) { /* does nothing */ });
}


