#include "RVFWHMRHKmodel.h"

using namespace Eigen;
#define TIMING false
#define DEBUG false

const double halflog2pi = 0.5*log(2.*M_PI);


void RVFWHMRHKmodel::initialize_from_data(RVData& data)
{
    if (data.actind.size() < 4) // need at least two activity indicator (the FWHM and R'HK)
    {
        std::string msg = "kima: RVFWHMRHKmodel: no data for activity indicators (FWHM and R'HK)";
        throw std::runtime_error(msg);
    }

    // inst1    inst2   inst3          --   n_instruments
    //      off1    off2        RV     |
    //      off3    off4        FWHM   |-   3 * n_instruments - 3
    //      off5    off6        R'HK   |
    // jit1     jit2    jit3    RV     |
    // jit4     jit5    jit6    FWHM   |-   3 * n_instruments
    // jit7     jit8    jit9    R'HK   |

    offsets.resize(3 * data.number_instruments - 3);
    jitters.resize(3 * data.number_instruments);
    individual_offset_prior.resize(data.number_instruments - 1);
    individual_offset_fwhm_prior.resize(data.number_instruments - 1);
    individual_offset_rhk_prior.resize(data.number_instruments - 1);

    #if DEBUG
        std::cout << "RVFWHMRHKmodel::initialize_from_data" << std::endl;
        std::cout << "n_instruments: " << data.number_instruments << std::endl;
        std::cout << "offsets.size(): " << offsets.size() << std::endl;
        std::cout << "jitters.size(): " << jitters.size() << std::endl;
    #endif

    size_t N = data.N();

    // resize RV, FWHM model vectors
    mu.resize(N);
    mu_fwhm.resize(N);
    mu_rhk.resize(N);
    // resize covariance matrices
    C.resize(N, N);
    C_fwhm.resize(N, N);
    C_rhk.resize(N, N);

    // copy uncertainties
    sig_copy = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(data.sig.data(), N);
    auto sig_fwhm = data.get_actind()[1]; // temporary
    sig_fwhm_copy = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(sig_fwhm.data(), N);
    auto sig_rhk = data.get_actind()[3]; // temporary
    sig_rhk_copy = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(sig_rhk.data(), N);

    // set default conditional priors that depend on data
    auto conditional = planets.get_conditional_prior();
    conditional->set_default_priors(data);
}

void RVFWHMRHKmodel::set_known_object(size_t n)
{
    known_object = true;
    n_known_object = n;

    KO_Pprior.resize(n); KO_Kprior.resize(n); KO_eprior.resize(n); KO_phiprior.resize(n); KO_wprior.resize(n);
    KO_P.resize(n); KO_K.resize(n); KO_e.resize(n); KO_phi.resize(n); KO_w.resize(n);
}

void RVFWHMRHKmodel::set_transiting_planet(size_t n)
{
    transiting_planet = true;
    n_transiting_planet = n;

    TR_Pprior.resize(n); TR_Kprior.resize(n); TR_eprior.resize(n); TR_Tcprior.resize(n); TR_wprior.resize(n);
    TR_P.resize(n); TR_K.resize(n); TR_e.resize(n); TR_Tc.resize(n); TR_w.resize(n);
}

/// set default priors if the user didn't change them
void RVFWHMRHKmodel::setPriors()  // BUG: should be done by only one thread!
{
    auto defaults = DefaultPriors(data);

    // systemic velocity
    if (!Cprior)
    Cprior = defaults.get("Cprior");
    
    // "systemic FWHM"
    if (!Cfwhm_prior)
        Cfwhm_prior = defaults.get("Cfwhm_prior");
    
    // "systemic R'HK"
    if (!Crhk_prior)
        Crhk_prior = defaults.get("Crhk_prior");

    // jitter for the RVs
    if (!Jprior)
        Jprior = defaults.get("Jprior");

    // jitter for the FWHM
    if (!Jfwhm_prior)
        Jfwhm_prior = defaults.get("Jfwhm_prior");

    // jitter for the R'HK
    if (!Jrhk_prior)
        Jrhk_prior = defaults.get("Jrhk_prior");

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
    if (data._multi)
    {
        if (!offsets_prior)
            offsets_prior = defaults.get("offsets_prior");
        if (!offsets_fwhm_prior)
            offsets_fwhm_prior = defaults.get("offsets_fwhm_prior");

        if (!offsets_rhk_prior) {
            auto minRHK = *min_element(data.actind[2].begin(), data.actind[2].end());
            auto maxRHK = *max_element(data.actind[2].begin(), data.actind[2].end());
            auto spanRHK = maxRHK - minRHK;
            offsets_rhk_prior = make_prior<Uniform>( -spanRHK, spanRHK );
        }

        for (size_t j = 0; j < data.number_instruments - 1; j++)
        {
            // if individual_offset_prior is not (re)defined, assume offsets_prior
            if (!individual_offset_prior[j])
                individual_offset_prior[j] = offsets_prior;
            if (!individual_offset_fwhm_prior[j])
                individual_offset_fwhm_prior[j] = offsets_fwhm_prior;
            if (!individual_offset_rhk_prior[j])
                individual_offset_rhk_prior[j] = offsets_rhk_prior;
        }
    }

    // KO mode!
    if (known_object)
    {
        for (int i = 0; i < n_known_object; i++)
        {
            if (!KO_Pprior[i] || !KO_Kprior[i] || !KO_eprior[i] || !KO_phiprior[i] || !KO_wprior[i])
            {
                std::string msg = "When known_object=true, must set priors for each of KO_Pprior, KO_Kprior, KO_eprior, KO_phiprior, KO_wprior";
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

    /* GP parameters */
    if (kernel != qp) {
        std::string msg = "kima: RVFWHMRHKmodel: only the QP kernel is currently supported";
        throw std::runtime_error(msg);
    }

    switch (kernel)
    {
    case qp:

        // eta1's are never shared, so they get default priors if
        // they haven't been set
        if (!eta1_prior)
        {
            eta1_prior = defaults.get("eta1_prior");
        }

        if (!eta1_fwhm_prior)
        {
            eta1_fwhm_prior = defaults.get("eta1_fwhm_prior");
        }

        if (!eta1_rhk_prior)
        {
            eta1_rhk_prior = defaults.get("eta1_rhk_prior");
        }

        // eta2 can be shared
        if (!eta2_prior)
        {
            eta2_prior = defaults.get("eta2_prior");
        }

        if (share_eta2)
        {
            eta2_fwhm_prior = eta2_prior;
            eta2_rhk_prior = eta2_prior;
        }
        else
        {
            if (!eta2_fwhm_prior)
                eta2_fwhm_prior = defaults.get("eta2_fwhm_prior");
            if (!eta2_rhk_prior)
                eta2_rhk_prior = defaults.get("eta2_rhk_prior");
        }

        // eta3 can be shared
        if (!eta3_prior)
        {
            eta3_prior = defaults.get("eta3_prior");
        }
        if (share_eta3)
        {
            eta3_fwhm_prior = eta3_prior;
            eta3_rhk_prior = eta3_prior;
        }
        else
        {
            if (!eta3_fwhm_prior)
                eta3_fwhm_prior = defaults.get("eta3_fwhm_prior");
            if (!eta3_rhk_prior)
                eta3_rhk_prior = defaults.get("eta3_rhk_prior");
        }

        // eta4 can be shared
        if (!eta4_prior)
        {
            eta4_prior = defaults.get("eta4_prior");
        }
        if (share_eta4)
        {
            eta4_fwhm_prior = eta4_prior;
            eta4_rhk_prior = eta4_prior;
        }
        else
        {
            if (!eta4_fwhm_prior)
                eta4_fwhm_prior = defaults.get("eta4_fwhm_prior");
            if (!eta4_rhk_prior)
                eta4_rhk_prior = defaults.get("eta4_rhk_prior");
        }
        break;
    
    default:
        break;
    }

    if (magnetic_cycle_kernel) {
        if (!eta5_prior)
            eta5_prior = make_prior<ModifiedLogUniform>(0.1, 5*data.get_max_RV_span());
        if (!eta5_fwhm_prior)
            eta5_fwhm_prior = make_prior<ModifiedLogUniform>(0.1, 5*data.get_actind_span(0));
        if (!eta5_rhk_prior)
            eta5_rhk_prior = make_prior<ModifiedLogUniform>(0.1, 5*data.get_actind_span(2));

        if (!eta6_prior)
            eta6_prior = make_prior<LogUniform>(365, 10*data.get_timespan());
        if (!eta7_prior)
            eta7_prior = make_prior<Uniform>(1, 10);
    }

    #if DEBUG
    std::cout << std::endl << "setPriors done" << std::endl;
    #endif

}


void RVFWHMRHKmodel::from_prior(RNG& rng)
{
    // preliminaries
    setPriors();
    save_setup();
    
    planets.from_prior(rng);
    planets.consolidate_diff();
    
    bkg = Cprior->generate(rng);
    bkg_fwhm = Cfwhm_prior->generate(rng);
    bkg_rhk = Crhk_prior->generate(rng);

    if(data._multi)
    {
        // inst1    inst2   inst3          --   n_instruments
        //      off1    off2        RV     |
        //      off3    off4        FWHM   |-   3 * n_instruments - 3
        //      off5    off6        R'HK   |
        // jit1     jit2    jit3    RV     |
        // jit4     jit5    jit6    FWHM   |-   3 * n_instruments
        // jit7     jit8    jit9    R'HK   |

        if (data._multi)
        {
            auto no = offsets.size();
            auto lim1 = no / 3;
            auto lim2 = 2 * no / 3;
            size_t j=0, k=0;

            // draw instrument offsets for the RVs
            for (size_t i = 0; i < lim1; i++) {
                offsets[i] = individual_offset_prior[i]->generate(rng);
            }
            // draw instrument offsets for the FWHM
            for (size_t i = lim1; i < lim2; i++)
            {
                offsets[i] = individual_offset_fwhm_prior[j]->generate(rng);
                j++;
            }
            // draw instrument offsets for the R'hk
            for (size_t i = lim2; i < no; i++)
            {
                offsets[i] = individual_offset_rhk_prior[k]->generate(rng);
                k++;
            }
        }

        {
            auto nj = jitters.size();
            auto lim1 = nj / 3;
            auto lim2 = 2 * nj / 3;
            // draw RV jitter
            for (size_t i = 0; i < lim1; i++)
            {
                jitters[i] = Jprior->generate(rng);
            }
            // draw FWHM jitter
            for (size_t i = lim1; i < lim2; i++)
            {
                jitters[i] = Jfwhm_prior->generate(rng);
            }
            // draw R'hk jitter
            for (size_t i = lim2; i < nj; i++)
            {
                jitters[i] = Jrhk_prior->generate(rng);
            }
        }
    }
    else
    {
        jitter = Jprior->generate(rng);
        jitter_fwhm = Jfwhm_prior->generate(rng);
        jitter_rhk = Jrhk_prior->generate(rng);
    }

    if(trend)
    {
        if (degree >= 1) slope = slope_prior->generate(rng);
        if (degree >= 2) quadr = quadr_prior->generate(rng);
        if (degree == 3) cubic = cubic_prior->generate(rng);
    }


    if (known_object) { // KO mode!
        for (int i=0; i<n_known_object; i++){
            KO_P[i] = KO_Pprior[i]->generate(rng);
            KO_K[i] = KO_Kprior[i]->generate(rng);
            KO_e[i] = KO_eprior[i]->generate(rng);
            KO_phi[i] = KO_phiprior[i]->generate(rng);
            KO_w[i] = KO_wprior[i]->generate(rng);
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

    // GP
    switch (kernel)
    {
    case qp:
        eta1 = eta1_prior->generate(rng);  // m/s
        eta1_fw = eta1_fwhm_prior->generate(rng);  // m/s
        eta1_rhk = eta1_rhk_prior->generate(rng);

        eta3 = eta3_prior->generate(rng); // days
        if (!share_eta3) {
            eta3_fw = eta3_fwhm_prior->generate(rng); // days
            eta3_rhk = eta3_rhk_prior->generate(rng); // days
        }

        eta2 = eta2_prior->generate(rng); // days
        if (!share_eta2) {
            eta2_fw = eta2_fwhm_prior->generate(rng); // days
            eta2_rhk = eta2_rhk_prior->generate(rng); // days
        }

        eta4 = eta4_prior->generate(rng);
        if (!share_eta4) {
            eta4_fw = eta4_fwhm_prior->generate(rng);
            eta4_rhk = eta4_rhk_prior->generate(rng);
        }
        break;

    default:
        break;
    }

    if (magnetic_cycle_kernel) {
        eta5 = eta5_prior->generate(rng);  // m/s
        eta5_fw = eta5_fwhm_prior->generate(rng);  // m/s
        eta5_rhk = eta5_rhk_prior->generate(rng);  // 
        eta6 = eta6_prior->generate(rng);  // days
        eta7 = eta7_prior->generate(rng);
    }


    calculate_mu();
    calculate_mu_fwhm();
    calculate_mu_rhk();

    calculate_C();
    calculate_C_fwhm();
    calculate_C_rhk();

    #if DEBUG
    std::cout << std::endl << "from_prior done" << std::endl;
    #endif

}

/// @brief Calculate the full RV model
void RVFWHMRHKmodel::calculate_mu()
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
            for (size_t j = 0; j < offsets.size() / 3; j++)
            {
                for (size_t i = 0; i < N; i++)
                {
                    if (data.obsi[i] == j+1) { mu[i] += offsets[j]; }
                }
            }
        }

        if (known_object) { // KO mode!
            add_known_object();
        }

        if (transiting_planet) {
            add_transiting_planet();
        }
    }
    else // just updating (adding) planets
        staleness++;


    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif


    double P, K, phi, ecc, omega;
    for (size_t j = 0; j < components.size(); j++)
    {
        if (false) // hyperpriors
            P = exp(components[j][0]);
        else
            P = components[j][0];

        K = components[j][1];
        phi = components[j][2];
        ecc = components[j][3];
        omega = components[j][4];

        auto v = brandt::keplerian(data.t, P, K, ecc, omega, phi, data.M0_epoch);
        for (size_t i = 0; i < N; i++)
            mu[i] += v[i];
    }


    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Model eval took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif

    #if DEBUG
    std::cout << std::endl << "calculate_mu done" << std::endl;
    #endif

}


/// @brief Calculate the full FWHM model
void RVFWHMRHKmodel::calculate_mu_fwhm()
{
    size_t N = data.N();
    int Ni = data.Ninstruments();

    mu_fwhm.assign(mu_fwhm.size(), bkg_fwhm);

    if (data._multi) {
        auto obsi = data.get_obsi();
        for (size_t j = offsets.size() / 3; j < 2 * offsets.size() / 3; j++) {
            for (size_t i = 0; i < N; i++) {
                if (obsi[i] == j + 2 - Ni) {
                    mu_fwhm[i] += offsets[j];
                }
            }
        }
    }

    #if DEBUG
    std::cout << std::endl << "calculate_mu_fwhm done" << std::endl;
    #endif

}


/// @brief Calculate the full RHK model
void RVFWHMRHKmodel::calculate_mu_rhk()
{
    size_t N = data.N();
    int Ni = data.Ninstruments();

    mu_rhk.assign(mu_rhk.size(), bkg_rhk);

    if (data._multi) {
        auto obsi = data.get_obsi();
        for (size_t j = 2 * offsets.size() / 3; j < offsets.size(); j++) {
            for (size_t i = 0; i < N; i++) {
                if (obsi[i] == j + 2 - Ni) {
                    mu_rhk[i] += offsets[j];
                }
            }
        }
    }

    #if DEBUG
    std::cout << std::endl << "calculate_mu_rhk done" << std::endl;
    #endif
}



/// @brief Fill the GP covariance matrix
void RVFWHMRHKmodel::calculate_C()
{
    size_t N = data.N();

    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    switch (kernel)
    {
    case qp:
        C = QP(data.t, eta1, eta2, eta3, eta4);
        break;
    default:
        break;
    }

    if (data._multi)
    {
        for (size_t i = 0; i < data.N(); i++)
        {
            double jit = jitters[data.obsi[i] - 1];
            C(i, i) += data.sig[i] * data.sig[i] + jit * jit;
        }
    }
    else
    {
        C.diagonal().array() += sig_copy.array().square() + jitter * jitter;
    }

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "GP build matrix: ";
    cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
    cout << " ns" << "\t"; // << std::endl;
    #endif

    #if DEBUG
    std::cout << std::endl << "calculate_C done" << std::endl;
    #endif
}


/// @brief Fill the GP covariance matrix
void RVFWHMRHKmodel::calculate_C_fwhm()
{
    auto sig = data.get_actind()[1];    

    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    switch (kernel)
    {
    case qp:
        C_fwhm = QP(data.t, eta1_fw, eta2_fw, eta3_fw, eta4_fw);
        break;
    default:
        break;
    }

    if (data._multi)
    {
        for (size_t i = 0; i < data.N(); i++)
        {
            double jit = jitters[jitters.size() / 3 + data.obsi[i] - 1];
            C_fwhm(i, i) += sig[i] * sig[i] + jit * jit;
        }
    }
    else
    {
        C_fwhm.diagonal().array() += sig_fwhm_copy.array().square() + jitter_fwhm * jitter_fwhm;
    }

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "GP build matrix: ";
    cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
    cout << " ns" << "\t"; // << std::endl;
    #endif

    #if DEBUG
    std::cout << std::endl << "calculate_C_fwhm done" << std::endl;
    #endif
}


/// @brief Fill the GP covariance matrix
void RVFWHMRHKmodel::calculate_C_rhk()
{
    auto sig = data.get_actind()[1];    

    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    switch (kernel)
    {
    case qp:
        C_rhk = QP(data.t, eta1_rhk, eta2_rhk, eta3_rhk, eta4_rhk);
        break;
    default:
        break;
    }

    if (data._multi)
    {
        for (size_t i = 0; i < data.N(); i++)
        {
            double jit = jitters[jitters.size() / 3 + data.obsi[i] - 1];
            C_rhk(i, i) += sig[i] * sig[i] + jit * jit;
        }
    }
    else
    {
        C_rhk.diagonal().array() += sig_rhk_copy.array().square() + jitter_rhk * jitter_rhk;
    }

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "GP build matrix: ";
    cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
    cout << " ns" << "\t"; // << std::endl;
    #endif

    #if DEBUG
    std::cout << std::endl << "calculate_C_rhk done" << std::endl;
    #endif
}



void RVFWHMRHKmodel::remove_known_object()
{
    for (int j = 0; j < n_known_object; j++)
    {
        auto v = brandt::keplerian(data.t, KO_P[j], KO_K[j], KO_e[j], KO_w[j], KO_phi[j], data.M0_epoch);
        for (size_t i = 0; i < data.N(); i++)
        {
            mu[i] -= v[i];
        }
    }
}

void RVFWHMRHKmodel::add_known_object()
{
    for (int j = 0; j < n_known_object; j++)
    {
        auto v = brandt::keplerian(data.t, KO_P[j], KO_K[j], KO_e[j], KO_w[j], KO_phi[j], data.M0_epoch);
        for (size_t i = 0; i < data.N(); i++)
        {
            mu[i] += v[i];
        }
    }
}

void RVFWHMRHKmodel::remove_transiting_planet()
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

void RVFWHMRHKmodel::add_transiting_planet()
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


int RVFWHMRHKmodel::is_stable() const
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


double RVFWHMRHKmodel::perturb(RNG& rng)
{
    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    auto actind = data.get_actind();
    double logH = 0.;
    double tmid = data.get_t_middle();

    int maxpl = planets.get_max_num_components();
    if(maxpl > 0 && rng.rand() <= 0.5) // perturb planet parameters
    {
        logH += planets.perturb(rng);
        planets.consolidate_diff();
        calculate_mu();
    }
    else if(rng.rand() <= 0.5) // perturb GP parameters
    {
        switch (kernel)
        {
        case qp:
            if(rng.rand() <= 0.25)
            {
                eta1_prior->perturb(eta1, rng);
                eta1_fwhm_prior->perturb(eta1_fw, rng);
                eta1_rhk_prior->perturb(eta1_rhk, rng);
            }
            else if(rng.rand() <= 0.33330)
            {
                eta3_prior->perturb(eta3, rng);
                if (share_eta3) {
                    eta3_fw = eta3;
                    eta3_rhk = eta3;
                }
                else {
                    eta3_fwhm_prior->perturb(eta3_fw, rng);
                    eta3_rhk_prior->perturb(eta3_rhk, rng);
                }
            }
            else if(rng.rand() <= 0.5)
            {
                eta2_prior->perturb(eta2, rng);
                if (share_eta2) {
                    eta2_fw = eta2;
                    eta2_rhk = eta2;
                }
                else {
                    eta2_fwhm_prior->perturb(eta2_fw, rng);
                    eta2_rhk_prior->perturb(eta2_rhk, rng);
                }
            }
            else
            {
                eta4_prior->perturb(eta4, rng);
                if (share_eta4) {
                    eta4_fw = eta4;
                    eta4_rhk = eta4;
                }
                else {
                    eta4_fwhm_prior->perturb(eta4_fw, rng);
                    eta4_rhk_prior->perturb(eta4_rhk, rng);
                }
            }
            break;

        default:
            break;
        }

        if (magnetic_cycle_kernel) {
            eta5_prior->perturb(eta5, rng);
            eta5_fwhm_prior->perturb(eta5_fw, rng);
            eta5_rhk_prior->perturb(eta5_rhk, rng);
            eta6_prior->perturb(eta6, rng);
            eta7_prior->perturb(eta7, rng);
        }

        calculate_C();
        calculate_C_fwhm();
        calculate_C_rhk();
    }

    else if(rng.rand() <= 0.5) // perturb jitter(s) + known_object
    {
        if(data._multi)
        {
            for (size_t i = 0; i < jitters.size() / 3; i++)
            {
                Jprior->perturb(jitters[i], rng);
            }
            for (size_t i = jitters.size() / 3; i < 2 * jitters.size() / 3; i++)
            {
                Jfwhm_prior->perturb(jitters[i], rng);
            }
            for (size_t i = 2 * jitters.size() / 3; i < jitters.size(); i++)
            {
                Jrhk_prior->perturb(jitters[i], rng);
            }
        }
        else
        {
            Jprior->perturb(jitter, rng);
            Jfwhm_prior->perturb(jitter_fwhm, rng);
            Jrhk_prior->perturb(jitter_rhk, rng);
        }

        calculate_C(); // recalculate covariance matrix
        calculate_C_fwhm(); // recalculate covariance matrix
        calculate_C_rhk(); // recalculate covariance matrix

        if (known_object)
        {
            remove_known_object();

            for (int i=0; i<n_known_object; i++){
                KO_Pprior[i]->perturb(KO_P[i], rng);
                KO_Kprior[i]->perturb(KO_K[i], rng);
                KO_eprior[i]->perturb(KO_e[i], rng);
                KO_phiprior[i]->perturb(KO_phi[i], rng);
                KO_wprior[i]->perturb(KO_w[i], rng);
            }

            add_known_object();
        }

        if (transiting_planet)
        {
            remove_transiting_planet();

            for (int i = 0; i < n_transiting_planet; i++)
            {
                TR_Pprior[i]->perturb(TR_P[i], rng);
                TR_Kprior[i]->perturb(TR_K[i], rng);
                TR_eprior[i]->perturb(TR_e[i], rng);
                TR_Tcprior[i]->perturb(TR_Tc[i], rng);
                TR_wprior[i]->perturb(TR_w[i], rng);
            }

            add_transiting_planet();
        }
    
    }
    else
    {
        for (size_t i = 0; i < mu.size(); i++)
        {
            mu[i] -= bkg;

            if(trend) {
                mu[i] -= slope * (data.t[i] - tmid) +
                         quadr * pow(data.t[i] - tmid, 2) +
                         cubic * pow(data.t[i] - tmid, 3);
            }

            if (data._multi)
            {
                for (size_t j = 0; j < offsets.size() / 3; j++)
                {
                    if (data.obsi[i] == j+1) { mu[i] -= offsets[j]; }
                }
            }
        }

        // propose new vsys
        Cprior->perturb(bkg, rng);
        Cfwhm_prior->perturb(bkg_fwhm, rng);
        Crhk_prior->perturb(bkg_rhk, rng);

        // propose new instrument offsets
        if (data._multi)
        {
            for (size_t j = 0; j < offsets.size() / 3; j++)
            {
                individual_offset_prior[j]->perturb(offsets[j], rng);
            }
            size_t k = 0;
            for (size_t j = offsets.size() / 3; j < 2 * offsets.size() / 3; j++)
            {
                individual_offset_fwhm_prior[k]->perturb(offsets[j], rng);
                k++;
            }
            size_t l = 0;
            for (size_t j = 2 * offsets.size() / 3; j < offsets.size(); j++)
            {
                individual_offset_rhk_prior[l]->perturb(offsets[j], rng);
                l++;
            }
        }

        // propose new slope
        if(trend) {
            if (degree >= 1) slope_prior->perturb(slope, rng);
            if (degree >= 2) quadr_prior->perturb(quadr, rng);
            if (degree == 3) cubic_prior->perturb(cubic, rng);
        }


        for(size_t i=0; i<mu.size(); i++)
        {
            mu[i] += bkg;

            if(trend) {
                mu[i] += slope * (data.t[i] - tmid) +
                         quadr * pow(data.t[i] - tmid, 2) +
                         cubic * pow(data.t[i] - tmid, 3);
            }

            if (data._multi)
            {
                for (size_t j = 0; j < offsets.size(); j++)
                {
                    if (data.obsi[i] == j+1) { mu[i] += offsets[j]; }
                }
            }
        }

        calculate_mu_fwhm();
        calculate_mu_rhk();
    }


    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Perturb took ";
    cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
    cout << " μs" << std::endl;
    #endif

    return logH;
}


double RVFWHMRHKmodel::log_likelihood() const
{
    size_t N = data.N();
    const auto& y = data.y;
    const auto& obsi = data.obsi;
    const auto fwhm = data.actind[0];
    const auto rhk = data.actind[2];

    double logL = 0.;

    if (enforce_stability){
        int stable = is_stable();
        if (stable != 0)
            return -std::numeric_limits<double>::infinity();
    }


    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    double logL_RV, logL_FWHM, logL_RHK;

    { // RVs
        VectorXd residual(y.size());
        for (size_t i = 0; i < y.size(); i++)
            residual(i) = y[i] - mu[i];

        Eigen::LLT<Eigen::MatrixXd> cholesky = C.llt();
        MatrixXd L = cholesky.matrixL();

        double logDeterminant = 0.;
        for (size_t i = 0; i < y.size(); i++)
            logDeterminant += 2. * log(L(i, i));

        VectorXd solution = cholesky.solve(residual);

        double exponent = 0.;
        for (size_t i = 0; i < y.size(); i++)
            exponent += residual(i) * solution(i);

        logL_RV = -0.5*y.size()*log(2*M_PI) - 0.5*logDeterminant - 0.5*exponent;
    }

    { // FWHM
        VectorXd residual(fwhm.size());
        for (size_t i = 0; i < fwhm.size(); i++)
            residual(i) = fwhm[i] - mu_fwhm[i];

        Eigen::LLT<Eigen::MatrixXd> cholesky = C_fwhm.llt();
        MatrixXd L = cholesky.matrixL();

        double logDeterminant = 0.;
        for (size_t i = 0; i < fwhm.size(); i++)
            logDeterminant += 2. * log(L(i, i));

        VectorXd solution = cholesky.solve(residual);

        double exponent = 0.;
        for (size_t i = 0; i < fwhm.size(); i++)
            exponent += residual(i) * solution(i);

        logL_FWHM = -0.5*fwhm.size()*log(2*M_PI) - 0.5*logDeterminant - 0.5*exponent;
    }

    { // R'HK
        VectorXd residual(rhk.size());
        for (size_t i = 0; i < rhk.size(); i++)
            residual(i) = rhk[i] - mu_rhk[i];

        Eigen::LLT<Eigen::MatrixXd> cholesky = C_rhk.llt();
        MatrixXd L = cholesky.matrixL();

        double logDeterminant = 0.;
        for (size_t i = 0; i < rhk.size(); i++)
            logDeterminant += 2. * log(L(i, i));

        VectorXd solution = cholesky.solve(residual);

        double exponent = 0.;
        for (size_t i = 0; i < rhk.size(); i++)
            exponent += residual(i) * solution(i);

        logL_RHK = -0.5*rhk.size()*log(2*M_PI) - 0.5*logDeterminant - 0.5*exponent;
    }

    logL = logL_RV + logL_FWHM + logL_RHK;

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


void RVFWHMRHKmodel::print(std::ostream& out) const
{
    // output precision
    out.setf(ios::fixed,ios::floatfield);
    out.precision(8);

    if (data._multi)
    {
        for (int j = 0; j < jitters.size(); j++)
            out << jitters[j] << '\t';
    }
    else
    {
        out << jitter << '\t';
        out << jitter_fwhm << '\t';
        out << jitter_rhk << '\t';
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
        for (int j = 0; j < offsets.size(); j++)
        {
            out << offsets[j] << '\t';
        }
    }

    // write GP parameters
    switch (kernel)
    {
    case qp:
        out << eta1 << '\t' << eta1_fw << '\t' << eta1_rhk << '\t';

        out << eta2 << '\t';
        if (!share_eta2) out << eta2_fw << '\t' << eta2_rhk << '\t';

        out << eta3 << '\t';
        if (!share_eta3) out << eta3_fw << '\t' << eta3_rhk << '\t';
        
        out << eta4 << '\t';
        if (!share_eta4) out << eta4_fw << '\t' << eta4_rhk << '\t';

        break;
    
    default:
        break;
    }

    if (magnetic_cycle_kernel)
        out << eta5 << '\t' << eta5_fw << '\t' << eta5_rhk << '\t' << eta6 << '\t' << eta7 << '\t';

    // write KO parameters
    if(known_object){ // KO mode!
        for (auto P: KO_P) out << P << "\t";
        for (auto K: KO_K) out << K << "\t";
        for (auto phi: KO_phi) out << phi << "\t";
        for (auto e: KO_e) out << e << "\t";
        for (auto w: KO_w) out << w << "\t";
    }

    if(transiting_planet){
        for (auto P: TR_P) out << P << "\t";
        for (auto K: TR_K) out << K << "\t";
        for (auto Tc: TR_Tc) out << Tc << "\t";
        for (auto e: TR_e) out << e << "\t";
        for (auto w: TR_w) out << w << "\t";
    }


    // write planet parameters
    planets.print(out);

    out << staleness << '\t';

    out << bkg_rhk  << '\t';
    out << bkg_fwhm << '\t';
    out << bkg;
}


string RVFWHMRHKmodel::description() const
{
    string desc;
    string sep = "   ";

    if (data._multi)
    {
        for(int j=0; j<jitters.size(); j++)
           desc += "jitter" + std::to_string(j+1) + sep;
    }
    else
    {
        desc += "jitter1" + sep;
        desc += "jitter2" + sep;
        desc += "jitter3" + sep;
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
    }

    // GP parameters
    switch (kernel)
    {
    case qp:
        desc += "eta1" + sep + "eta1_fwhm" + sep + "eta1_rhk" + sep;

        desc += "eta2" + sep;
        if (!share_eta2) desc += "eta2_fwhm" + sep + "eta2_rhk" + sep;

        desc += "eta3" + sep;
        if (!share_eta3) desc += "eta3_fwhm" + sep + "eta3_rhk" + sep;

        desc += "eta4" + sep;
        if (!share_eta4) desc += "eta4_fwhm" + sep + "eta4_rhk" + sep;

        break;

    default:
        break;
    }

    if (magnetic_cycle_kernel)
        desc += "eta5" + sep + "eta5_fw" + sep + "eta5_rhk" + sep + "eta6" + sep + "eta7" + sep;

    if(known_object) { // KO mode!
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_P" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_K" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_phi" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_ecc" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_w" + std::to_string(i) + sep;
    }

    if(transiting_planet) {
        for (int i = 0; i < n_transiting_planet; i++)
            desc += "TR_P" + std::to_string(i) + sep;
        for (int i = 0; i < n_transiting_planet; i++)
            desc += "TR_K" + std::to_string(i) + sep;
        for (int i = 0; i < n_transiting_planet; i++)
            desc += "TR_Tc" + std::to_string(i) + sep;
        for (int i = 0; i < n_transiting_planet; i++)
            desc += "TR_ecc" + std::to_string(i) + sep;
        for (int i = 0; i < n_transiting_planet; i++)
            desc += "TR_w" + std::to_string(i) + sep;
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
    }

    desc += "staleness" + sep;

    desc += "crhk" + sep;
    desc += "cfwhm" + sep;
    desc += "vsys";

    return desc;
}


void RVFWHMRHKmodel::save_setup() {
	std::fstream fout("kima_model_setup.txt", std::ios::out);
    fout << std::boolalpha;

    fout << "; " << timestamp() << endl;

    fout << "[kima]" << endl;

    fout << "model: " << "RVFWHMRHKmodel" << endl << endl;
    fout << "fix: " << fix << endl;
    fout << "npmax: " << npmax << endl << endl;

    fout << "GP: " << true << endl;
    fout << "kernel: " << kernel << endl;
    fout << "share_eta2: " << share_eta2 << endl;
    fout << "share_eta3: " << share_eta3 << endl;
    fout << "share_eta4: " << share_eta4 << endl;

    fout << "hyperpriors: " << false << endl;
    fout << "trend: " << trend << endl;
    fout << "degree: " << degree << endl;
    fout << "multi_instrument: " << data._multi << endl;
    fout << "known_object: " << known_object << endl;
    fout << "n_known_object: " << n_known_object << endl;
    fout << "transiting_planet: " << transiting_planet << endl;
    fout << "n_transiting_planet: " << n_transiting_planet << endl;
    fout << "magnetic_cycle_kernel: " << magnetic_cycle_kernel << endl;
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
    fout << "Cfwhm_prior: " << *Cfwhm_prior << endl;
    fout << "Crhk_prior: " << *Crhk_prior << endl;

    fout << "Jprior: " << *Jprior << endl;
    fout << "Jfwhm_prior: " << *Jfwhm_prior << endl;
    fout << "Jrhk_prior: " << *Jrhk_prior << endl;


    if (trend){
        if (degree >= 1) fout << "slope_prior: " << *slope_prior << endl;
        if (degree >= 2) fout << "quadr_prior: " << *quadr_prior << endl;
        if (degree == 3) fout << "cubic_prior: " << *cubic_prior << endl;
    }

    if (data._multi) {
        fout << "offsets_prior: " << *offsets_prior << endl;
        int i = 0;
        for (auto &p : individual_offset_prior)
        {
            fout << "individual_offset_prior[" << i << "]: " << *p << endl;
            i++;
        }

        fout << "offsets_fwhm_prior: " << *offsets_fwhm_prior << endl;
        i = 0;
        for (auto &p : individual_offset_fwhm_prior)
        {
            fout << "individual_offset_fwhm_prior[" << i << "]: " << *p << endl;
            i++;
        }

        fout << "offsets_rhk_prior: " << *offsets_rhk_prior << endl;
        i = 0;
        for (auto &p : individual_offset_rhk_prior)
        {
            fout << "individual_offset_rhk_prior[" << i << "]: " << *p << endl;
            i++;
        }
    }

    fout << endl << "[priors.GP]" << endl;
    switch (kernel)
    {
    case qp:
        fout << "eta1_prior: " << *eta1_prior << endl;
        fout << "eta2_prior: " << *eta2_prior << endl;
        fout << "eta3_prior: " << *eta3_prior << endl;
        fout << "eta4_prior: " << *eta4_prior << endl;
        fout << endl;
        fout << "eta1_fwhm_prior: " << *eta1_fwhm_prior << endl;
        fout << "eta2_fwhm_prior: " << *eta2_fwhm_prior << endl;
        fout << "eta3_fwhm_prior: " << *eta3_fwhm_prior << endl;
        fout << "eta4_fwhm_prior: " << *eta4_fwhm_prior << endl;
        fout << endl;
        fout << "eta1_rhk_prior: " << *eta1_rhk_prior << endl;
        fout << "eta2_rhk_prior: " << *eta2_rhk_prior << endl;
        fout << "eta3_rhk_prior: " << *eta3_rhk_prior << endl;
        fout << "eta4_rhk_prior: " << *eta4_rhk_prior << endl;
        break;
    default:
        break;
    }

    if (magnetic_cycle_kernel) {
        fout << "eta5_prior: " << *eta5_prior << endl;
        fout << "eta5_fwhm_prior: " << *eta5_fwhm_prior << endl;
        fout << "eta5_rhk_prior: " << *eta5_rhk_prior << endl;
        fout << "eta6_prior: " << *eta6_prior << endl;
        fout << "eta7_prior: " << *eta7_prior << endl;
    }

    fout << endl;

    if (planets.get_max_num_components()>0){
        auto conditional = planets.get_conditional_prior();

        if (false){
            fout << endl << "[prior.hyperpriors]" << endl;
            fout << "log_muP_prior: " << *conditional->log_muP_prior << endl;
            fout << "wP_prior: " << *conditional->wP_prior << endl;
            fout << "log_muK_prior: " << *conditional->log_muK_prior << endl;
        }

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

    fout << endl;
	fout.close();

    #if DEBUG
    std::cout << std::endl << "kima_model_setup.txt saved" << std::endl;
    #endif

}


using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;

auto RVFWHMRHKMODEL_DOC = R"D(
    Implements a joint model for RVs, FWHM, and R'HK with a GP component for activity signals.
    
    Args:
        fix (bool):
            whether the number of Keplerians should be fixed
        npmax (int):
            maximum number of Keplerians
        data (RVData):
            the RV data
    )D";

class RVFWHMRHKmodel_publicist : public RVFWHMRHKmodel
{
    public:
        using RVFWHMRHKmodel::fix;
        using RVFWHMRHKmodel::npmax;
        using RVFWHMRHKmodel::data;
        // 
        using RVFWHMRHKmodel::trend;
        using RVFWHMRHKmodel::degree;
        using RVFWHMRHKmodel::star_mass;
        using RVFWHMRHKmodel::enforce_stability;

        using RVFWHMRHKmodel::share_eta2;
        using RVFWHMRHKmodel::share_eta3;
        using RVFWHMRHKmodel::share_eta4;
        using RVFWHMRHKmodel::magnetic_cycle_kernel;
};

NB_MODULE(RVFWHMRHKmodel, m) {
    nb::class_<RVFWHMRHKmodel>(m, "RVFWHMRHKmodel")
        .def(nb::init<bool&, int&, RVData&>(), "fix"_a, "npmax"_a, "data"_a, RVFWHMRHKMODEL_DOC)
        //
        .def_rw("directory", &RVFWHMRHKmodel::directory,
                "directory where the model ran")
        // 
        .def_rw("fix", &RVFWHMRHKmodel_publicist::fix,
                "whether the number of Keplerians is fixed")
        .def_rw("npmax", &RVFWHMRHKmodel_publicist::npmax,
                "maximum number of Keplerians")
        .def_ro("data", &RVFWHMRHKmodel_publicist::data,
                "the data")

        //
        .def_rw("trend", &RVFWHMRHKmodel_publicist::trend,
                "whether the model includes a polynomial trend")
        .def_rw("degree", &RVFWHMRHKmodel_publicist::degree,
                "degree of the polynomial trend")

        // KO mode
        .def("set_known_object", &RVFWHMRHKmodel::set_known_object)
        .def_prop_ro("known_object", [](RVFWHMRHKmodel &m) { return m.get_known_object(); },
                     "whether the model includes (better) known extra Keplerian curve(s)")
        .def_prop_ro("n_known_object", [](RVFWHMRHKmodel &m) { return m.get_n_known_object(); },
                     "how many known objects")

        // transiting planets
        .def("set_transiting_planet", &RVFWHMRHKmodel::set_transiting_planet)
        .def_prop_ro("transiting_planet", [](RVFWHMRHKmodel &m) { return m.get_transiting_planet(); },
                     "whether the model includes transiting planet(s)")
        .def_prop_ro("n_transiting_planet", [](RVFWHMRHKmodel &m) { return m.get_n_transiting_planet(); },
                     "how many transiting planets")

        //
        .def_rw("star_mass", &RVFWHMRHKmodel_publicist::star_mass,
                "stellar mass [Msun]")
        .def_rw("enforce_stability", &RVFWHMRHKmodel_publicist::enforce_stability, 
                "whether to enforce AMD-stability")

        .def_rw("share_eta2", &RVFWHMRHKmodel_publicist::share_eta2,
                "whether the η2 parameter is shared between RVs and FWHM")
        .def_rw("share_eta3", &RVFWHMRHKmodel_publicist::share_eta3,
                "whether the η3 parameter is shared between RVs and FWHM")
        .def_rw("share_eta4", &RVFWHMRHKmodel_publicist::share_eta4,
                "whether the η4 parameter is shared between RVs and FWHM")

        .def_rw("magnetic_cycle_kernel", &RVFWHMRHKmodel_publicist::magnetic_cycle_kernel, 
                "whether to consider a (periodic) GP kernel for a magnetic cycle")

        // priors
        .def_prop_rw("Cprior",
            [](RVFWHMRHKmodel &m) { return m.Cprior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.Cprior = d; },
            "Prior for the systemic velocity")
        .def_prop_rw("Cfwhm_prior",
            [](RVFWHMRHKmodel &m) { return m.Cfwhm_prior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.Cfwhm_prior = d; },
            "Prior for the 'systemic' FWHM")
        .def_prop_rw("Crhk_prior",
            [](RVFWHMRHKmodel &m) { return m.Crhk_prior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.Crhk_prior = d; },
            "Prior for the 'systemic' R'hk ")

        .def_prop_rw("Jprior",
            [](RVFWHMRHKmodel &m) { return m.Jprior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.Jprior = d; },
            "Prior for the extra white noise (jitter)")
        .def_prop_rw("Jfwhm_prior",
            [](RVFWHMRHKmodel &m) { return m.Jfwhm_prior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.Jfwhm_prior = d; },
            "Prior for the extra white noise (jitter) in the FWHM")
        .def_prop_rw("Jrhk_prior",
            [](RVFWHMRHKmodel &m) { return m.Jrhk_prior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.Jrhk_prior = d; },
            "Prior for the extra white noise (jitter) in the R'hk")


        .def_prop_rw("slope_prior",
            [](RVFWHMRHKmodel &m) { return m.slope_prior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.slope_prior = d; },
            "Prior for the slope")
        .def_prop_rw("quadr_prior",
            [](RVFWHMRHKmodel &m) { return m.quadr_prior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.quadr_prior = d; },
            "Prior for the quadratic coefficient of the trend")
        .def_prop_rw("cubic_prior",
            [](RVFWHMRHKmodel &m) { return m.cubic_prior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.cubic_prior = d; },
            "Prior for the cubic coefficient of the trend")
        
        // priors for the GP hyperparameters
        .def_prop_rw("eta1_prior",
            [](RVFWHMRHKmodel &m) { return m.eta1_prior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.eta1_prior = d; },
            "Prior for the GP 'amplitude' on the RVs")
        .def_prop_rw("eta1_fwhm_prior",
            [](RVFWHMRHKmodel &m) { return m.eta1_fwhm_prior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.eta1_fwhm_prior = d; },
            "Prior for the GP 'amplitude' on the FWHM")
        .def_prop_rw("eta1_rhk_prior",
            [](RVFWHMRHKmodel &m) { return m.eta1_rhk_prior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.eta1_rhk_prior = d; },
            "Prior for the GP 'amplitude' on the R'HK")

        .def_prop_rw("eta2_prior",
            [](RVFWHMRHKmodel &m) { return m.eta2_prior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.eta2_prior = d; },
            "Prior for η2, the GP correlation timescale, on the RVs")
        .def_prop_rw("eta2_fwhm_prior",
            [](RVFWHMRHKmodel &m) { return m.eta2_fwhm_prior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.eta2_fwhm_prior = d; },
            "Prior for η2, the GP correlation timescale, on the FWHM")
        .def_prop_rw("eta2_rhk_prior",
            [](RVFWHMRHKmodel &m) { return m.eta2_rhk_prior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.eta2_rhk_prior = d; },
            "Prior for η2, the GP correlation timescale, on the R'HK")

        .def_prop_rw("eta3_prior",
            [](RVFWHMRHKmodel &m) { return m.eta3_prior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.eta3_prior = d; },
            "Prior for η3, the GP period, on the RVs")
        .def_prop_rw("eta3_fwhm_prior",
            [](RVFWHMRHKmodel &m) { return m.eta3_fwhm_prior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.eta3_fwhm_prior = d; },
            "Prior for η3, the GP period, on the FWHM")
        .def_prop_rw("eta3_rhk_prior",
            [](RVFWHMRHKmodel &m) { return m.eta3_rhk_prior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.eta3_rhk_prior = d; },
            "Prior for η3, the GP period, on the R'HK")

        .def_prop_rw("eta4_prior",
            [](RVFWHMRHKmodel &m) { return m.eta4_prior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.eta4_prior = d; },
            "Prior for η4, the recurrence timescale or (inverse) harmonic complexity, on the RVs")
        .def_prop_rw("eta4_fwhm_prior",
            [](RVFWHMRHKmodel &m) { return m.eta4_fwhm_prior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.eta4_fwhm_prior = d; },
            "Prior for η4, the recurrence timescale or (inverse) harmonic complexity, on the FWHM")
        .def_prop_rw("eta4_rhk_prior",
            [](RVFWHMRHKmodel &m) { return m.eta4_rhk_prior; },
            [](RVFWHMRHKmodel &m, distribution &d) { m.eta4_rhk_prior = d; },
            "Prior for η4, the recurrence timescale or (inverse) harmonic complexity, on the R'HK")

        // known object priors
        // ? should these setters check if known_object is true?
        .def_prop_rw("KO_Pprior",
                     [](RVFWHMRHKmodel &m) { return m.KO_Pprior; },
                     [](RVFWHMRHKmodel &m, std::vector<distribution>& vd) { m.KO_Pprior = vd; },
                     "Prior for KO orbital period")
        .def_prop_rw("KO_Kprior",
                     [](RVFWHMRHKmodel &m) { return m.KO_Kprior; },
                     [](RVFWHMRHKmodel &m, std::vector<distribution>& vd) { m.KO_Kprior = vd; },
                     "Prior for KO semi-amplitude")
        .def_prop_rw("KO_eprior",
                     [](RVFWHMRHKmodel &m) { return m.KO_eprior; },
                     [](RVFWHMRHKmodel &m, std::vector<distribution>& vd) { m.KO_eprior = vd; },
                     "Prior for KO eccentricity")
        .def_prop_rw("KO_wprior",
                     [](RVFWHMRHKmodel &m) { return m.KO_wprior; },
                     [](RVFWHMRHKmodel &m, std::vector<distribution>& vd) { m.KO_wprior = vd; },
                     "Prior for KO argument of periastron")
        .def_prop_rw("KO_phiprior",
                     [](RVFWHMRHKmodel &m) { return m.KO_phiprior; },
                     [](RVFWHMRHKmodel &m, std::vector<distribution>& vd) { m.KO_phiprior = vd; },
                     "Prior for KO mean anomaly(ies)")

        // transiting planet priors
        // ? should these setters check if transiting_planet is true?
        .def_prop_rw("TR_Pprior",
            [](RVFWHMRHKmodel &m) { return m.TR_Pprior; },
            [](RVFWHMRHKmodel &m, std::vector<distribution>& vd) { m.TR_Pprior = vd; },
            "Prior for TR orbital period")
        .def_prop_rw("TR_Kprior",
                    [](RVFWHMRHKmodel &m) { return m.TR_Kprior; },
                    [](RVFWHMRHKmodel &m, std::vector<distribution>& vd) { m.TR_Kprior = vd; },
                    "Prior for TR semi-amplitude")
        .def_prop_rw("TR_eprior",
                    [](RVFWHMRHKmodel &m) { return m.TR_eprior; },
                    [](RVFWHMRHKmodel &m, std::vector<distribution>& vd) { m.TR_eprior = vd; },
                    "Prior for TR eccentricity")
        .def_prop_rw("TR_wprior",
                    [](RVFWHMRHKmodel &m) { return m.TR_wprior; },
                    [](RVFWHMRHKmodel &m, std::vector<distribution>& vd) { m.TR_wprior = vd; },
                    "Prior for TR argument of periastron")
        .def_prop_rw("TR_Tcprior",
                    [](RVFWHMRHKmodel &m) { return m.TR_Tcprior; },
                    [](RVFWHMRHKmodel &m, std::vector<distribution>& vd) { m.TR_Tcprior = vd; },
                    "Prior for TR mean anomaly(ies)")

        // conditional object
        .def_prop_rw("conditional",
                     [](RVFWHMRHKmodel &m) { return m.get_conditional_prior(); },
                     [](RVFWHMRHKmodel &m, KeplerianConditionalPrior& c) { /* does nothing */ });
}