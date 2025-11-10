#include "RVFWHMmodel.h"

using namespace Eigen;
#define TIMING false

const double halflog2pi = 0.5*log(2.*M_PI);


void RVFWHMmodel::initialize_from_data(RVData& data)
{
    if (data.actind.size() < 2) // need at least one activity indicator (the FWHM)
    {
        std::string msg = "kima: RVFWHMmodel: no data for activity indicators (FWHM)";
        throw std::runtime_error(msg);
    }

    offsets.resize(2 * data.number_instruments - 2);
    jitters.resize(2 * data.number_instruments);
    individual_offset_prior.resize(data.number_instruments - 1);
    individual_offset_fwhm_prior.resize(data.number_instruments - 1);

    size_t N = data.N();

    // resize RV, FWHM model vectors
    mu.resize(N);
    mu_fwhm.resize(N);
    // resize covariance matrices
    C.resize(N, N);
    C_fwhm.resize(N, N);

    // copy uncertainties
    sig_copy = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(data.sig.data(), N);
    auto sig_fwhm = data.get_actind()[1]; // temporary
    sig_fwhm_copy = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(sig_fwhm.data(), N);

    // set default conditional priors that depend on data
    auto conditional = planets.get_conditional_prior();
    conditional->set_default_priors(data);
}

void RVFWHMmodel::set_known_object(size_t n)
{
    known_object = true;
    n_known_object = n;

    KO_Pprior.resize(n); KO_Kprior.resize(n); KO_eprior.resize(n); KO_phiprior.resize(n); KO_wprior.resize(n);
    KO_P.resize(n); KO_K.resize(n); KO_e.resize(n); KO_phi.resize(n); KO_w.resize(n);
}

void RVFWHMmodel::set_transiting_planet(size_t n)
{
    transiting_planet = true;
    n_transiting_planet = n;

    TR_Pprior.resize(n); TR_Kprior.resize(n); TR_eprior.resize(n); TR_Tcprior.resize(n); TR_wprior.resize(n);
    TR_P.resize(n); TR_K.resize(n); TR_e.resize(n); TR_Tc.resize(n); TR_w.resize(n);
}

/// set default priors if the user didn't change them
void RVFWHMmodel::setPriors()  // BUG: should be done by only one thread!
{
    auto defaults = DefaultPriors(data);

    // systemic velocity
    if (!Cprior)
        Cprior = defaults.get("Cprior");
    
    // "systemic FWHM"
    if (!Cfwhm_prior)
        Cfwhm_prior = defaults.get("Cfwhm_prior");

    // jitter for the RVs
    if (!Jprior)
        Jprior = defaults.get("Jprior");

    // jitter for the FWHM
    if (!Jfwhm_prior)
        Jfwhm_prior = defaults.get("Jfwhm_prior");

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

    if (trend_fwhm)
    {
        if (degree_fwhm == 0)
            throw std::logic_error("trend_fwhm=true but degree_fwhm=0");
        if (degree_fwhm > 3)
            throw std::range_error("can't go higher than 3rd degree trends");
        if (degree_fwhm >= 1 && !slope_fwhm_prior)
            slope_fwhm_prior = defaults.get("slope_fwhm_prior");
        if (degree_fwhm >= 2 && !quadr_fwhm_prior)
            quadr_fwhm_prior = defaults.get("quadr_fwhm_prior");
        if (degree_fwhm == 3 && !cubic_fwhm_prior)
            cubic_fwhm_prior = defaults.get("cubic_fwhm_prior");
    }


    // if offsets_prior is not (re)defined, assume a default
    if (data._multi)
    {
        if (!offsets_prior)
            offsets_prior = defaults.get("offsets_prior");
        if (!offsets_fwhm_prior)
            offsets_fwhm_prior = defaults.get("offsets_fwhm_prior");

        // if individual_offset_prior is not (re)defined, assume offsets_prior
        for (size_t j = 0; j < data.number_instruments - 1; j++)
        {
            if (!individual_offset_prior[j])
                individual_offset_prior[j] = offsets_prior;
            if (!individual_offset_fwhm_prior[j])
                individual_offset_fwhm_prior[j] = offsets_fwhm_prior;
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
        std::string msg = "kima: RVFWHMmodel: only the QP kernel is currently supported";
        throw std::runtime_error(msg);
    }

    switch (kernel)
    {
    case qp:
        // eta1 and eta1_fwhm are never shared, so they get default priors if
        // they haven't been set
        if (!eta1_prior)
        {
            eta1_prior = defaults.get("eta1_prior");
        }

        if (!eta1_fwhm_prior)
        {
            eta1_fwhm_prior = defaults.get("eta1_fwhm_prior");
        }

        // eta2 can be shared
        if (!eta2_prior)
        {
            eta2_prior = defaults.get("eta2_prior");
        }

        if (share_eta2)
        {
            eta2_fwhm_prior = eta2_prior;
        }
        else
        {
            if (!eta2_fwhm_prior)
                eta2_fwhm_prior = defaults.get("eta2_fwhm_prior");
        }

        // eta3 can be shared
        if (!eta3_prior)
        {
            eta3_prior = defaults.get("eta3_prior");
        }
        if (share_eta3)
        {
            eta3_fwhm_prior = eta3_prior;
        }
        else
        {
            if (!eta3_fwhm_prior)
                eta3_fwhm_prior = defaults.get("eta3_fwhm_prior");
        }

        // eta4 can be shared
        if (!eta4_prior)
        {
            eta4_prior = defaults.get("eta4_prior");
        }
        if (share_eta4)
        {
            eta4_fwhm_prior = eta4_prior;
        }
        else
        {
            if (!eta4_fwhm_prior)
                eta4_fwhm_prior = defaults.get("eta4_fwhm_prior");
        }
        break;
    
    default:
        break;
    }
}


void RVFWHMmodel::from_prior(RNG& rng)
{
    // preliminaries
    setPriors();
    save_setup();

    planets.from_prior(rng);
    planets.consolidate_diff();

    bkg = Cprior->generate(rng);
    bkg_fwhm = Cfwhm_prior->generate(rng);

    if(data._multi)
    {
        // draw instrument offsets for the RVs
        for (size_t i = 0; i < offsets.size() / 2; i++) {
            offsets[i] = individual_offset_prior[i]->generate(rng);
        }
        // draw instrument offsets for the FWHM
        size_t j = 0;
        for (size_t i = offsets.size() / 2; i < offsets.size(); i++) {
            offsets[i] = individual_offset_fwhm_prior[j]->generate(rng);
            j++;
        }

        for (size_t i = 0; i < jitters.size() / 2; i++) {
            jitters[i] = Jprior->generate(rng);
        }

        for (size_t i = jitters.size() / 2; i < jitters.size(); i++) {
            jitters[i] = Jfwhm_prior->generate(rng);
        }
    }
    else
    {
        jitter = Jprior->generate(rng);
        jitter_fwhm = Jfwhm_prior->generate(rng);
    }

    if(trend)
    {
        if (degree >= 1) slope = slope_prior->generate(rng);
        if (degree >= 2) quadr = quadr_prior->generate(rng);
        if (degree == 3) cubic = cubic_prior->generate(rng);
    }

    if (trend_fwhm)
    {
        if (degree_fwhm >= 1) slope_fwhm = slope_fwhm_prior->generate(rng);
        if (degree_fwhm >= 2) quadr_fwhm = quadr_fwhm_prior->generate(rng);
        if (degree_fwhm == 3) cubic_fwhm = cubic_fwhm_prior->generate(rng);
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

        eta3 = eta3_prior->generate(rng); // days
        if (!share_eta3)
            eta3_fw = eta3_fwhm_prior->generate(rng); // days

        eta2 = eta2_prior->generate(rng); // days
        if (!share_eta2)
            eta2_fw = eta2_fwhm_prior->generate(rng); // days

        eta4 = exp(eta4_prior->generate(rng));
        if (!share_eta4)
            eta4_fw = eta4_fwhm_prior->generate(rng);
        break;
    
    default:
        break;
    }

    calculate_mu();
    calculate_mu_fwhm();

    calculate_C();
    calculate_C_fwhm();
}

/// @brief Calculate the full RV model
void RVFWHMmodel::calculate_mu()
{
    size_t N = data.N();

    // Update or from scratch?
    bool update = (planets.get_added().size() < planets.get_components().size()) &&
            (staleness <= 10);
    update = false;

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
            for (size_t j = 0; j < offsets.size() / 2; j++)
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
        if(false) //hyperpriors
            P = exp(components[j][0]);
        else
            P = components[j][0];

        K = components[j][1];
        phi = components[j][2];
        ecc = components[j][3];
        omega = components[j][4];

        auto v = brandt::keplerian(data.t, P, K, ecc, omega, phi, data.M0_epoch);
        for (size_t i = 0; i < N; i++)
        {
            mu[i] += v[i];
        }
    }


    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Model eval took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif

}


/// @brief Calculate the full FWHM model
void RVFWHMmodel::calculate_mu_fwhm()
{
    size_t N = data.N();
    int Ni = data.Ninstruments();

    mu_fwhm.assign(mu_fwhm.size(), bkg_fwhm);

    if(trend_fwhm)
    {
        double tmid = data.get_t_middle();
        for (size_t i = 0; i < N; i++)
        {
            mu_fwhm[i] += slope_fwhm * (data.t[i] - tmid) +
                          quadr_fwhm * pow(data.t[i] - tmid, 2) +
                          cubic_fwhm * pow(data.t[i] - tmid, 3);
        }
    }

    if (data._multi) {
        auto obsi = data.get_obsi();
        for (size_t j = offsets.size() / 2; j < offsets.size(); j++) {
            for (size_t i = 0; i < N; i++) {
                if (obsi[i] == j + 2 - Ni) {
                    mu_fwhm[i] += offsets[j];
                }
            }
        }
    }
}


/// @brief Fill the GP covariance matrix
void RVFWHMmodel::calculate_C()
{

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



    // /* This implements the "standard" quasi-periodic kernel, see R&W2006 */
    // for (size_t i = 0; i < N; i++)
    // {
    //     for (size_t j = i; j < N; j++)
    //     {
    //         double r = data.t[i] - data.t[j];
    //         C(i, j) = eta1*eta1*exp(-0.5*pow(r/eta2, 2)
    //                     -2.0*pow(sin(M_PI*r/eta3)/eta4, 2) );

    //         if (i == j)
    //         {
    //             double sig = data.sig[i];
    //             if (data._multi)
    //             {
    //                 double jit = jitters[data.obsi[i] - 1];
    //                 C(i, j) += sig * sig + jit * jit;
    //             }
    //             else
    //             {
    //                 C(i, j) += sig * sig + jitter * jitter;
    //             }
    //         }
    //         else
    //         {
    //             C(j, i) = C(i, j);
    //         }
    //     }
    // }

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "GP build matrix: ";
    cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
    cout << " ns" << "\t"; // << std::endl;
    #endif
}


/// @brief Fill the GP covariance matrix
void RVFWHMmodel::calculate_C_fwhm()
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
            double jit = jitters[jitters.size() / 2 + data.obsi[i] - 1];
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
}


void RVFWHMmodel::remove_known_object()
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

void RVFWHMmodel::add_known_object()
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

void RVFWHMmodel::remove_transiting_planet()
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

void RVFWHMmodel::add_transiting_planet()
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

// TODO: compute stability for transiting planet(s)
int RVFWHMmodel::is_stable() const
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


double RVFWHMmodel::perturb(RNG& rng)
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
            }
            else if(rng.rand() <= 0.33330)
            {
                eta3_prior->perturb(eta3, rng);
                if (share_eta3)
                    eta3_fw = eta3;
                else
                    eta3_fwhm_prior->perturb(eta3_fw, rng);
            }
            else if(rng.rand() <= 0.5)
            {
                eta2_prior->perturb(eta2, rng);
                if (share_eta2)
                    eta2_fw = eta2;
                else
                    eta2_fwhm_prior->perturb(eta2_fw, rng);
            }
            else
            {
                eta4_prior->perturb(eta4, rng);
                if (share_eta4)
                    eta4_fw = eta4;
                else
                    eta4_fwhm_prior->perturb(eta4_fw, rng);
            }
            break;

        default:
            break;
        }

        calculate_C();
        calculate_C_fwhm();
    }

    else if(rng.rand() <= 0.5) // perturb jitter(s) + known_object
    {
        if(data._multi)
        {
            for (size_t i = 0; i < jitters.size() / 2; i++)
            {
                Jprior->perturb(jitters[i], rng);
            }
            for (size_t i = jitters.size() / 2; i < jitters.size(); i++)
            {
                Jfwhm_prior->perturb(jitters[i], rng);
            }
        }
        else
        {
            Jprior->perturb(jitter, rng);
            Jfwhm_prior->perturb(jitter_fwhm, rng);
        }

        calculate_C(); // recalculate covariance matrix
        calculate_C_fwhm(); // recalculate covariance matrix

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
        // NOTE: this only modifies mu (and not mu_fwhm), because
        // calculate_mu_fwhm() is called at the end

        // for (size_t i = 0; i < mu.size(); i++)
        // {
        //     mu[i] -= bkg;

        //     if(trend) {
        //         mu[i] -= slope * (data.t[i] - tmid) +
        //                  quadr * pow(data.t[i] - tmid, 2) +
        //                  cubic * pow(data.t[i] - tmid, 3);
        //     }

        //     if (data._multi)
        //     {
        //         for (size_t j = 0; j < offsets.size() / 2; j++)
        //         {
        //             if (data.obsi[i] == j+1) { mu[i] -= offsets[j]; }
        //         }
        //     }
        // }

        // propose new vsys
        Cprior->perturb(bkg, rng);
        Cfwhm_prior->perturb(bkg_fwhm, rng);

        // propose new instrument offsets
        if (data._multi)
        {
            for (size_t j = 0; j < offsets.size() / 2; j++)
            {
                individual_offset_prior[j]->perturb(offsets[j], rng);
            }
            size_t k = 0;
            for (size_t j = offsets.size() / 2; j < offsets.size(); j++)
            {
                individual_offset_fwhm_prior[k]->perturb(offsets[j], rng);
                k++;
            }
        }

        // propose new slope
        if (trend)
        {
            if (degree >= 1) slope_prior->perturb(slope, rng);
            if (degree >= 2) quadr_prior->perturb(quadr, rng);
            if (degree == 3) cubic_prior->perturb(cubic, rng);
        }

        if (trend_fwhm)
        {
            if (degree_fwhm >= 1) slope_fwhm_prior->perturb(slope_fwhm, rng);
            if (degree_fwhm >= 2) quadr_fwhm_prior->perturb(quadr_fwhm, rng);
            if (degree_fwhm == 3) cubic_fwhm_prior->perturb(cubic_fwhm, rng);
        }

        // for (size_t i = 0; i < mu.size(); i++)
        // {
        //     mu[i] += bkg;

        //     if(trend) {
        //         mu[i] += slope * (data.t[i] - tmid) +
        //                  quadr * pow(data.t[i] - tmid, 2) +
        //                  cubic * pow(data.t[i] - tmid, 3);
        //     }

        //     if (data._multi)
        //     {
        //         for (size_t j = 0; j < offsets.size(); j++)
        //         {
        //             if (data.obsi[i] == j+1) { mu[i] += offsets[j]; }
        //         }
        //     }
        // }

        calculate_mu();
        calculate_mu_fwhm();

    }


    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Perturb took ";
    cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
    cout << " μs" << std::endl;
    #endif

    return logH;
}


double RVFWHMmodel::log_likelihood() const
{
    size_t N = data.N();
    const auto& y = data.y;
    const auto& fwhm = data.actind[0];

    double logL = 0.;

    if (enforce_stability){
        int stable = is_stable();
        if (stable != 0)
            return -std::numeric_limits<double>::infinity();
    }


    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    double logL_RV, logL_FWHM;

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

    logL = logL_RV + logL_FWHM;


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


void RVFWHMmodel::print(std::ostream& out) const
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
    }

    if (trend)
    {
        out.precision(15);
        if (degree >= 1) out << slope << '\t';
        if (degree >= 2) out << quadr << '\t';
        if (degree == 3) out << cubic << '\t';
        out.precision(8);
    }

    if (trend_fwhm)
    {
        out.precision(15);
        if (degree_fwhm >= 1) out << slope_fwhm << '\t';
        if (degree_fwhm >= 2) out << quadr_fwhm << '\t';
        if (degree_fwhm == 3) out << cubic_fwhm << '\t';
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
        out << eta1 << '\t' << eta1_fw << '\t';

        out << eta2 << '\t';
        if (!share_eta2) out << eta2_fw << '\t';

        out << eta3 << '\t';
        if (!share_eta3) out << eta3_fw << '\t';
        
        out << eta4 << '\t';
        if (!share_eta4) out << eta4_fw << '\t';
        
        break;
    
    default:
        break;
    }

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

    out << bkg_fwhm << '\t';
    out << bkg;
}


string RVFWHMmodel::description() const
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
    }

    if(trend)
    {
        if (degree >= 1) desc += "slope" + sep;
        if (degree >= 2) desc += "quadr" + sep;
        if (degree == 3) desc += "cubic" + sep;
    }

    if (trend_fwhm)
    {
        if (degree_fwhm >= 1) desc += "slope_fwhm" + sep;
        if (degree_fwhm >= 2) desc += "quadr_fwhm" + sep;
        if (degree_fwhm == 3) desc += "cubic_fwhm" + sep;
    }

    if (data._multi){
        for(unsigned j=0; j<offsets.size(); j++)
            desc += "offset" + std::to_string(j+1) + sep;
    }

    // GP parameters
    switch (kernel)
    {
    case qp:
        desc += "eta1" + sep + "eta1_fwhm" + sep;

        desc += "eta2" + sep;
        if (!share_eta2) desc += "eta2_fwhm" + sep;

        desc += "eta3" + sep;
        if (!share_eta3) desc += "eta3_fwhm" + sep;

        desc += "eta4" + sep;
        if (!share_eta4) desc += "eta4_fwhm" + sep;

        break;

    default:
        break;
    }


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

    desc += "cfwhm" + sep;
    desc += "vsys";

    return desc;
}


void RVFWHMmodel::save_setup() {
	std::fstream fout("kima_model_setup.txt", std::ios::out);
    fout << std::boolalpha;

    fout << "; " << timestamp() << endl << endl;

    fout << "[kima]" << endl;

    fout << "model: " << "RVFWHMmodel" << endl << endl;
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
    fout << "trend_fwhm: " << trend_fwhm << endl;
    fout << "degree_fwhm: " << degree_fwhm << endl;
    fout << "multi_instrument: " << data._multi << endl;
    fout << "known_object: " << known_object << endl;
    fout << "n_known_object: " << n_known_object << endl;
    fout << "transiting_planet: " << transiting_planet << endl;
    fout << "n_transiting_planet: " << n_transiting_planet << endl;
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
    fout << "Jprior: " << *Jprior << endl;
    fout << "Jfwhm_prior: " << *Jfwhm_prior << endl;

    if (trend)
    {
        if (degree >= 1) fout << "slope_prior: " << *slope_prior << endl;
        if (degree >= 2) fout << "quadr_prior: " << *quadr_prior << endl;
        if (degree == 3) fout << "cubic_prior: " << *cubic_prior << endl;
    }

    if (trend_fwhm)
    {
        if (degree_fwhm >= 1) fout << "slope_fwhm_prior: " << *slope_fwhm_prior << endl;
        if (degree_fwhm >= 2) fout << "quadr_fwhm_prior: " << *quadr_fwhm_prior << endl;
        if (degree_fwhm == 3) fout << "cubic_fwhm_prior: " << *cubic_fwhm_prior << endl;
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
        break;
    default:
        break;
    }


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

    fout << endl;
	fout.close();
}


using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;

auto RVFWHMMODEL_DOC = R"D(
Implements a joint model for RVs and FWHM with a GP component for activity signals.

Args:
    fix (bool):
        whether the number of Keplerians should be fixed
    npmax (int):
        maximum number of Keplerians
    data (RVData):
        the RV data
)D";

class RVFWHMmodel_publicist : public RVFWHMmodel
{
    public:
        using RVFWHMmodel::fix;
        using RVFWHMmodel::npmax;
        using RVFWHMmodel::data;
        //
        using RVFWHMmodel::trend, RVFWHMmodel::degree;
        using RVFWHMmodel::trend_fwhm, RVFWHMmodel::degree_fwhm;
        using RVFWHMmodel::star_mass;
        using RVFWHMmodel::enforce_stability;

        using RVFWHMmodel::share_eta2;
        using RVFWHMmodel::share_eta3;
        using RVFWHMmodel::share_eta4;
};

NB_MODULE(RVFWHMmodel, m) {
    nb::class_<RVFWHMmodel>(m, "RVFWHMmodel")
        .def(nb::init<bool&, int&, RVData&>(), "fix"_a, "npmax"_a, "data"_a, RVFWHMMODEL_DOC)
        //
        .def_rw("directory", &RVFWHMmodel::directory,
                "directory where the model ran")
        // 
        .def_rw("fix", &RVFWHMmodel_publicist::fix,
                "whether the number of Keplerians is fixed")
        .def_rw("npmax", &RVFWHMmodel_publicist::npmax,
                "maximum number of Keplerians")
        .def_ro("data", &RVFWHMmodel_publicist::data,
                "the data")

        //
        .def_rw("trend", &RVFWHMmodel_publicist::trend, "whether the model includes a polynomial trend (in the RVs)")
        .def_rw("degree", &RVFWHMmodel_publicist::degree, "degree of the polynomial trend (in the RVs)")

        .def_rw("trend_fwhm", &RVFWHMmodel_publicist::trend_fwhm, "whether the model includes a polynomial trend (in the FWHM)")
        .def_rw("degree_fwhm", &RVFWHMmodel_publicist::degree_fwhm, "degree of the polynomial trend (in the FWHM)")


        // KO mode
        .def("set_known_object", &RVFWHMmodel::set_known_object)
        .def_prop_ro("known_object", [](RVFWHMmodel &m) { return m.get_known_object(); },
                     "whether the model includes (better) known extra Keplerian curve(s)")
        .def_prop_ro("n_known_object", [](RVFWHMmodel &m) { return m.get_n_known_object(); },
                     "how many known objects")

        // transiting planets
        .def("set_transiting_planet", &RVFWHMmodel::set_transiting_planet)
        .def_prop_ro("transiting_planet", [](RVFWHMmodel &m) { return m.get_transiting_planet(); },
                     "whether the model includes transiting planet(s)")
        .def_prop_ro("n_transiting_planet", [](RVFWHMmodel &m) { return m.get_n_transiting_planet(); },
                     "how many transiting planets")

        //
        .def_rw("star_mass", &RVFWHMmodel_publicist::star_mass,
                "stellar mass [Msun]")
        .def_rw("enforce_stability", &RVFWHMmodel_publicist::enforce_stability, 
                "whether to enforce AMD-stability")

        .def_rw("share_eta2", &RVFWHMmodel_publicist::share_eta2,
                "whether the η2 parameter is shared between RVs and FWHM")
        .def_rw("share_eta3", &RVFWHMmodel_publicist::share_eta3,
                "whether the η3 parameter is shared between RVs and FWHM")
        .def_rw("share_eta4", &RVFWHMmodel_publicist::share_eta4,
                "whether the η4 parameter is shared between RVs and FWHM")

        // priors
        .def_prop_rw("Cprior",
            [](RVFWHMmodel &m) { return m.Cprior; },
            [](RVFWHMmodel &m, distribution &d) { m.Cprior = d; },
            "Prior for the systemic velocity")
        .def_prop_rw("Cfwhm_prior",
            [](RVFWHMmodel &m) { return m.Cfwhm_prior; },
            [](RVFWHMmodel &m, distribution &d) { m.Cfwhm_prior = d; },
            "Prior for the 'systemic' FWHM")

        .def_prop_rw("Jprior",
            [](RVFWHMmodel &m) { return m.Jprior; },
            [](RVFWHMmodel &m, distribution &d) { m.Jprior = d; },
            "Prior for the extra white noise (jitter)")
        .def_prop_rw("Jfwhm_prior",
            [](RVFWHMmodel &m) { return m.Jfwhm_prior; },
            [](RVFWHMmodel &m, distribution &d) { m.Jfwhm_prior = d; },
            "Prior for the extra white noise (jitter) in the FWHM")
    
        .def_prop_rw("slope_prior",
            [](RVFWHMmodel &m) { return m.slope_prior; },
            [](RVFWHMmodel &m, distribution &d) { m.slope_prior = d; },
            "Prior for the slope (in the RVs)")
        .def_prop_rw("quadr_prior",
            [](RVFWHMmodel &m) { return m.quadr_prior; },
            [](RVFWHMmodel &m, distribution &d) { m.quadr_prior = d; },
            "Prior for the quadratic coefficient of the trend (in the RVs)")
        .def_prop_rw("cubic_prior",
            [](RVFWHMmodel &m) { return m.cubic_prior; },
            [](RVFWHMmodel &m, distribution &d) { m.cubic_prior = d; },
            "Prior for the cubic coefficient of the trend (in the RVs)")
        
        .def_prop_rw("slope_fwhm_prior",
            [](RVFWHMmodel &m) { return m.slope_fwhm_prior; },
            [](RVFWHMmodel &m, distribution &d) { m.slope_fwhm_prior = d; },
            "Prior for the slope in the FWHM (in the FWHM)")
        .def_prop_rw("quadr_fwhm_prior",
            [](RVFWHMmodel &m) { return m.quadr_fwhm_prior; },
            [](RVFWHMmodel &m, distribution &d) { m.quadr_fwhm_prior = d; },
            "Prior for the quadratic coefficient of the trend in the FWHM (in the FWHM)")
        .def_prop_rw("cubic_fwhm_prior",
            [](RVFWHMmodel &m) { return m.cubic_fwhm_prior; },
            [](RVFWHMmodel &m, distribution &d) { m.cubic_fwhm_prior = d; },
            "Prior for the cubic coefficient of the trend in the FWHM (in the FWHM)")

        
        // priors for the GP hyperparameters
        .def_prop_rw("eta1_prior",
            [](RVFWHMmodel &m) { return m.eta1_prior; },
            [](RVFWHMmodel &m, distribution &d) { m.eta1_prior = d; },
            "Prior for the GP 'amplitude' on the RVs")
        .def_prop_rw("eta1_fwhm_prior",
            [](RVFWHMmodel &m) { return m.eta1_fwhm_prior; },
            [](RVFWHMmodel &m, distribution &d) { m.eta1_fwhm_prior = d; },
            "Prior for the GP 'amplitude' on the FWHM")

        .def_prop_rw("eta2_prior",
            [](RVFWHMmodel &m) { return m.eta2_prior; },
            [](RVFWHMmodel &m, distribution &d) { m.eta2_prior = d; },
            "Prior for η2, the GP correlation timescale, on the RVs")
        .def_prop_rw("eta2_fwhm_prior",
            [](RVFWHMmodel &m) { return m.eta2_fwhm_prior; },
            [](RVFWHMmodel &m, distribution &d) { m.eta2_fwhm_prior = d; },
            "Prior for η2, the GP correlation timescale, on the FWHM")

        .def_prop_rw("eta3_prior",
            [](RVFWHMmodel &m) { return m.eta3_prior; },
            [](RVFWHMmodel &m, distribution &d) { m.eta3_prior = d; },
            "Prior for η3, the GP period, on the RVs")
        .def_prop_rw("eta3_fwhm_prior",
            [](RVFWHMmodel &m) { return m.eta3_fwhm_prior; },
            [](RVFWHMmodel &m, distribution &d) { m.eta3_fwhm_prior = d; },
            "Prior for η3, the GP period, on the FWHM")

        .def_prop_rw("eta4_prior",
            [](RVFWHMmodel &m) { return m.eta4_prior; },
            [](RVFWHMmodel &m, distribution &d) { m.eta4_prior = d; },
            "Prior for η4, the recurrence timescale or (inverse) harmonic complexity, on the RVs")
        .def_prop_rw("eta4_fwhm_prior",
            [](RVFWHMmodel &m) { return m.eta4_fwhm_prior; },
            [](RVFWHMmodel &m, distribution &d) { m.eta4_fwhm_prior = d; },
            "Prior for η4, the recurrence timescale or (inverse) harmonic complexity, on the FWHM")

        // known object priors
        // ? should these setters check if known_object is true?
        .def_prop_rw("KO_Pprior",
                     [](RVFWHMmodel &m) { return m.KO_Pprior; },
                     [](RVFWHMmodel &m, std::vector<distribution>& vd) { m.KO_Pprior = vd; },
                     "Prior for KO orbital period")
        .def_prop_rw("KO_Kprior",
                     [](RVFWHMmodel &m) { return m.KO_Kprior; },
                     [](RVFWHMmodel &m, std::vector<distribution>& vd) { m.KO_Kprior = vd; },
                     "Prior for KO semi-amplitude")
        .def_prop_rw("KO_eprior",
                     [](RVFWHMmodel &m) { return m.KO_eprior; },
                     [](RVFWHMmodel &m, std::vector<distribution>& vd) { m.KO_eprior = vd; },
                     "Prior for KO eccentricity")
        .def_prop_rw("KO_wprior",
                     [](RVFWHMmodel &m) { return m.KO_wprior; },
                     [](RVFWHMmodel &m, std::vector<distribution>& vd) { m.KO_wprior = vd; },
                     "Prior for KO argument of periastron")
        .def_prop_rw("KO_phiprior",
                     [](RVFWHMmodel &m) { return m.KO_phiprior; },
                     [](RVFWHMmodel &m, std::vector<distribution>& vd) { m.KO_phiprior = vd; },
                     "Prior for KO mean anomaly(ies)")

        // transiting planet priors
        // ? should these setters check if transiting_planet is true?
        .def_prop_rw("TR_Pprior",
                     [](RVFWHMmodel &m) { return m.TR_Pprior; },
                     [](RVFWHMmodel &m, std::vector<distribution>& vd) { m.TR_Pprior = vd; },
                     "Prior for TR orbital period")
        .def_prop_rw("TR_Kprior",
                     [](RVFWHMmodel &m) { return m.TR_Kprior; },
                     [](RVFWHMmodel &m, std::vector<distribution>& vd) { m.TR_Kprior = vd; },
                     "Prior for TR semi-amplitude")
        .def_prop_rw("TR_eprior",
                     [](RVFWHMmodel &m) { return m.TR_eprior; },
                     [](RVFWHMmodel &m, std::vector<distribution>& vd) { m.TR_eprior = vd; },
                     "Prior for TR eccentricity")
        .def_prop_rw("TR_wprior",
                     [](RVFWHMmodel &m) { return m.TR_wprior; },
                     [](RVFWHMmodel &m, std::vector<distribution>& vd) { m.TR_wprior = vd; },
                     "Prior for TR argument of periastron")
        .def_prop_rw("TR_Tcprior",
                     [](RVFWHMmodel &m) { return m.TR_Tcprior; },
                     [](RVFWHMmodel &m, std::vector<distribution>& vd) { m.TR_Tcprior = vd; },
                     "Prior for TR mean anomaly(ies)")


        // conditional object
        .def_prop_rw("conditional",
                     [](RVFWHMmodel &m) { return m.get_conditional_prior(); },
                     [](RVFWHMmodel &m, KeplerianConditionalPrior& c) { /* does nothing */ });
}