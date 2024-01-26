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

    // resize RV, FWHM model vectors
    mu.resize(data.N());
    mu_fwhm.resize(data.N());
    // resize covariance matrices
    C.resize(data.N(), data.N());
    C_fwhm.resize(data.N(), data.N());

    // set default conditional priors that depend on data
    auto conditional = planets.get_conditional_prior();
    conditional->set_default_priors(data);
}

void RVFWHMmodel::set_known_object(size_t n)
{
    known_object = true;
    n_known_object = n;

    KO_Pprior.resize(n);
    KO_Kprior.resize(n);
    KO_eprior.resize(n);
    KO_phiprior.resize(n);
    KO_wprior.resize(n);
}

/// set default priors if the user didn't change them
void RVFWHMmodel::setPriors()  // BUG: should be done by only one thread!
{
    // systemic velocity
    if (!Cprior)
        Cprior = make_prior<Uniform>(data.get_RV_min(), data.get_RV_max());
    
    // "systemic FWHM"
    if (!C2prior)
    {
        auto minFWHM = *min_element(data.actind[0].begin(), data.actind[0].end());
        auto maxFWHM = *max_element(data.actind[0].begin(), data.actind[0].end());
        C2prior = make_prior<Uniform>(minFWHM, maxFWHM);
    }

    // jitter for the RVs
    if (!Jprior)
        Jprior = make_prior<ModifiedLogUniform>(min(1.0, 0.1 * data.get_max_RV_span()), data.get_max_RV_span());

    // jitter for the FWHM
    if (!J2prior)
    {
        auto minFWHM = *min_element(data.actind[0].begin(), data.actind[0].end());
        auto maxFWHM = *max_element(data.actind[0].begin(), data.actind[0].end());
        auto spanFWHM = maxFWHM - minFWHM;
        J2prior = make_prior<ModifiedLogUniform>(min(1.0, 0.1 * spanFWHM), spanFWHM);
    }

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
            offsets_prior = make_prior<Uniform>( -data.get_RV_span(), data.get_RV_span() );
        if (!offsets_fwhm_prior) {
            auto minFWHM = *min_element(data.actind[0].begin(), data.actind[0].end());
            auto maxFWHM = *max_element(data.actind[0].begin(), data.actind[0].end());
            auto spanFWHM = maxFWHM - minFWHM;
            offsets_fwhm_prior = make_prior<Uniform>( -spanFWHM, spanFWHM );
        }

        for (size_t j = 0; j < data.number_instruments - 1; j++)
        {
            // if individual_offset_prior is not (re)defined, assume offsets_prior
            if (!individual_offset_prior[j])
                individual_offset_prior[j] = offsets_prior;
            if (!individual_offset_fwhm_prior[j])
                individual_offset_fwhm_prior[j] = offsets_fwhm_prior;
        }
    }

    if (known_object) { // KO mode!
        // if (n_known_object == 0) cout << "Warning: `known_object` is true, but `n_known_object` is set to 0";
        for (int i = 0; i < n_known_object; i++){
            if (!KO_Pprior[i] || !KO_Kprior[i] || !KO_eprior[i] || !KO_phiprior[i] || !KO_wprior[i])
                throw std::logic_error("When known_object=true, please set priors for each (KO_Pprior, KO_Kprior, KO_eprior, KO_phiprior, KO_wprior)");
        }
    }

    /* GP parameters */
    if (!eta1_prior)
        eta1_prior = make_prior<LogUniform>(0.1, 100);
    if (!eta1_fwhm_prior)
        eta1_fwhm_prior = make_prior<LogUniform>(0.1, 100);

    if (!eta2_prior)
        eta2_prior = make_prior<LogUniform>(1, 100);
    if (!eta2_fwhm_prior)
        eta2_fwhm_prior = make_prior<LogUniform>(1, 100);

    if (!eta3_prior)
        eta3_prior = make_prior<Uniform>(10, 40);
    if (!eta3_fwhm_prior)
        eta3_fwhm_prior = make_prior<Uniform>(10, 40);

    if (!eta4_prior)
        eta4_prior = make_prior<Uniform>(0.2, 5);
    if (!eta4_fwhm_prior)
        eta4_fwhm_prior = make_prior<Uniform>(0.2, 5);
}


void RVFWHMmodel::from_prior(RNG& rng)
{
    // preliminaries
    setPriors();
    save_setup();

    planets.from_prior(rng);
    planets.consolidate_diff();

    bkg = Cprior->generate(rng);
    bkg_fwhm = C2prior->generate(rng);

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
            jitters[i] = J2prior->generate(rng);
        }
    }
    else
    {
        jitter = Jprior->generate(rng);
        jitter_fwhm = J2prior->generate(rng);
    }

    if(trend)
    {
        if (degree >= 1) slope = slope_prior->generate(rng);
        if (degree >= 2) quadr = quadr_prior->generate(rng);
        if (degree == 3) cubic = cubic_prior->generate(rng);
    }


    if (known_object) { // KO mode!
        KO_P.resize(n_known_object);
        KO_K.resize(n_known_object);
        KO_e.resize(n_known_object);
        KO_phi.resize(n_known_object);
        KO_w.resize(n_known_object);

        for (int i=0; i<n_known_object; i++){
            KO_P[i] = KO_Pprior[i]->generate(rng);
            KO_K[i] = KO_Kprior[i]->generate(rng);
            KO_e[i] = KO_eprior[i]->generate(rng);
            KO_phi[i] = KO_phiprior[i]->generate(rng);
            KO_w[i] = KO_wprior[i]->generate(rng);
        }
    }

    // GP

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
    }
    else // just updating (adding) planets
        staleness++;


    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif


    double f, v, ti;
    double P, K, phi, ecc, omega, Tp;
    for(size_t j=0; j<components.size(); j++)
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
        for(size_t i=0; i<N; i++)
            mu[i] += v[i];
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
    size_t N = data.N();

    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    /* This implements the "standard" quasi-periodic kernel, see R&W2006 */
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = i; j < N; j++)
        {
            double r = data.t[i] - data.t[j];
            C(i, j) = eta1*eta1*exp(-0.5*pow(r/eta2, 2)
                        -2.0*pow(sin(M_PI*r/eta3)/eta4, 2) );

            if (i == j)
            {
                double sig = data.sig[i];
                if (data._multi)
                {
                    double jit = jitters[data.obsi[i] - 1];
                    C(i, j) += sig * sig + jit * jit;
                }
                else
                {
                    C(i, j) += sig * sig + jitter * jitter;
                }
            }
            else
            {
                C(j, i) = C(i, j);
            }
        }
    }

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
    size_t N = data.N();
    auto t = data.get_t();
    auto sig = data.get_actind()[1];

    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    /* This implements the "standard" quasi-periodic kernel, see R&W2006 */
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = i; j < N; j++)
        {
            double r = data.t[i] - data.t[j];
            C_fwhm(i, j) = eta1_fw * eta1_fw * exp(-0.5 * pow(r / eta2_fw, 2) - 2.0 * pow(sin(M_PI * r / eta3_fw) / eta4_fw, 2));

            if (i == j)
            {
                if (data._multi)
                {
                    double jit = jitters[data.obsi[i] - 1];
                    C_fwhm(i, j) += sig[i] * sig[i] + jit * jit;
                }
                else
                {
                    C_fwhm(i, j) += sig[i] * sig[i] + jitter * jitter;
                }
            }
            else
            {
                C_fwhm(j, i) = C_fwhm(i, j);
            }
        }
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
    double f, v, ti, Tp;
    for (int j = 0; j < n_known_object; j++) {
        auto v = brandt::keplerian(data.t, KO_P[j], KO_K[j], KO_e[j], KO_w[j], KO_phi[j], data.M0_epoch);
        for (size_t i = 0; i < data.N(); i++) {
            mu[i] -= v[i];
        }
    }
}

void RVFWHMmodel::add_known_object()
{
    for (int j = 0; j < n_known_object; j++) {
        auto v = brandt::keplerian(data.t, KO_P[j], KO_K[j], KO_e[j], KO_w[j], KO_phi[j], data.M0_epoch);
        for (size_t i = 0; i < data.N(); i++) {
            mu[i] += v[i];
        }
    }
}


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


    if(rng.rand() <= 0.5) // perturb planet parameters
    {
        logH += planets.perturb(rng);
        planets.consolidate_diff();
        calculate_mu();
    }
    else if(rng.rand() <= 0.5) // perturb GP parameters
    {
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


        calculate_C();
        calculate_C_fwhm();
    }

    else if(rng.rand() <= 0.5) // perturb jitter(s) + known_object
    {
        if(data._multi)
        {
            for (int i = 0; i < jitters.size() / 2; i++)
            {
                Jprior->perturb(jitters[i], rng);
            }
            for (int i = jitters.size() / 2; i < jitters.size(); i++)
            {
                J2prior->perturb(jitters[i], rng);
            }
        }
        else
        {
            Jprior->perturb(jitter, rng);
            J2prior->perturb(jitter_fwhm, rng);
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
                for (size_t j = 0; j < offsets.size() / 2; j++)
                {
                    if (data.obsi[i] == j+1) { mu[i] -= offsets[j]; }
                }
            }
        }

        // propose new vsys
        Cprior->perturb(bkg, rng);
        C2prior->perturb(bkg_fwhm, rng);

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
    out << eta1 << '\t' << eta1_fw << '\t';

    out << eta2 << '\t';
    if (!share_eta2) out << eta2_fw << '\t';

    out << eta3 << '\t';
    if (!share_eta3) out << eta3_fw << '\t';
    
    out << eta4 << '\t';
    if (!share_eta4) out << eta4_fw << '\t';

    // write KO parameters
    if(known_object){ // KO mode!
        for (auto P: KO_P) out << P << "\t";
        for (auto K: KO_K) out << K << "\t";
        for (auto phi: KO_phi) out << phi << "\t";
        for (auto e: KO_e) out << e << "\t";
        for (auto w: KO_w) out << w << "\t";
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


    if (data._multi){
        for(unsigned j=0; j<offsets.size(); j++)
            desc += "offset" + std::to_string(j+1) + sep;
    }

    // GP parameters
    desc += "eta1" + sep + "eta1_fw" + sep;

    desc += "eta2" + sep;
    if (!share_eta2) desc += "eta2_fw" + sep;

    desc += "eta3" + sep;
    if (!share_eta3) desc += "eta3_fw" + sep;

    desc += "eta4" + sep;
    if (!share_eta4) desc += "eta4_fw" + sep;


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

    time_t rawtime;
    time (&rawtime);
    fout << ";" << ctime(&rawtime) << endl;

    fout << "[kima]" << endl;

    fout << "model: " << "RVFWHMmodel" << endl << endl;
    fout << "fix: " << fix << endl;
    fout << "npmax: " << npmax << endl << endl;

    fout << "GP: " << true << endl;
    fout << "kernel: " << "standard" << endl;
    fout << "share_eta2: " << share_eta2 << endl;
    fout << "share_eta3: " << share_eta3 << endl;
    fout << "share_eta4: " << share_eta4 << endl;

    fout << "hyperpriors: " << false << endl;
    fout << "trend: " << trend << endl;
    fout << "degree: " << degree << endl;
    fout << "multi_instrument: " << data._multi << endl;
    fout << "known_object: " << known_object << endl;
    fout << "n_known_object: " << n_known_object << endl;
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

    fout << endl;
	fout.close();
}


using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;

class RVFWHMmodel_publicist : public RVFWHMmodel
{
    public:
        using RVFWHMmodel::trend;
        using RVFWHMmodel::degree;
        using RVFWHMmodel::star_mass;
        using RVFWHMmodel::enforce_stability;

        using RVFWHMmodel::share_eta2;
        using RVFWHMmodel::share_eta3;
        using RVFWHMmodel::share_eta4;
};

NB_MODULE(RVFWHMmodel, m) {
    nb::class_<RVFWHMmodel>(m, "RVFWHMmodel")
        .def(nb::init<bool&, int&, RVData&>(), "fix"_a, "npmax"_a, "data"_a)
        //
        .def_rw("trend", &RVFWHMmodel_publicist::trend,
                "whether the model includes a polynomial trend")
        .def_rw("degree", &RVFWHMmodel_publicist::degree,
                "degree of the polynomial trend")

        // KO mode
        .def("set_known_object", &RVFWHMmodel::set_known_object)
        .def_prop_ro("known_object", [](RVFWHMmodel &m) { return m.get_known_object(); },
                     "whether the model includes (better) known extra Keplerian curve(s)")
        .def_prop_ro("n_known_object", [](RVFWHMmodel &m) { return m.get_n_known_object(); },
                     "how many known objects")

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
        .def_prop_rw("Jprior",
            [](RVFWHMmodel &m) { return m.Jprior; },
            [](RVFWHMmodel &m, distribution &d) { m.Jprior = d; },
            "Prior for the extra white noise (jitter)")
        .def_prop_rw("slope_prior",
            [](RVFWHMmodel &m) { return m.slope_prior; },
            [](RVFWHMmodel &m, distribution &d) { m.slope_prior = d; },
            "Prior for the slope")
        .def_prop_rw("quadr_prior",
            [](RVFWHMmodel &m) { return m.quadr_prior; },
            [](RVFWHMmodel &m, distribution &d) { m.quadr_prior = d; },
            "Prior for the quadratic coefficient of the trend")
        .def_prop_rw("cubic_prior",
            [](RVFWHMmodel &m) { return m.cubic_prior; },
            [](RVFWHMmodel &m, distribution &d) { m.cubic_prior = d; },
            "Prior for the cubic coefficient of the trend")
        
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

        // conditional object
        .def_prop_rw("conditional",
                     [](RVFWHMmodel &m) { return m.get_conditional_prior(); },
                     [](RVFWHMmodel &m, RVConditionalPrior& c) { /* does nothing */ });
}