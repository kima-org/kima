#include "SPLEAFmodel.h"

#define TIMING false

const double halflog2pi = 0.5*log(2.*M_PI);


void SPLEAFmodel::initialize_from_data(RVData& data)
{
    if (data.number_instruments > 1)
        throw std::exception("SPLEAFmodel currently only supports one instrument");

    offsets.resize(data.number_instruments - 1);
    jitters.resize(data.number_instruments);
    individual_offset_prior.resize(data.number_instruments - 1);

    size_t N = data.N();
    nseries = 1 + data.number_indicators / 2;
    size_t Nfull = nseries * N;

    // resize RV model vector
    mu.resize(N);

    // each activity indicator has uncertainties, so the number of 
    // series is 1 + number_indicators / 2 corresponding to RVs + AIi
    zero_points.resize(nseries - 1);
    zero_points_prior.resize(nseries - 1);

    series_jitters.resize(nseries - 1);
    series_jitters_prior.resize(nseries - 1);

    t_full.resize(Nfull);
    yerr_full.resize(Nfull);

    for (size_t j = 0; j < N; j++)
    {
        for (size_t i = 0; i < nseries; i++)
        {
            t_full[nseries*j + i] = data.t[j];
            if (i == 0)
                yerr_full[nseries*j + i] = data.sig[j];
            else
                yerr_full[nseries*j + i] = data.actind[2 * i - 1][j];
        }
    }

    dt = t_full.segment(1, Nfull-1).array() - t_full.segment(0, Nfull-1).array();

    // the assumption of simultaneous series makes building series_index relatively easy
    Eigen::ArrayXi ind = Eigen::ArrayXi::LinSpaced(N, 0, N);
    for (size_t i = 0; i < nseries; i++)
    {
        series_index.push_back(ind * nseries + i);
    }

    // amplitudes for GP and GP derivative terms
    alpha.resize(nseries);
    beta.resize(nseries);

    // set default conditional priors that depend on data
    auto conditional = planets.get_conditional_prior();
    conditional->set_default_priors(data);

}

void SPLEAFmodel::set_known_object(size_t n)
{
    known_object = true;
    n_known_object = n;

    KO_Pprior.resize(n);
    KO_Kprior.resize(n);
    KO_eprior.resize(n);
    KO_phiprior.resize(n);
    KO_wprior.resize(n);

    KO_P.resize(n);
    KO_K.resize(n);
    KO_e.resize(n);
    KO_phi.resize(n);
    KO_w.resize(n);
}

void SPLEAFmodel::set_transiting_planet(size_t n)
{
    transiting_planet = true;
    n_transiting_planet = n;

    TR_Pprior.resize(n);
    TR_Kprior.resize(n);
    TR_eprior.resize(n);
    TR_Tcprior.resize(n);
    TR_wprior.resize(n);

    TR_P.resize(n);
    TR_K.resize(n);
    TR_e.resize(n);
    TR_Tc.resize(n);
    TR_w.resize(n);
}

/* set default priors if the user didn't change them */

void SPLEAFmodel::setPriors()  // BUG: should be done by only one thread!
{
    // betaprior = make_prior<Gaussian>(0, 1);

    if (!Cprior)
        Cprior = make_prior<Uniform>(data.get_RV_min(), data.get_RV_max());

    if (!Jprior)
        Jprior = make_prior<ModifiedLogUniform>(min(1.0, 0.1*data.get_max_RV_span()), data.get_max_RV_span());

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

    for (size_t j = 0; j < nseries - 1; j++)
    {
        if (!zero_points_prior[j])
            zero_points_prior[j] = make_prior<Uniform>( data.get_actind_min(2*j), data.get_actind_max(2*j) );
        if (!series_jitters_prior[j])
            series_jitters_prior[j] = make_prior<Uniform>( 0, data.get_actind_span(2*j) );
    }


    if (known_object) { // KO mode!
        // if (n_known_object == 0) cout << "Warning: `known_object` is true, but `n_known_object` is set to 0";
        for (int i = 0; i < n_known_object; i++){
            if (!KO_Pprior[i] || !KO_Kprior[i] || !KO_eprior[i] || !KO_phiprior[i] || !KO_wprior[i])
                throw std::logic_error("When known_object=true, please set priors for each (KO_Pprior, KO_Kprior, KO_eprior, KO_phiprior, KO_wprior)");
        }
    }

    /* GP parameters */
    switch (kernel)
    {
    case spleaf_matern32:
        if (!eta1_prior)
            eta1_prior = make_prior<LogUniform>(0.1, data.get_max_RV_span());
        if (!eta2_prior)
            eta2_prior = make_prior<LogUniform>(1, data.get_timespan());
        break;
    case spleaf_sho:
        if (!eta1_prior)
            eta1_prior = make_prior<LogUniform>(0.1, data.get_max_RV_span());
        if (!eta3_prior)
            eta3_prior = make_prior<LogUniform>(1, data.get_timespan());
        if (!Q_prior)
            Q_prior = make_prior<Uniform>(0.1, 10);
        break;
    case spleaf_mep:
        if (!eta1_prior)
            eta1_prior = make_prior<LogUniform>(0.1, data.get_max_RV_span());
        if (!eta2_prior)
            eta2_prior = make_prior<LogUniform>(1, data.get_timespan());
        if (!eta3_prior)
            eta3_prior = make_prior<LogUniform>(1, data.get_timespan());
        if (!eta4_prior)
            eta4_prior = make_prior<Uniform>(0.2, 5);
        break;

    default:
        break;
    }

    if (!alpha0_prior)
        alpha0_prior = make_prior<Uniform>(0.0, data.get_RV_span());
    if (!alpha_prior)
        alpha_prior = make_prior<Uniform>(-data.get_RV_span(), data.get_RV_span());
    if (!beta_prior)
        beta_prior = make_prior<Uniform>(-10*data.get_RV_span(), 10*data.get_RV_span());
}


void SPLEAFmodel::from_prior(RNG& rng)
{
    // preliminaries
    setPriors();
    save_setup();

    planets.from_prior(rng);
    planets.consolidate_diff();

    background = Cprior->generate(rng);

    for (size_t j = 0; j < nseries - 1; j++)
        zero_points[j] = zero_points_prior[j]->generate(rng);

    if(data._multi)
    {
        for (size_t i = 0; i < offsets.size(); i++)
            offsets[i] = individual_offset_prior[i]->generate(rng);
        for (size_t i = 0; i < jitters.size(); i++)
            jitters[i] = Jprior->generate(rng);
    }
    else
    {
        extra_sigma = Jprior->generate(rng);
    }

    for (size_t j = 0; j < nseries - 1; j++)
        series_jitters[j] = series_jitters_prior[j]->generate(rng);


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
    switch (kernel)
    {
    case spleaf_matern32:
        eta1 = eta1_prior->generate(rng);  // m/s
        eta2 = eta2_prior->generate(rng); // days
        break;

    case spleaf_sho:
        eta1 = eta1_prior->generate(rng);  // m/s
        eta3 = eta3_prior->generate(rng); // days
        Q = Q_prior->generate(rng);
        break;

    case spleaf_mep:
        eta1 = eta1_prior->generate(rng);  // m/s
        eta2 = eta2_prior->generate(rng); // days
        eta3 = eta3_prior->generate(rng); // days
        eta4 = eta4_prior->generate(rng);
        break;

    default:
        break;
    }

    for (size_t j = 0; j < nseries; j++)
    {
        if (j > 0) alpha[j] = alpha_prior->generate(rng);
        beta[j] = beta_prior->generate(rng);
    }
    alpha[0] = alpha0_prior->generate(rng);

    calculate_mu();
}

/**
 * @brief Calculate the full RV model
 * 
*/
void SPLEAFmodel::calculate_mu()
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

        // if(data.indicator_correlations)
        // {
        //     for(size_t i=0; i<N; i++)
        //     {
        //         for(size_t j = 0; j < data.number_indicators; j++)
        //            mu[i] += betas[j] * data.actind[j][i];
        //     }   
        // }

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


void SPLEAFmodel::remove_known_object()
{
    double f, v, ti, Tp;
    for (int j = 0; j < n_known_object; j++) {
        auto v = brandt::keplerian(data.t, KO_P[j], KO_K[j], KO_e[j], KO_w[j], KO_phi[j], data.M0_epoch);
        for (size_t i = 0; i < data.N(); i++) {
            mu[i] -= v[i];
        }
    }
}

void SPLEAFmodel::add_known_object()
{
    for (int j = 0; j < n_known_object; j++) {
        auto v = brandt::keplerian(data.t, KO_P[j], KO_K[j], KO_e[j], KO_w[j], KO_phi[j], data.M0_epoch);
        for (size_t i = 0; i < data.N(); i++) {
            mu[i] += v[i];
        }
    }
}

void SPLEAFmodel::remove_transiting_planet()
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

void SPLEAFmodel::add_transiting_planet()
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

int SPLEAFmodel::is_stable() const
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


double SPLEAFmodel::perturb(RNG& rng)
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
            case spleaf_matern32:
                if (rng.rand() <= 0.5)
                    eta1_prior->perturb(eta1, rng);
                else
                    eta2_prior->perturb(eta2, rng);
                break;

            case spleaf_sho:
                if (rng.rand() <= 0.33330)
                    eta1_prior->perturb(eta1, rng);
                else if (rng.rand() <= 0.5)
                    eta3_prior->perturb(eta3, rng);
                else
                    Q_prior->perturb(Q, rng);
                break;

            case spleaf_mep:
                if (rng.rand() <= 0.25)
                    eta1_prior->perturb(eta1, rng);
                else if(rng.rand() <= 0.33330)
                    eta2_prior->perturb(eta2, rng);
                else if(rng.rand() <= 0.5)
                    eta3_prior->perturb(eta3, rng);
                else
                    eta4_prior->perturb(eta4, rng);
                break;

            default:
                break;
        }
    }
    else if(rng.rand() <= 0.5) // perturb GP and GP' coefficients
    {
        alpha0_prior->perturb(alpha[0], rng);
        for (int j = 0; j < nseries; j++)
        {
            if (j > 0) alpha_prior->perturb(alpha[j], rng);
            beta_prior->perturb(beta[j], rng);
        }
    }
    else if(rng.rand() <= 0.5) // perturb jitter(s) + known_object
    {
        if(data._multi)
        {
            for (size_t i = 0; i < jitters.size(); i++)
                Jprior->perturb(jitters[i], rng);
        }
        else
        {
            Jprior->perturb(extra_sigma, rng);
        }

        for (size_t j = 0; j < nseries - 1; j++)
        {
            series_jitters_prior[j]->perturb(series_jitters[j], rng);
        }

        if (known_object)
        {
            remove_known_object();

            for (int i = 0; i < n_known_object; i++)
            {
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
            mu[i] -= background;
            if(trend) {
                mu[i] -= slope * (data.t[i] - tmid) +
                            quadr * pow(data.t[i] - tmid, 2) +
                            cubic * pow(data.t[i] - tmid, 3);
            }
            if(data._multi) {
                for(size_t j=0; j<offsets.size(); j++){
                    if (data.obsi[i] == j+1) { mu[i] -= offsets[j]; }
                }
            }

            // if(data.indicator_correlations) {
            //     for(size_t j = 0; j < data.number_indicators; j++){
            //         mu[i] -= betas[j] * actind[j][i];
            //     }
            // }
        }

        // propose new vsys
        Cprior->perturb(background, rng);

        // propose new instrument offsets
        if (data._multi){
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

        // propose new activity indicator zero-points
        for (size_t j = 0; j < nseries - 1; j++)
        {
            zero_points_prior[j]->perturb(zero_points[j], rng);
        }


        for(size_t i=0; i<mu.size(); i++)
        {
            mu[i] += background;
            if(trend) {
                mu[i] += slope * (data.t[i] - tmid) +
                            quadr * pow(data.t[i] - tmid, 2) +
                            cubic * pow(data.t[i] - tmid, 3);
            }
            if(data._multi) {
                for(size_t j=0; j<offsets.size(); j++){
                    if (data.obsi[i] == j+1) { mu[i] += offsets[j]; }
                }
            }

            // if(data.indicator_correlations) {
            //     for(size_t j = 0; j < data.number_indicators; j++){
            //         mu[i] += betas[j]*actind[j][i];
            //     }
            // }
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
double SPLEAFmodel::log_likelihood()
{
    size_t N = data.N();
    const auto& y = data.get_y();
    const auto& actind = data.get_actind();
    const auto& sig = data.get_sig();
    // const auto& obsi = data.get_obsi();

    double logL;

    if (enforce_stability){
        int stable = is_stable();
        if (stable != 0)
            return -std::numeric_limits<double>::infinity();
    }

    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    VectorXd residual(nseries * N);  // residual vector (observed y minus model y)
    VectorXd diagonal = yerr_full.array().square();  // diagonal of covariance matrix, including RV errors and jitters

    int k = 0;
    for (auto &ik : series_index)
    {
        if (k == 0)
            diagonal(ik).array() += extra_sigma * extra_sigma;
        else
            diagonal(ik).array() += series_jitters[k - 1] * series_jitters[k - 1];
        k++;
    }

    for (size_t j = 0; j < N; j++)
    {
        for (size_t i = 0; i < nseries; i++)
        {
            if (i == 0)
                residual(nseries*j + i) = y[j] - mu[j];
            else
                residual(nseries*j + i) = data.actind[2 * i - 2][j] - zero_points[i - 1];
        }
    }

    // cout << endl;
    // cout << "diagonal: " << diagonal.transpose() << endl << endl;
    // cout << "residual: " << residual.transpose() << endl << endl;
    // cout << background << endl;
    // for (size_t i=0; i<nseries-1; i++)
    // {
    //     cout << "zero_points[" << i << "]: " << zero_points[i] << endl;
    // }
    // abort();

    switch (kernel)
    {
    case spleaf_matern32:
        logL = spleaf_loglike_multiseries<spleaf_Matern32Kernel, 2>(residual, t_full, diagonal, dt, series_index,
                                                                     {eta1, eta2}, alpha, beta);
        break;

    case spleaf_sho:
        logL = spleaf_loglike_multiseries<spleaf_SHOKernel, 3>(residual, t_full, diagonal, dt, series_index,
                                                                {eta1, eta3, Q}, alpha, beta);
        break;
    
    case spleaf_mep:
        logL = spleaf_loglike_multiseries<spleaf_MEPKernel, 4>(residual, t_full, diagonal, dt, series_index,
                                                               {eta1, eta2, eta3, eta4}, alpha, beta);
        // cout << "logL: " << logL << endl;
        // abort();
        break;
    
    default:
        abort();
        break;
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


void SPLEAFmodel::print(std::ostream& out) const
{
    // output precision
    out.setf(ios::fixed,ios::floatfield);
    out.precision(8);

    if (data._multi)
    {
        for (size_t j = 0; j < jitters.size(); j++)
            out << jitters[j] << '\t';
    }
    else
        out << extra_sigma << '\t';

    for (size_t j = 0; j < nseries - 1; j++)
        out << series_jitters[j] << '\t';


    if(trend)
    {
        out.precision(15);
        if (degree >= 1) out << slope << '\t';
        if (degree >= 2) out << quadr << '\t';
        if (degree == 3) out << cubic << '\t';
        out.precision(8);
    }
        
    if (data._multi){
        for (size_t j = 0; j < offsets.size(); j++)
        {
            out << offsets[j] << '\t';
        }
    }

    // if(data.indicator_correlations){
    //     for (int j = 0; j < data.number_indicators; j++)
    //     {
    //         out<<betas[j]<<'\t';
    //     }
    // }

    // write GP parameters
    switch (kernel)
    {
    case spleaf_matern32:
        out << eta1 << '\t' << eta2 << '\t';
        break;

    case spleaf_sho:
        out << eta1 << '\t' << eta3 << '\t' << Q << '\t';
        break;

    case spleaf_mep:
        out << eta1 << '\t' << eta2 << '\t' << eta3 << '\t' << eta4 << '\t';
        break;

    default:
        break;
    }
    for (size_t j = 0; j < nseries; j++)
    {
        out << alpha[j] << '\t' << beta[j] << '\t';
    }

    if(known_object){ // KO mode!
        for (auto P: KO_P) out << P << "\t";
        for (auto K: KO_K) out << K << "\t";
        for (auto phi: KO_phi) out << phi << "\t";
        for (auto e: KO_e) out << e << "\t";
        for (auto w: KO_w) out << w << "\t";
    }

    planets.print(out);

    out << staleness << '\t';

    for (size_t j = 0; j < nseries - 1; j++)
        out << zero_points[j] << '\t';

    out << background;
}


string SPLEAFmodel::description() const
{
    string desc;
    string sep = "   ";

    if (data._multi)
    {
        for (size_t j = 0; j < jitters.size(); j++)
            desc += "jitter" + std::to_string(j + 1) + sep;
    }
    else
        desc += "extra_sigma" + sep;


    for (size_t j = 0; j < nseries - 1; j++)
        desc += "j" + std::to_string(j + 1) + sep;

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

    // if(data.indicator_correlations){
    //     for(size_t j=0; j<data.number_indicators; j++){
    //         desc += "beta" + std::to_string(j+1) + sep;
    //     }
    // }

    // GP parameters
    switch (kernel)
    {
        case spleaf_matern32:
            desc += "eta1" + sep + "eta2" + sep;
            break;

        case spleaf_sho:
            desc += "eta1" + sep + "eta3" + sep + "Q" + sep;
            break;

        case spleaf_mep:
            desc += "eta1" + sep + "eta2" + sep + "eta3" + sep + "eta4" + sep;
            break;

        default:
            break;
    }
    for (size_t j = 0; j < nseries; j++)
    {
        desc += "alpha" + std::to_string(j+1) + sep;
        desc += "beta" + std::to_string(j+1) + sep;
    }

    if(known_object) { // KO mode!
        for (int i = 0; i < n_known_object; i++)
            desc += "KO_P" + std::to_string(i) + sep;
        for (int i = 0; i < n_known_object; i++)
            desc += "KO_K" + std::to_string(i) + sep;
        for (int i = 0; i < n_known_object; i++)
            desc += "KO_phi" + std::to_string(i) + sep;
        for (int i = 0; i < n_known_object; i++)
            desc += "KO_ecc" + std::to_string(i) + sep;
        for (int i = 0; i < n_known_object; i++)
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
    
    for (size_t j = 0; j < nseries - 1; j++)
    {
        desc += "zp" + std::to_string(j + 1) + sep;
    }

    desc += "vsys";

    return desc;
}

/**
 * Save the options of the current model in a INI file.
 * 
*/
void SPLEAFmodel::save_setup() {
	std::fstream fout("kima_model_setup.txt", std::ios::out);
    fout << std::boolalpha;

    fout << "; " << timestamp() << endl << endl;

    fout << "[kima]" << endl;

    fout << "model: " << "SPLEAFmodel" << endl << endl;
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
    fout << "nseries: " << nseries << endl;
    fout << "kernel: " << kernel << endl;
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
    
    for (size_t j = 0; j < nseries - 1; j++)
        fout << "series_jitters_prior_" << j + 1 << ": " << *series_jitters_prior[j] << endl;


    if (trend){
        if (degree >= 1) fout << "slope_prior: " << *slope_prior << endl;
        if (degree >= 2) fout << "quadr_prior: " << *quadr_prior << endl;
        if (degree == 3) fout << "cubic_prior: " << *cubic_prior << endl;
    }
    if (data._multi)
        fout << "offsets_prior: " << *offsets_prior << endl;

    for (size_t j = 0; j < nseries - 1; j++)
    {
        fout << "zero_points_prior_" << j + 1 << ": " << *zero_points_prior[j] << endl;
    }


    fout << endl << "[priors.GP]" << endl;
    switch (kernel)
    {
    case spleaf_matern32:
        fout << "eta1_prior: " << *eta1_prior << endl;
        fout << "eta2_prior: " << *eta2_prior << endl;
        break;
    case spleaf_sho:
        fout << "eta1_prior: " << *eta1_prior << endl;
        fout << "eta3_prior: " << *eta3_prior << endl;
        fout << "Q_prior: " << *Q_prior << endl;
        break;
    case spleaf_mep:
        fout << "eta1_prior: " << *eta1_prior << endl;
        fout << "eta2_prior: " << *eta2_prior << endl;
        fout << "eta3_prior: " << *eta3_prior << endl;
        fout << "eta4_prior: " << *eta4_prior << endl;
        break;
    default:
        break;
    }

    fout << "alpha0_prior: " << *alpha0_prior << endl;
    for (int j = 0; j < nseries; j++)
    {
        if (j > 0) fout << "alpha" << j << "_prior: " << *alpha_prior << endl;
        fout << "beta" << j << "_prior: " << *beta_prior << endl;
    }


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
        // for(size_t i=0; i<n_known_object; i++){
        //     fout << "Pprior_" << i << ": " << *KO_Pprior[i] << endl;
        //     fout << "Kprior_" << i << ": " << *KO_Kprior[i] << endl;
        //     fout << "eprior_" << i << ": " << *KO_eprior[i] << endl;
        //     fout << "phiprior_" << i << ": " << *KO_phiprior[i] << endl;
        //     fout << "wprior_" << i << ": " << *KO_wprior[i] << endl;
        // }
    }

    fout << endl;
	fout.close();
}


using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;

class SPLEAFmodel_publicist : public SPLEAFmodel {
    public:
        using SPLEAFmodel::fix;
        using SPLEAFmodel::npmax;
        using SPLEAFmodel::data;
        // 
        using SPLEAFmodel::trend;
        using SPLEAFmodel::degree;
        using SPLEAFmodel::star_mass;
        using SPLEAFmodel::enforce_stability;
};

NB_MODULE(SPLEAFmodel, m) {
    nb::class_<SPLEAFmodel>(m, "SPLEAFmodel")
        .def(nb::init<bool&, int&, RVData&>(), "fix"_a, "npmax"_a, "data"_a)
        //
        .def_rw("directory", &SPLEAFmodel::directory,
                "directory where the model ran")
        // 
        .def_rw("fix", &SPLEAFmodel_publicist::fix,
                "whether the number of Keplerians is fixed")
        .def_rw("npmax", &SPLEAFmodel_publicist::npmax,
                "maximum number of Keplerians")
        .def_ro("data", &SPLEAFmodel_publicist::data,
                "the data")
        // 

        .def_rw("trend", &SPLEAFmodel_publicist::trend,
                "whether the model includes a polynomial trend")
        .def_rw("degree", &SPLEAFmodel_publicist::degree,
                "degree of the polynomial trend")

        // KO mode
        .def("set_known_object", &SPLEAFmodel::set_known_object)
        .def_prop_ro("known_object", [](SPLEAFmodel &m) { return m.get_known_object(); },
                     "whether the model includes (better) known extra Keplerian curve(s)")
        .def_prop_ro("n_known_object", [](SPLEAFmodel &m) { return m.get_n_known_object(); },
                     "how many known objects")

        // transiting planets
        .def("set_transiting_planet", &SPLEAFmodel::set_transiting_planet)
        .def_prop_ro("transiting_planet", [](SPLEAFmodel &m) { return m.get_transiting_planet(); },
                     "whether the model includes transiting planet(s)")
        .def_prop_ro("n_transiting_planet", [](SPLEAFmodel &m) { return m.get_n_transiting_planet(); },
                     "how many transiting planets")

        //
        .def_rw("star_mass", &SPLEAFmodel_publicist::star_mass,
                "stellar mass [Msun]")
        .def_rw("enforce_stability", &SPLEAFmodel_publicist::enforce_stability, 
                "whether to enforce AMD-stability")

        // 
        .def_prop_rw("kernel",
            [](SPLEAFmodel &m) { return m.kernel; },
            [](SPLEAFmodel &m, KernelType k) { m.kernel = k; },
            "GP kernel to use")

        // priors
        .def_prop_rw("Cprior",
            [](SPLEAFmodel &m) { return m.Cprior; },
            [](SPLEAFmodel &m, distribution &d) { m.Cprior = d; },
            "Prior for the systemic velocity")

        .def_prop_rw("zero_points_prior",
            [](SPLEAFmodel &m) { return m.zero_points_prior; },
            [](SPLEAFmodel &m, std::vector<distribution>& vd) { m.zero_points_prior = vd; },
            "Priors for the activity indicator zero points")

        .def_prop_rw("Jprior",
            [](SPLEAFmodel &m) { return m.Jprior; },
            [](SPLEAFmodel &m, distribution &d) { m.Jprior = d; },
            "Prior for the extra white noise (jitter)")

        .def_prop_rw("series_jitters_prior",
            [](SPLEAFmodel &m) { return m.series_jitters_prior; },
            [](SPLEAFmodel &m, std::vector<distribution>& vd) { m.series_jitters_prior = vd; },
            "Priors for the activity indicator jitters")

        .def_prop_rw("slope_prior",
            [](SPLEAFmodel &m) { return m.slope_prior; },
            [](SPLEAFmodel &m, distribution &d) { m.slope_prior = d; },
            "Prior for the slope")
        .def_prop_rw("quadr_prior",
            [](SPLEAFmodel &m) { return m.quadr_prior; },
            [](SPLEAFmodel &m, distribution &d) { m.quadr_prior = d; },
            "Prior for the quadratic coefficient of the trend")
        .def_prop_rw("cubic_prior",
            [](SPLEAFmodel &m) { return m.cubic_prior; },
            [](SPLEAFmodel &m, distribution &d) { m.cubic_prior = d; },
            "Prior for the cubic coefficient of the trend")

        // priors for the GP hyperparameters
        .def_prop_rw("eta1_prior",
            [](SPLEAFmodel &m) { return m.eta1_prior; },
            [](SPLEAFmodel &m, distribution &d) { m.eta1_prior = d; },
            "Prior for η1, the GP 'amplitude'")
        .def_prop_rw("eta2_prior",
            [](SPLEAFmodel &m) { return m.eta2_prior; },
            [](SPLEAFmodel &m, distribution &d) { m.eta2_prior = d; },
            "Prior for η2, the GP correlation timescale")
        .def_prop_rw("eta3_prior",
            [](SPLEAFmodel &m) { return m.eta3_prior; },
            [](SPLEAFmodel &m, distribution &d) { m.eta3_prior = d; },
            "Prior for η3, the GP period")
        .def_prop_rw("eta4_prior",
            [](SPLEAFmodel &m) { return m.eta4_prior; },
            [](SPLEAFmodel &m, distribution &d) { m.eta4_prior = d; },
            "Prior for η4, the recurrence timescale or (inverse) harmonic complexity")
        .def_prop_rw("Q_prior",
            [](SPLEAFmodel &m) { return m.Q_prior; },
            [](SPLEAFmodel &m, distribution &d) { m.Q_prior = d; },
            "Prior for Q, the quality factor in SHO kernels")


        .def_prop_rw("alpha0_prior",
            [](SPLEAFmodel &m) { return m.alpha0_prior; },
            [](SPLEAFmodel &m, distribution &d) { m.alpha0_prior = d; },
            "")
        .def_prop_rw("alpha_prior",
            [](SPLEAFmodel &m) { return m.alpha_prior; },
            [](SPLEAFmodel &m, distribution &d) { m.alpha_prior = d; },
            "")
        .def_prop_rw("beta_prior",
            [](SPLEAFmodel &m) { return m.beta_prior; },
            [](SPLEAFmodel &m, distribution &d) { m.beta_prior = d; },
            "")


        // conditional object
        .def_prop_rw("conditional",
                     [](SPLEAFmodel &m) { return m.get_conditional_prior(); },
                     [](SPLEAFmodel &m, RVConditionalPrior& c) { /* does nothing */ });
        // // covariance kernel
        // .def_prop_rw("kernel",
        //              [](SPLEAFmodel &m) { return m.get_kernel(); },
        //              [](SPLEAFmodel &m, Term& k) { /* does nothing */ });
}