#include "GPmodel.h"

using namespace Eigen;

#define TIMING false

const double halflog2pi = 0.5*log(2.*M_PI);


void GPmodel::initialize_from_data(RVData& data)
{
    offsets.resize(data.number_instruments - 1);
    jitters.resize(data.number_instruments);
    betas.resize(data.number_indicators);
    individual_offset_prior.resize(data.number_instruments - 1);

    // resize RV model vector
    mu.resize(data.N());
    // resize covariance matrix
    C.resize(data.N(), data.N());

    // set default conditional priors that depend on data
    auto conditional = planets.get_conditional_prior();
    conditional->set_default_priors(data);
}

void GPmodel::set_known_object(size_t n)
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

void GPmodel::set_transiting_planet(size_t n)
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

void GPmodel::setPriors()  // BUG: should be done by only one thread!
{
    betaprior = make_prior<Gaussian>(0, 1);

    if (!Cprior)
        Cprior = make_prior<Uniform>(data.get_RV_min(), data.get_RV_max());

    if (!Jprior)
        Jprior = make_prior<ModifiedLogUniform>(
            min(1.0, 0.1*data.get_max_RV_span()), 
            data.get_max_RV_span()
        );

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
            if (!KO_Pprior[i] || !KO_Kprior[i] || !KO_eprior[i] || !KO_phiprior[i] || !KO_wprior[i])
            {
                std::string msg = "When known_object=true, must set priors for each of KO_Pprior, KO_Kprior, KO_eprior, KO_phiprior, KO_wprior";
                throw std::logic_error(msg);
            }
        }
    }

    if (transiting_planet) {
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
    if (!eta1_prior)
        eta1_prior = make_prior<LogUniform>(0.1, 100);
    if (!eta2_prior)
        eta2_prior = make_prior<LogUniform>(1, 100);
    if (!eta3_prior)
        eta3_prior = make_prior<Uniform>(10, 40);
    if (!eta4_prior)
        eta4_prior = make_prior<Uniform>(0.2, 5);

}


void GPmodel::from_prior(RNG& rng)
{
    // preliminaries
    setPriors();
    save_setup();

    planets.from_prior(rng);
    planets.consolidate_diff();

    background = Cprior->generate(rng);

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
    }


    if(trend)
    {
        if (degree >= 1) slope = slope_prior->generate(rng);
        if (degree >= 2) quadr = quadr_prior->generate(rng);
        if (degree == 3) cubic = cubic_prior->generate(rng);
    }

    if (indicator_correlations)
    {
        for (int i = 0; i < data.number_indicators; i++)
            betas[i] = betaprior->generate(rng);
    }

    if (known_object) { // KO mode!
        for (int i = 0; i < n_known_object; i++)
        {
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
    eta1 = eta1_prior->generate(rng);  // m/s
    eta2 = eta2_prior->generate(rng); // days
    eta3 = eta3_prior->generate(rng); // days
    eta4 = eta4_prior->generate(rng);

    calculate_mu();
    calculate_C();
}

/**
 * @brief Calculate the full RV model
 * 
*/
void GPmodel::calculate_mu()
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

        if(indicator_correlations)
        {
            for (size_t i = 0; i < N; i++)
            {
                for (size_t j = 0; j < data.number_indicators; j++)
                    mu[i] += betas[j] * data.actind[j][i];
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

/// @brief Fill the GP covariance matrix
void GPmodel::calculate_C()
{
    size_t N = data.N();

    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    /* This implements the "standard" quasi-periodic kernel, see R&W2006 */
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            double r = data.t[i] - data.t[j];
            C(i, j) = eta1*eta1*exp(-0.5*pow(r/eta2, 2)
                        -2.0*pow(sin(M_PI*r/eta3)/eta4, 2) );

            if(i==j)
            {
                double sig = data.sig[i];
                if (data._multi)
                {
                    double jit = jitters[data.obsi[i]-1];
                    C(i, j) += sig*sig + jit*jit;
                }
                else
                {
                    C(i, j) += sig*sig + extra_sigma*extra_sigma;
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

void GPmodel::remove_known_object()
{
    double f, v, ti, Tp;
    for (int j = 0; j < n_known_object; j++) {
        auto v = brandt::keplerian(data.t, KO_P[j], KO_K[j], KO_e[j], KO_w[j], KO_phi[j], data.M0_epoch);
        for (size_t i = 0; i < data.N(); i++) {
            mu[i] -= v[i];
        }
    }
}

void GPmodel::add_known_object()
{
    for (int j = 0; j < n_known_object; j++) {
        auto v = brandt::keplerian(data.t, KO_P[j], KO_K[j], KO_e[j], KO_w[j], KO_phi[j], data.M0_epoch);
        for (size_t i = 0; i < data.N(); i++) {
            mu[i] += v[i];
        }
    }
}

void GPmodel::remove_transiting_planet()
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

void GPmodel::add_transiting_planet()
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
int GPmodel::is_stable() const
{
    // Get the components
    const vector< vector<double> >& components = planets.get_components();
    if (components.size() == 0 && !known_object)
        return 0;
    
    int stable_planets = 0;
    int stable_known_object = 0;
    int stable_transiting_planet = 0;

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

    return stable_planets + stable_known_object + stable_transiting_planet;
}


double GPmodel::perturb(RNG& rng)
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
        if (rng.rand() <= 0.25)
        {
            eta1_prior->perturb(eta1, rng);
        }
        else if(rng.rand() <= 0.33330)
        {
            eta3_prior->perturb(eta3, rng);
        }
        else if(rng.rand() <= 0.5)
        {
            eta2_prior->perturb(eta2, rng);
        }
        else
        {
            eta4_prior->perturb(eta4, rng);
        }
        
        calculate_C(); // recalculate covariance matrix
    }
    else if(rng.rand() <= 0.5) // perturb jitter(s) + known_object
    {
        if(data._multi)
        {
            for(int i=0; i<jitters.size(); i++)
                Jprior->perturb(jitters[i], rng);
        }
        else
        {
            Jprior->perturb(extra_sigma, rng);
        }

        calculate_C(); // recalculate covariance matrix

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
        for(size_t i=0; i<mu.size(); i++)
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

            if(indicator_correlations) {
                for(size_t j = 0; j < data.number_indicators; j++){
                    mu[i] -= betas[j] * actind[j][i];
                }
            }
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

        // propose new indicator correlations
        if(indicator_correlations){
            for(size_t j = 0; j < data.number_indicators; j++){
                betaprior->perturb(betas[j], rng);
            }
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

            if(indicator_correlations) {
                for(size_t j = 0; j < data.number_indicators; j++){
                    mu[i] += betas[j]*actind[j][i];
                }
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
double GPmodel::log_likelihood() const
{
    size_t N = data.N();
    const auto& y = data.get_y();
    const auto& sig = data.get_sig();
    const auto& obsi = data.get_obsi();

    double logL = 0.;

    if (enforce_stability){
        int stable = is_stable();
        if (stable != 0)
            return -std::numeric_limits<double>::infinity();
    }

    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    /** The following code calculates the log likelihood of a GP model */
    // residual vector (observed y minus model y)
    VectorXd residual(y.size());
    for (size_t i = 0; i < y.size(); i++)
        residual(i) = y[i] - mu[i];

    // perform the cholesky decomposition of C
    Eigen::LLT<Eigen::MatrixXd> cholesky = C.llt();
    // get the lower triangular matrix L
    MatrixXd L = cholesky.matrixL();

    double logDeterminant = 0.;
    for (size_t i = 0; i < y.size(); i++)
        logDeterminant += 2. * log(L(i, i));

    VectorXd solution = cholesky.solve(residual);

    // y*solution
    double exponent = 0.;
    for (size_t i = 0; i < y.size(); i++)
        exponent += residual(i) * solution(i);

    logL = -0.5*y.size()*log(2*M_PI) - 0.5*logDeterminant - 0.5*exponent;

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


void GPmodel::print(std::ostream& out) const
{
    // output precision
    out.setf(ios::fixed,ios::floatfield);
    out.precision(8);

    if (data._multi)
    {
        for(int j=0; j<jitters.size(); j++)
            out<<jitters[j]<<'\t';
    }
    else
        out<<extra_sigma<<'\t';

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
    }

    if(indicator_correlations){
        for (int j = 0; j < data.number_indicators; j++)
        {
            out << betas[j] << '\t';
        }
    }

    // write GP parameters
    out << eta1 << '\t' << eta2 << '\t' << eta3 << '\t' << eta4 << '\t';

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

    planets.print(out);

    out << staleness << '\t';

    out << background;
}


string GPmodel::description() const
{
    string desc;
    string sep = "   ";

    if (data._multi)
    {
        for(int j=0; j<jitters.size(); j++)
           desc += "jitter" + std::to_string(j+1) + sep;
    }
    else
        desc += "extra_sigma" + sep;

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

    // GP parameters
    desc += "eta1" + sep + "eta2" + sep + "eta3" + sep + "eta4" + sep;

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
    
    desc += "vsys";

    return desc;
}

/**
 * Save the options of the current model in a INI file.
 * 
*/
void GPmodel::save_setup() {
	std::fstream fout("kima_model_setup.txt", std::ios::out);
    fout << std::boolalpha;

    time_t rawtime;
    time (&rawtime);
    fout << ";" << ctime(&rawtime) << endl;

    fout << "[kima]" << endl;

    fout << "model: " << "GPmodel" << endl << endl;
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
    fout << "indicator_correlations: " << indicator_correlations << endl;
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

    fout << endl << "[priors.GP]" << endl;
    fout << "eta1_prior: " << *eta1_prior << endl;
    fout << "eta2_prior: " << *eta2_prior << endl;
    fout << "eta3_prior: " << *eta3_prior << endl;
    fout << "eta4_prior: " << *eta4_prior << endl;
    fout << endl;

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

class GPmodel_publicist : public GPmodel
{
    public:
        using GPmodel::trend;
        using GPmodel::degree;
        using GPmodel::star_mass;
        using GPmodel::enforce_stability;
        using GPmodel::indicator_correlations;
};

NB_MODULE(GPmodel, m) {
    nb::class_<GPmodel>(m, "GPmodel")
        .def(nb::init<bool&, int&, RVData&>(), "fix"_a, "npmax"_a, "data"_a)
        //
        .def_rw("trend", &GPmodel_publicist::trend,
                "whether the model includes a polynomial trend")
        .def_rw("degree", &GPmodel_publicist::degree,
                "degree of the polynomial trend")
        
        // KO mode
        .def("set_known_object", &GPmodel::set_known_object)
        .def_prop_ro("known_object", [](GPmodel &m) { return m.get_known_object(); },
                     "whether the model includes (better) known extra Keplerian curve(s)")
        .def_prop_ro("n_known_object", [](GPmodel &m) { return m.get_n_known_object(); },
                     "how many known objects")

        // transiting planets
        .def("set_transiting_planet", &GPmodel::set_transiting_planet)
        .def_prop_ro("transiting_planet", [](GPmodel &m) { return m.get_transiting_planet(); },
                     "whether the model includes transiting planet(s)")
        .def_prop_ro("n_transiting_planet", [](GPmodel &m) { return m.get_n_transiting_planet(); },
                     "how many transiting planets")


        //
        .def_rw("star_mass", &GPmodel_publicist::star_mass,
                "stellar mass [Msun]")
        .def_rw("enforce_stability", &GPmodel_publicist::enforce_stability, 
                "whether to enforce AMD-stability")

        //
        .def_rw("indicator_correlations", &GPmodel_publicist::indicator_correlations, 
                "include in the model linear correlations with indicators")

        // priors
        .def_prop_rw("Cprior",
            [](GPmodel &m) { return m.Cprior; },
            [](GPmodel &m, distribution &d) { m.Cprior = d; },
            "Prior for the systemic velocity")
        .def_prop_rw("Jprior",
            [](GPmodel &m) { return m.Jprior; },
            [](GPmodel &m, distribution &d) { m.Jprior = d; },
            "Prior for the extra white noise (jitter)")
        .def_prop_rw("slope_prior",
            [](GPmodel &m) { return m.slope_prior; },
            [](GPmodel &m, distribution &d) { m.slope_prior = d; },
            "Prior for the slope")
        .def_prop_rw("quadr_prior",
            [](GPmodel &m) { return m.quadr_prior; },
            [](GPmodel &m, distribution &d) { m.quadr_prior = d; },
            "Prior for the quadratic coefficient of the trend")
        .def_prop_rw("cubic_prior",
            [](GPmodel &m) { return m.cubic_prior; },
            [](GPmodel &m, distribution &d) { m.cubic_prior = d; },
            "Prior for the cubic coefficient of the trend")
        .def_prop_rw("offsets_prior",
            [](GPmodel &m) { return m.offsets_prior; },
            [](GPmodel &m, distribution &d) { m.offsets_prior = d; },
            "Common prior for the between-instrument offsets")

        // priors for the GP hyperparameters
        .def_prop_rw("eta1_prior",
            [](GPmodel &m) { return m.eta1_prior; },
            [](GPmodel &m, distribution &d) { m.eta1_prior = d; },
            "Prior for η1, the GP 'amplitude'")
        .def_prop_rw("eta2_prior",
            [](GPmodel &m) { return m.eta2_prior; },
            [](GPmodel &m, distribution &d) { m.eta2_prior = d; },
            "Prior for η2, the GP correlation timescale")
        .def_prop_rw("eta3_prior",
            [](GPmodel &m) { return m.eta3_prior; },
            [](GPmodel &m, distribution &d) { m.eta3_prior = d; },
            "Prior for η3, the GP period")
        .def_prop_rw("eta4_prior",
            [](GPmodel &m) { return m.eta4_prior; },
            [](GPmodel &m, distribution &d) { m.eta4_prior = d; },
            "Prior for η4, the recurrence timescale or (inverse) harmonic complexity")

        // known object priors
        // ? should these setters check if known_object is true?
        .def_prop_rw("KO_Pprior",
                     [](GPmodel &m) { return m.KO_Pprior; },
                     [](GPmodel &m, std::vector<distribution>& vd) { m.KO_Pprior = vd; },
                     "Prior for KO orbital period")
        .def_prop_rw("KO_Kprior",
                     [](GPmodel &m) { return m.KO_Kprior; },
                     [](GPmodel &m, std::vector<distribution>& vd) { m.KO_Kprior = vd; },
                     "Prior for KO semi-amplitude")
        .def_prop_rw("KO_eprior",
                     [](GPmodel &m) { return m.KO_eprior; },
                     [](GPmodel &m, std::vector<distribution>& vd) { m.KO_eprior = vd; },
                     "Prior for KO eccentricity")
        .def_prop_rw("KO_wprior",
                     [](GPmodel &m) { return m.KO_wprior; },
                     [](GPmodel &m, std::vector<distribution>& vd) { m.KO_wprior = vd; },
                     "Prior for KO argument of periastron")
        .def_prop_rw("KO_phiprior",
                     [](GPmodel &m) { return m.KO_phiprior; },
                     [](GPmodel &m, std::vector<distribution>& vd) { m.KO_phiprior = vd; },
                     "Prior for KO mean anomaly(ies)")


        // conditional object
        .def_prop_rw("conditional",
                     [](GPmodel &m) { return m.get_conditional_prior(); },
                     [](GPmodel &m, RVConditionalPrior& c) { /* does nothing */ });
}