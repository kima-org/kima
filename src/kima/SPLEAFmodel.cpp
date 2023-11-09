#include "SPLEAFmodel.h"

#define TIMING false

const double halflog2pi = 0.5*log(2.*M_PI);


void SPLEAFmodel::initialize_from_data(RVData& data, Term& kernel)
{
    offsets.resize(data.number_instruments - 1);
    jitters.resize(data.number_instruments);
    // betas.resize(data.number_indicators);
    individual_offset_prior.resize(data.number_instruments - 1);

    // resize RV model vector
    size_t N = data.N();

    mu.resize(N);

    // initialize SPLEAF covariance
    assert ( kernel._priors_defined() && "kernel priors are defined");

    if (multi_series) // multiple series, create a MultiSeriesKernel
    { 
        nseries = 1 + data.number_indicators / 2;
        residuals.resize(N * nseries);

        VectorXd t_full(nseries * N);
        VectorXd yerr_full(nseries * N);
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

        // the assumption of simultaneous series makes building series_index
        // relatively easy
        Eigen::ArrayXi ind = Eigen::ArrayXi::LinSpaced(data.N(), 0, data.N());
        std::vector<Eigen::ArrayXi> series_index;
        for (size_t i = 0; i < nseries; i++) {
            series_index.push_back(ind * nseries + i);
        }

        // amplitudes for GP and GP derivative terms
        VectorXd alpha = VectorXd::Ones(nseries);
        VectorXd beta = VectorXd::Ones(nseries);

        ms = MultiSeriesKernel(kernel, series_index, alpha, beta);
        // err = new Error(yerr_full);
        cov = Cov(t_full, yerr_full, 0, ms._r);
        cov.link(ms);
    }
    else // just RVs, use the kernel directly
    {
        nseries = 1;
        residuals.resize(N);

        VectorXd _t = Eigen::Map<VectorXd, Eigen::Unaligned> (data.t.data(), data.t.size());
        VectorXd _yerr = Eigen::Map<VectorXd, Eigen::Unaligned> (data.sig.data(), data.sig.size());
        // err = new Error(_yerr);
        
        cov = Cov(_t, _yerr, 0, kernel._r);
        cov.link(kernel);
    }

    // set default conditional priors that depend on data
    auto conditional = planets.get_conditional_prior();
    conditional->set_default_priors(data);

}

/* set default priors if the user didn't change them */

void SPLEAFmodel::setPriors()  // BUG: should be done by only one thread!
{
    // betaprior = make_prior<Gaussian>(0, 1);

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
                throw std::logic_error("When known_object=true, please set priors for each (KO_Pprior, KO_Kprior, KO_eprior, KO_phiprior, KO_wprior)");
        }
    }

    // /* GP parameters */
    // if (!eta1_prior)
    //     eta1_prior = make_prior<LogUniform>(0.1, 100);
    // if (!eta2_prior)
    //     eta2_prior = make_prior<LogUniform>(1, 100);
    // if (!eta3_prior)
    //     eta3_prior = make_prior<Uniform>(10, 40);
    // if (!eta4_prior)
    //     eta4_prior = make_prior<Uniform>(0.2, 5);

}


void SPLEAFmodel::from_prior(RNG& rng)
{
    // preliminaries
    setPriors();
    save_setup();

    planets.from_prior(rng);
    planets.consolidate_diff();

    background = Cprior->generate(rng);

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


    if(trend)
    {
        if (degree >= 1) slope = slope_prior->generate(rng);
        if (degree >= 2) quadr = quadr_prior->generate(rng);
        if (degree == 3) cubic = cubic_prior->generate(rng);
    }

    // if (data.indicator_correlations)
    // {
    //     for (unsigned i=0; i<data.number_indicators; i++)
    //         betas[i] = betaprior->generate(rng);
    // }

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
    cov.generate(rng);

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
        cov.perturb(rng);
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

        // // propose new indicator correlations
        // if(data.indicator_correlations){
        //     for(size_t j = 0; j < data.number_indicators; j++){
        //         betaprior->perturb(betas[j], rng);
        //     }
        // }

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
    // const auto& sig = data.get_sig();
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

    for (size_t j = 0; j < nseries; j++)
    {
        for (size_t i = 0; i < N; i++)
        {
            if (j == 0)
                residuals[i * nseries] = y[i] - mu[i];
            else
                residuals[i * nseries + j] = actind[2*j - 2][i] - 0.0;
        }
    }

    logL = cov.loglike(residuals);
    // cout << residuals.transpose() << endl << endl;
    // cout << cov.to_string() << '\t' << cov.logdet() << '\t' << cov.chi2(residuals) << endl;
    // cout << logL << endl;

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
        for(size_t j=0; j<jitters.size(); j++)
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
        for (size_t j = 0; j < offsets.size(); j++)
        {
            out<<offsets[j]<<'\t';
        }
    }

    // if(data.indicator_correlations){
    //     for (int j = 0; j < data.number_indicators; j++)
    //     {
    //         out<<betas[j]<<'\t';
    //     }
    // }

    // write GP parameters
    cov.print(out);
    // out << eta1 << '\t' << eta2 << '\t' << eta3 << '\t' << eta4 << '\t';

    if(known_object){ // KO mode!
        for (auto P: KO_P) out << P << "\t";
        for (auto K: KO_K) out << K << "\t";
        for (auto phi: KO_phi) out << phi << "\t";
        for (auto e: KO_e) out << e << "\t";
        for (auto w: KO_w) out << w << "\t";
    }

    planets.print(out);

    out << staleness << '\t';

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
    cov.description(desc, sep);
    // desc += "eta1" + sep + "eta2" + sep + "eta3" + sep + "eta4" + sep;

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

    time_t rawtime;
    time (&rawtime);
    fout << ";" << ctime(&rawtime) << endl;

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
    fout << "multi_series: " << multi_series << endl;
    fout << "nseries: " << nseries << endl;
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

    // fout << endl << "[priors.GP]" << endl;
    // fout << "eta1_prior: " << *eta1_prior << endl;
    // fout << "eta2_prior: " << *eta2_prior << endl;
    // fout << "eta3_prior: " << *eta3_prior << endl;
    // fout << "eta4_prior: " << *eta4_prior << endl;
    // fout << endl;

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

NB_MODULE(SPLEAFmodel, m) {
    nb::class_<SPLEAFmodel>(m, "SPLEAFmodel")
        .def(nb::init<bool&, int&, RVData&, Term&, bool&>(), "fix"_a, "npmax"_a, "data"_a, "kernel"_a, "multi_series"_a)
        .def_prop_rw("trend",
                     [](SPLEAFmodel &m) { return m.get_trend(); },
                     [](SPLEAFmodel &m, bool t) { m.set_trend(t); })
        .def_prop_rw("degree",
                     [](SPLEAFmodel &m) { return m.get_degree(); },
                     [](SPLEAFmodel &m, double t) { m.set_degree(t); })
        // priors
        .def_prop_rw("Cprior",
            [](SPLEAFmodel &m) { return m.Cprior; },
            [](SPLEAFmodel &m, distribution &d) { m.Cprior = d; },
            "Prior for the systemic velocity")
        .def_prop_rw("Jprior",
            [](SPLEAFmodel &m) { return m.Jprior; },
            [](SPLEAFmodel &m, distribution &d) { m.Jprior = d; },
            "Prior for the extra white noise (jitter)")
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
        // conditional object
        .def_prop_rw("conditional",
                     [](SPLEAFmodel &m) { return m.get_conditional_prior(); },
                     [](SPLEAFmodel &m, RVConditionalPrior& c) { /* does nothing */ });
        // // covariance kernel
        // .def_prop_rw("kernel",
        //              [](SPLEAFmodel &m) { return m.get_kernel(); },
        //              [](SPLEAFmodel &m, Term& k) { /* does nothing */ });
}