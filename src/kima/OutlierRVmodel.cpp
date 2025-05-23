#include "OutlierRVmodel.h"

#define TIMING false

const double halflog2pi = 0.5*log(2.*M_PI);


void OutlierRVmodel::initialize_from_data(RVData& data)
{
    offsets.resize(data.number_instruments - 1);
    jitters.resize(data.number_instruments);
    betas.resize(data.number_indicators);
    individual_offset_prior.resize(data.number_instruments - 1);

    // resize RV model vector
    mu.resize(data.N());

    // set default conditional priors that depend on data
    auto conditional = planets.get_conditional_prior();
    conditional->set_default_priors(data);
}


/* set default priors if the user didn't change them */
void OutlierRVmodel::setPriors()  // BUG: should be done by only one thread!
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
                throw std::logic_error("When known_object=true, please set priors for each (KO_Pprior, KO_Kprior, KO_eprior, KO_phiprior, KO_wprior)");
        }
    }

    if (studentt) {
        if (!nu_prior)
            nu_prior = make_prior<LogUniform>(2, 1000);
    }


    // outlier model priors
    if (!outlier_mean_prior)
        outlier_mean_prior = make_prior<Uniform>(data.get_RV_min(), data.get_RV_max());
    if (!outlier_sigma_prior)
        outlier_sigma_prior = make_prior<Uniform>(0, data.get_max_RV_span());
    if (!outlier_Q_prior)
        outlier_Q_prior = make_prior<Uniform>(0, 1);

}


void OutlierRVmodel::from_prior(RNG& rng)
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

    if (data.indicator_correlations)
    {
        for (int i = 0; i < data.number_indicators; i++)
            betas[i] = betaprior->generate(rng);
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

    if (studentt)
        nu = nu_prior->generate(rng);


    outlier_background = outlier_mean_prior->generate(rng);
    outlier_sigma = outlier_sigma_prior->generate(rng);
    Q = outlier_Q_prior->generate(rng);    

    calculate_mu();
}

/**
 * @brief Calculate the full RV model
 * 
*/
void OutlierRVmodel::calculate_mu()
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

        if(data.indicator_correlations)
        {
            for(size_t i=0; i<N; i++)
            {
                for(size_t j = 0; j < data.number_indicators; j++)
                   mu[i] += betas[j] * data.actind[j][i];
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


void OutlierRVmodel::remove_known_object()
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

void OutlierRVmodel::add_known_object()
{
    for (int j = 0; j < n_known_object; j++) {
        auto v = brandt::keplerian(data.t, KO_P[j], KO_K[j], KO_e[j], KO_w[j], KO_phi[j], data.M0_epoch);
        for (size_t i = 0; i < data.N(); i++) {
            mu[i] += v[i];
        }
    }
}

int OutlierRVmodel::is_stable() const
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


double OutlierRVmodel::perturb(RNG& rng)
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

        if (studentt)
            nu_prior->perturb(nu, rng);


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

        outlier_mean_prior->perturb(outlier_background, rng);
        outlier_sigma_prior->perturb(outlier_sigma, rng);
        outlier_Q_prior->perturb(Q, rng);
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

            if(data.indicator_correlations) {
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
        if(data.indicator_correlations){
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

            if(data.indicator_correlations) {
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
double OutlierRVmodel::log_likelihood() const
{
    size_t N = data.N();
    const auto& y = data.get_y();
    const auto& sig = data.get_sig();
    const auto& obsi = data.get_obsi();

    double logL = 0.0;
    double logL_inlier = 0.0;
    double logL_outlier = 0.0;

    if (enforce_stability){
        int stable = is_stable();
        if (stable != 0)
            return -std::numeric_limits<double>::infinity();
    }


    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    if (studentt)
    {
        // The following code calculates the log likelihood 
        // in the case of a t-Student model
        double var, jit, var_outlier, logLi, logLo;
        for (size_t i = 0; i < N; i++)
        {
            double sig2 = sig[i] * sig[i];

            if (data._multi)
            {
                jit = jitters[obsi[i] - 1];
                var = sig2 + jit * jit;
            }
            else
                var = sig2 + extra_sigma * extra_sigma;

            logLi = std::lgamma(0.5*(nu + 1.)) - std::lgamma(0.5*nu)
                    - 0.5*log(M_PI*nu) - 0.5*log(var)
                    - 0.5*(nu + 1.)*log(1. + pow(y[i] - mu[i], 2)/var/nu);
            
            var_outlier = sig2 + outlier_sigma * outlier_sigma;
            logLo = -halflog2pi - 0.5 * log(var_outlier) - 0.5 * (pow(y[i] - outlier_background, 2) / var_outlier);

            logL_inlier += logLi;
            logL_outlier += logLo;
            logL += DNest4::logsumexp(log(Q) + logLi, log(1.0 - Q) + logLo);
        }
    }

    else
    {
        // The following code calculates the log likelihood
        // in the case of a Gaussian likelihood
        double var, jit, var_outlier, logLi, logLo;
        for (size_t i = 0; i < N; i++)
        {
            double sig2 = sig[i] * sig[i];

            if (data._multi)
            {
                jit = jitters[obsi[i] - 1];
                var = sig2 + jit * jit;
            }
            else
                var = sig2 + extra_sigma * extra_sigma;

            logLi = -halflog2pi - 0.5 * log(var) - 0.5 * (pow(y[i] - mu[i], 2) / var);

            var_outlier = sig2 + outlier_sigma * outlier_sigma;
            logLo = -halflog2pi - 0.5 * log(var_outlier) - 0.5 * (pow(y[i] - outlier_background, 2) / var_outlier);

            logL_inlier += logLi;
            logL_outlier += logLo;
            logL += DNest4::logsumexp(log(Q) + logLi, log(1.0 - Q) + logLo); 
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


void OutlierRVmodel::print(std::ostream& out) const
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
    {
        out << extra_sigma << '\t';
    }

    if(trend)
    {
        out.precision(15);
        if (degree >= 1) out << slope << '\t';
        if (degree >= 2) out << quadr << '\t';
        if (degree == 3) out << cubic << '\t';
        out.precision(8);
    }
        
    if (data._multi)
    {
        for (size_t j = 0; j < offsets.size(); j++)
        {
            out << offsets[j] << '\t';
        }
    }

    if (data.indicator_correlations)
    {
        for (int j = 0; j < data.number_indicators; j++)
        {
            out << betas[j] << '\t';
        }
    }

    if(known_object){ // KO mode!
        for (auto P: KO_P) out << P << "\t";
        for (auto K: KO_K) out << K << "\t";
        for (auto phi: KO_phi) out << phi << "\t";
        for (auto e: KO_e) out << e << "\t";
        for (auto w: KO_w) out << w << "\t";
    }

    // outlier model parameters
    out << outlier_background << '\t';
    out << outlier_sigma << '\t';
    out << Q << '\t';

    planets.print(out);

    out << staleness << '\t';

    if (studentt)
        out << nu << '\t';

    out << background;
}


string OutlierRVmodel::description() const
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
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_phi" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_ecc" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_w" + std::to_string(i) + sep;
    }

    desc += "out_m" + sep + "out_s" + sep + "Q" + sep;

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
    if (studentt)
        desc += "nu" + sep;
    
    desc += "vsys";

    return desc;
}

/**
 * Save the options of the current model in a INI file.
 * 
*/
void OutlierRVmodel::save_setup() {
	std::fstream fout("kima_model_setup.txt", std::ios::out);
    fout << std::boolalpha;

    fout << "; " << timestamp() << endl << endl;

    fout << "[kima]" << endl;

    fout << "model: " << "OutlierRVmodel" << endl << endl;
    fout << "fix: " << fix << endl;
    fout << "npmax: " << npmax << endl << endl;

    fout << "hyperpriors: " << false << endl;
    fout << "trend: " << trend << endl;
    fout << "degree: " << degree << endl;
    fout << "multi_instrument: " << data._multi << endl;
    fout << "known_object: " << known_object << endl;
    fout << "n_known_object: " << n_known_object << endl;
    fout << "studentt: " << studentt << endl;
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
        // for(int i=0; i<n_known_object; i++){
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

auto OutlierRVMODEL_DOC = R"D(
Implements a sum-of-Keplerians model where the number of Keplerians can be free.
This model assumes white, uncorrelated noise.

Args:
    fix (bool, default=True): whether the number of Keplerians should be fixed
    npmax (int, default=0): maximum number of Keplerians
    data (RVData): the RV data
)D";

class OutlierRVmodel_publicist : public OutlierRVmodel
{
    public:
        using OutlierRVmodel::fix;
        using OutlierRVmodel::npmax;
        using OutlierRVmodel::data;
        //
        using OutlierRVmodel::trend;
        using OutlierRVmodel::degree;
        using OutlierRVmodel::studentt;
        using OutlierRVmodel::known_object;
        using OutlierRVmodel::n_known_object;
        using OutlierRVmodel::star_mass;
        using OutlierRVmodel::enforce_stability;
};


NB_MODULE(OutlierRVmodel, m) {
    nb::class_<OutlierRVmodel>(m, "OutlierRVmodel")
        .def(nb::init<bool&, int&, RVData&>(), "fix"_a, "npmax"_a, "data"_a, OutlierRVMODEL_DOC)
        //
        .def_rw("fix", &OutlierRVmodel_publicist::fix, "whether the number of Keplerians is fixed")
        .def_rw("npmax", &OutlierRVmodel_publicist::npmax, "maximum number of Keplerians")
        .def_ro("data", &OutlierRVmodel_publicist::data, "the data")
        //

        .def_rw("trend", &OutlierRVmodel_publicist::trend,
                "whether the model includes a polynomial trend")
        .def_rw("degree", &OutlierRVmodel_publicist::degree,
                "degree of the polynomial trend")
        //
        .def_rw("studentt", &OutlierRVmodel_publicist::studentt,
                "use a Student-t distribution for the likelihood (instead of Gaussian)")
        //
        .def_rw("known_object", &OutlierRVmodel_publicist::known_object,
                "whether to include (better) known extra Keplerian curve(s)")
        .def_rw("n_known_object", &OutlierRVmodel_publicist::n_known_object,
                "how many known objects")
        //
        .def_rw("star_mass", &OutlierRVmodel_publicist::star_mass,
                "stellar mass [Msun]")
        //
        .def_rw("enforce_stability", &OutlierRVmodel_publicist::enforce_stability, 
                "whether to enforce AMD-stability")

        // priors
        .def_prop_rw("Cprior",
            [](OutlierRVmodel &m) { return m.Cprior; },
            [](OutlierRVmodel &m, distribution &d) { m.Cprior = d; },
            "Prior for the systemic velocity")
        .def_prop_rw("Jprior",
            [](OutlierRVmodel &m) { return m.Jprior; },
            [](OutlierRVmodel &m, distribution &d) { m.Jprior = d; },
            "Prior for the extra white noise (jitter)")
        .def_prop_rw("slope_prior",
            [](OutlierRVmodel &m) { return m.slope_prior; },
            [](OutlierRVmodel &m, distribution &d) { m.slope_prior = d; },
            "Prior for the slope")
        .def_prop_rw("quadr_prior",
            [](OutlierRVmodel &m) { return m.quadr_prior; },
            [](OutlierRVmodel &m, distribution &d) { m.quadr_prior = d; },
            "Prior for the quadratic coefficient of the trend")
        .def_prop_rw("cubic_prior",
            [](OutlierRVmodel &m) { return m.cubic_prior; },
            [](OutlierRVmodel &m, distribution &d) { m.cubic_prior = d; },
            "Prior for the cubic coefficient of the trend")
        .def_prop_rw("offsets_prior",
            [](OutlierRVmodel &m) { return m.offsets_prior; },
            [](OutlierRVmodel &m, distribution &d) { m.offsets_prior = d; },
            "Common prior for the between-instrument offsets")
        .def_prop_rw("nu_prior",
            [](OutlierRVmodel &m) { return m.nu_prior; },
            [](OutlierRVmodel &m, distribution &d) { m.nu_prior = d; },
            "Prior for the degrees of freedom of the Student-t likelihood")
        // conditional object
        .def_prop_rw("conditional",
                     [](OutlierRVmodel &m) { return m.get_conditional_prior(); },
                     [](OutlierRVmodel &m, KeplerianConditionalPrior& c) { /* does nothing */ });
}