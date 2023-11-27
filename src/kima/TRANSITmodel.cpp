#include "TRANSITmodel.h"

#define TIMING false

const double halflog2pi = 0.5*log(2.*M_PI);


void TRANSITmodel::initialize_from_data(PHOTdata& data)
{
    // resize RV model vector
    mu.resize(data.N());

    // set default conditional priors that depend on data
    auto conditional = planets.get_conditional_prior();
    // conditional->set_default_priors(data);
}


/* set default priors if the user didn't change them */
void TRANSITmodel::setPriors()  // BUG: should be done by only one thread!
{
    if (!Cprior)
        Cprior = make_prior<Gaussian>(0, 0.1);

    if (!Jprior)
        Jprior = make_prior<Uniform>(0, 2 * data.get_flux_std());

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

    if (known_object) { // KO mode!
        // if (n_known_object == 0) cout << "Warning: `known_object` is true, but `n_known_object` is set to 0";
        for (int i = 0; i < n_known_object; i++){
            if (!KO_Pprior[i] || !KO_Kprior[i] || !KO_eprior[i] || !KO_phiprior[i] || !KO_wprior[i])
                throw std::logic_error("When known_object=true, please set priors for each (KO_Pprior, KO_Kprior, KO_eprior, KO_phiprior, KO_wprior)");
        }
    }

    if (studentt)
        nu_prior = make_prior<LogUniform>(2, 1000);

}


void TRANSITmodel::from_prior(RNG& rng)
{
    // preliminaries
    setPriors();
    save_setup();

    planets.from_prior(rng);
    planets.consolidate_diff();

    background = Cprior->generate(rng);
    
    extra_sigma = Jprior->generate(rng);

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

    if (studentt)
        nu = nu_prior->generate(rng);

    calculate_mu();
}

/**
 * @brief Calculate the full RV model
 * 
*/
void TRANSITmodel::calculate_mu()
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

        if (known_object) { // KO mode!
            add_known_object();
        }
    }
    else // just updating (adding) planets
        staleness++;


    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    double P, t0, Rp, a, inc, ecc, omega;
    double tmid = data.get_t_middle();
    for (size_t j = 0; j < components.size(); j++)
    {
        P = components[j][0];
        t0 = components[j][1];
        Rp = components[j][2];
        a = components[j][3];
        inc = components[j][4];
        ecc = components[j][5];
        omega = components[j][6];
        // cout << t0 << " " << P << " " << a << " " << inc << " " << ecc << " " << omega << endl;
        auto ds = rsky(data.t, tmid + t0, P, a, inc, ecc, omega);
        auto f = quadratic_ld(ds, 0.5, 0.1, Rp);
        for (size_t i = 0; i < N; i++)
            mu[i] += f[i];
    }

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Model eval took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif

}


void TRANSITmodel::remove_known_object()
{
    double f, v, ti, Tp;
    for (int j = 0; j < n_known_object; j++) {
        auto v = brandt::keplerian(data.t, KO_P[j], KO_K[j], KO_e[j], KO_w[j], KO_phi[j], data.M0_epoch);
        for (size_t i = 0; i < data.N(); i++) {
            mu[i] -= v[i];
        }
    }
}

void TRANSITmodel::add_known_object()
{
    for (int j = 0; j < n_known_object; j++) {
        auto v = brandt::keplerian(data.t, KO_P[j], KO_K[j], KO_e[j], KO_w[j], KO_phi[j], data.M0_epoch);
        for (size_t i = 0; i < data.N(); i++) {
            mu[i] += v[i];
        }
    }
}

int TRANSITmodel::is_stable() const
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


double TRANSITmodel::perturb(RNG& rng)
{
    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

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
        Jprior->perturb(extra_sigma, rng);

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
        }

        // propose new vsys
        Cprior->perturb(background, rng);

        // propose new slope
        if(trend) {
            if (degree >= 1) slope_prior->perturb(slope, rng);
            if (degree >= 2) quadr_prior->perturb(quadr, rng);
            if (degree == 3) cubic_prior->perturb(cubic, rng);
        }


        for(size_t i=0; i<mu.size(); i++)
        {
            mu[i] += background;
            if(trend) {
                mu[i] += slope * (data.t[i] - tmid) +
                            quadr * pow(data.t[i] - tmid, 2) +
                            cubic * pow(data.t[i] - tmid, 3);
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
double TRANSITmodel::log_likelihood() const
{
    size_t N = data.N();
    const auto& y = data.get_y();
    const auto& sig = data.get_sig();

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
        double var, jit;
        for(size_t i=0; i<N; i++)
        {
            var = sig[i]*sig[i] + extra_sigma*extra_sigma;

            logL += std::lgamma(0.5*(nu + 1.)) - std::lgamma(0.5*nu)
                    - 0.5*log(M_PI*nu) - 0.5*log(var)
                    - 0.5*(nu + 1.)*log(1. + pow(y[i] - mu[i], 2)/var/nu);
        }

    }

    else{
        // The following code calculates the log likelihood
        // in the case of a Gaussian likelihood
        double var, jit;
        for(size_t i=0; i<N; i++)
        {
            var = sig[i]*sig[i] + extra_sigma*extra_sigma;

            logL += - halflog2pi - 0.5*log(var)
                    - 0.5*(pow(y[i] - mu[i], 2)/var);
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


void TRANSITmodel::print(std::ostream& out) const
{
    // output precision
    out.setf(ios::fixed,ios::floatfield);
    out.precision(8);

    out << extra_sigma << '\t';

    if(trend)
    {
        out.precision(15);
        if (degree >= 1) out << slope << '\t';
        if (degree >= 2) out << quadr << '\t';
        if (degree == 3) out << cubic << '\t';
        out.precision(8);
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

    if (studentt)
        out << nu << '\t';

    out << background;
}


string TRANSITmodel::description() const
{
    string desc;
    string sep = "   ";

    desc += "jitter" + sep;

    if(trend)
    {
        if (degree >= 1) desc += "slope" + sep;
        if (degree >= 2) desc += "quadr" + sep;
        if (degree == 3) desc += "cubic" + sep;
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

    desc += "ndim" + sep + "maxNp" + sep;
    if(false) // hyperpriors
        desc += "muP" + sep + "wP" + sep + "muK";

    desc += "Np" + sep;

    int maxpl = planets.get_max_num_components();
    if (maxpl > 0) {
        for (int i = 1; i <= maxpl; i++)
            desc += "P" + std::to_string(i) + sep;
        for (int i = 1; i <= maxpl; i++)
            desc += "tc" + std::to_string(i) + sep;
        for (int i = 1; i <= maxpl; i++)
            desc += "Rp" + std::to_string(i) + sep;
        for (int i = 1; i <= maxpl; i++)
            desc += "a" + std::to_string(i) + sep;
        for (int i = 1; i <= maxpl; i++)
            desc += "inc" + std::to_string(i) + sep;
        for (int i = 1; i <= maxpl; i++)
            desc += "ecc" + std::to_string(i) + sep;
        for (int i = 1; i <= maxpl; i++)
            desc += "w" + std::to_string(i) + sep;
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
void TRANSITmodel::save_setup() {
	std::fstream fout("kima_model_setup.txt", std::ios::out);
    fout << std::boolalpha;

    time_t rawtime;
    time (&rawtime);
    fout << ";" << ctime(&rawtime) << endl;

    fout << "[kima]" << endl;

    fout << "model: " << "TRANSITmodel" << endl << endl;
    fout << "fix: " << fix << endl;
    fout << "npmax: " << npmax << endl << endl;

    fout << "hyperpriors: " << false << endl;
    fout << "trend: " << trend << endl;
    fout << "degree: " << degree << endl;
    fout << "known_object: " << known_object << endl;
    fout << "n_known_object: " << n_known_object << endl;
    fout << "studentt: " << studentt << endl;
    fout << endl;

    fout << endl;

    fout << "[data]" << endl;
    fout << "file: " << data._datafile << endl;
    fout << "units: " << data._units << endl;
    fout << "skip: " << data._skip << endl;

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
    if (studentt)
        fout << "nu_prior: " << *nu_prior << endl;

    if (planets.get_max_num_components()>0){
        auto conditional = planets.get_conditional_prior();

        fout << endl << "[priors.planets]" << endl;
        fout << "Pprior: " << *conditional->Pprior << endl;
        fout << "t0prior: " << *conditional->t0prior << endl;
        fout << "RPprior: " << *conditional->RPprior << endl;
        fout << "aprior: " << *conditional->aprior << endl;
        fout << "incprior: " << *conditional->incprior << endl;
        fout << "eprior: " << *conditional->eprior << endl;
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

auto TRANSITMODEL_DOC = R"D(
Implements a sum-of-Keplerians model where the number of Keplerians can be free.
This model assumes white, uncorrelated noise.

Args:
    fix (bool, default=True): whether the number of Keplerians should be fixed
    npmax (int, default=0): maximum number of Keplerians
    data (PHOTdata): the photometric data
)D";

class TRANSITmodel_publicist : public TRANSITmodel
{
    public:
        using TRANSITmodel::trend;
        using TRANSITmodel::degree;
        using TRANSITmodel::studentt;
        using TRANSITmodel::known_object;
        using TRANSITmodel::n_known_object;
        using TRANSITmodel::star_mass;
        using TRANSITmodel::enforce_stability;
};


NB_MODULE(TRANSITmodel, m) {
    nb::class_<TRANSITmodel>(m, "TRANSITmodel")
        .def(nb::init<bool&, int&, PHOTdata&>(), "fix"_a, "npmax"_a, "data"_a, TRANSITMODEL_DOC)
        //
        .def_rw("trend", &TRANSITmodel_publicist::trend,
                "whether the model includes a polynomial trend")
        .def_rw("degree", &TRANSITmodel_publicist::degree,
                "degree of the polynomial trend")
        //
        .def_rw("studentt", &TRANSITmodel_publicist::studentt,
                "use a Student-t distribution for the likelihood (instead of Gaussian)")
        //
        .def_rw("known_object", &TRANSITmodel_publicist::known_object,
                "whether to include (better) known extra Keplerian curve(s)")
        .def_rw("n_known_object", &TRANSITmodel_publicist::n_known_object,
                "how many known objects")
        //
        .def_rw("star_mass", &TRANSITmodel_publicist::star_mass,
                "stellar mass [Msun]")
        //
        .def_rw("enforce_stability", &TRANSITmodel_publicist::enforce_stability, 
                "whether to enforce AMD-stability")

        // priors
        .def_prop_rw("Cprior",
            [](TRANSITmodel &m) { return m.Cprior; },
            [](TRANSITmodel &m, distribution &d) { m.Cprior = d; },
            "Prior for the systemic velocity")
        .def_prop_rw("Jprior",
            [](TRANSITmodel &m) { return m.Jprior; },
            [](TRANSITmodel &m, distribution &d) { m.Jprior = d; },
            "Prior for the extra white noise (jitter)")
        .def_prop_rw("slope_prior",
            [](TRANSITmodel &m) { return m.slope_prior; },
            [](TRANSITmodel &m, distribution &d) { m.slope_prior = d; },
            "Prior for the slope")
        .def_prop_rw("quadr_prior",
            [](TRANSITmodel &m) { return m.quadr_prior; },
            [](TRANSITmodel &m, distribution &d) { m.quadr_prior = d; },
            "Prior for the quadratic coefficient of the trend")
        .def_prop_rw("cubic_prior",
            [](TRANSITmodel &m) { return m.cubic_prior; },
            [](TRANSITmodel &m, distribution &d) { m.cubic_prior = d; },
            "Prior for the cubic coefficient of the trend")
        .def_prop_rw("nu_prior",
            [](TRANSITmodel &m) { return m.nu_prior; },
            [](TRANSITmodel &m, distribution &d) { m.nu_prior = d; },
            "Prior for the degrees of freedom of the Student-t likelihood")
        // conditional object
        .def_prop_rw("conditional",
                     [](TRANSITmodel &m) { return m.get_conditional_prior(); },
                     [](TRANSITmodel &m, TRANSITConditionalPrior& c) { /* does nothing */ });
}