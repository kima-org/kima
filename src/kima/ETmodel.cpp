#include "ETmodel.h"

#define TIMING false

const double halflog2pi = 0.5*log(2.*M_PI);

void ETmodel::initialize_from_data(ETData& data)
{

    // resize RV model vector
    mu.resize(data.N());

    // set default conditional priors that depend on data
    auto conditional = planets.get_conditional_prior();
    conditional->set_default_priors(data);
}

/* set default priors if the user didn't change them */

void ETmodel::setPriors()  // BUG: should be done by only one thread!
{
    
     if (!Jprior)
        Jprior = make_prior<ModifiedLogUniform>(0.1,100.);
    
    if (ephemeris >= 4)
        throw std::logic_error("can't go higher than cubic ephemeris ");
    if (ephemeris < 1)
        throw std::logic_error("ephemeris should be at least one since eclipse needs a period");
    if (ephemeris >= 1 && !ephem1_prior){
        //ephem1_prior = make_prior<Gaussian>(0.0,10.);
        ephem1_prior =  make_prior<LogUniform>(0.0001,100);
        printf("# No prior on Binary period specified, taken as LogUniform over 0.0001-100\n");
    }
    if (ephemeris >= 2 && !ephem2_prior)
        ephem2_prior = make_prior<Gaussian>(0.0,pow(10,-10.));
    if (ephemeris >= 3 && !ephem3_prior)
        ephem3_prior = make_prior<Gaussian>(0.0,pow(10,-12.));
        
    if (known_object) { // KO mode!
        for (size_t i = 0; i < n_known_object; i++)
        {
            if (!KO_Pprior[i] || !KO_Kprior[i] || !KO_eprior[i] || !KO_phiprior[i] || !KO_wprior[i])
            {
                std::string msg = "When known_object=true, must set priors for each of KO_Pprior, KO_Kprior, KO_eprior, KO_phiprior, KO_wprior";
                throw std::logic_error(msg);
            }
        }
    }

    if (studentt)
        nu_prior = make_prior<LogUniform>(2, 1000);

}


void ETmodel::from_prior(RNG& rng)
{
    // preliminaries
    setPriors();
    save_setup();

    planets.from_prior(rng);
    planets.consolidate_diff();
    
    jitter = Jprior->generate(rng);
    
    
    if (ephemeris >= 1) ephem1 = ephem1_prior->generate(rng);
    if (ephemeris >= 2) ephem2 = ephem2_prior->generate(rng);
    if (ephemeris == 3) ephem3 = ephem3_prior->generate(rng);

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
 * @brief Calculate the full ET model
 * 
*/
void ETmodel::calculate_mu()
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
        mu.assign(mu.size(), data.M0_epoch);
        
        staleness = 0;
        
        for(size_t i=0; i<mu.size(); i++)
        {
            mu[i] += ephem1*data.epochs[i]+ 0.5*ephem2*ephem1*pow(data.epochs[i],2.0) + ephem3*pow(ephem1,2.0)*pow(data.epochs[i],3.0)/6.0;
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


    auto epochs = data.get_epochs();
    double f, tau;
    double P, K, phi, ecc, omega, Tp;
    for(size_t j=0; j<components.size(); j++)
    {
        P = components[j][0];
        K = components[j][1];
        phi = components[j][2];
        ecc = components[j][3];
        omega = components[j][4];
        
        auto tau = brandt::keplerian_et(epochs, P, K, ecc, omega, phi, ephem1);
        for(size_t i=0; i<N; i++)
            mu[i] += tau[i]/(24*3600);
    }

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Model eval took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif

}


void ETmodel::remove_known_object()
{
    auto epochs = data.get_epochs();
    double f, tau, ti, Tp;
    // cout << "in remove_known_obj: " << KO_P[1] << endl;
    for(int j=0; j<n_known_object; j++)
    {
        auto tau = brandt::keplerian(epochs, KO_P[j], KO_K[j], KO_e[j], KO_w[j], KO_phi[j], ephem1);
        for (size_t i = 0; i < data.N(); i++) {
            mu[i] -= tau[i]/(24*3600);
        }
    }
}


void ETmodel::add_known_object()
{
    auto epochs = data.get_epochs();
    double f, tau, ti, Tp;
    std::vector<double> ts;
    for(int j=0; j<n_known_object; j++)
    {
        auto tau = brandt::keplerian(epochs, KO_P[j], KO_K[j], KO_e[j], KO_w[j], KO_phi[j], ephem1);
        for (size_t i = 0; i < data.N(); i++) {
            mu[i] += tau[i]/(24*3600);
        }
    }
}


double ETmodel::perturb(RNG& rng)
{
    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif


    const vector<double>& epochs = data.get_epochs();
    double logH = 0.;

    if(rng.rand() <= 0.75) // perturb planet parameters
    {
        logH += planets.perturb(rng);
        planets.consolidate_diff();
        calculate_mu();
    }
    else if(rng.rand() <= 0.5) // perturb jitter(s) + known_object
    {
        
        Jprior->perturb(jitter, rng);
        
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
        //subtract ephemeris
        for(size_t i=0; i<mu.size(); i++)
        {
            mu[i] += -data.M0_epoch - ephem1*data.epochs[i]- 0.5*ephem2*ephem1*pow(data.epochs[i],2.0) - ephem3*pow(ephem1,2.0)*pow(data.epochs[i],3.0)/6.0;
        }
        
        // propose new ephemeris
        if (ephemeris >= 1) ephem1_prior->perturb(ephem1, rng);
        if (ephemeris >= 2) ephem2_prior->perturb(ephem2, rng);
        if (ephemeris == 3) ephem3_prior->perturb(ephem3, rng);

        //add ephemeris back in
        for(size_t i=0; i<mu.size(); i++)
        {
            mu[i] += data.M0_epoch + ephem1*data.epochs[i]+ 0.5*ephem2*ephem1*pow(data.epochs[i],2.0) + ephem3*pow(ephem1,2.0)*pow(data.epochs[i],3.0)/6.0;
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
double ETmodel::log_likelihood() const
{
    size_t N = data.N();
    const auto& et = data.get_et();
    const auto& etsig = data.get_etsig();

    double logL = 0.;


    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif
    
    double jit = jitter/(24*3600);

    if (studentt){
        // The following code calculates the log likelihood 
        // in the case of a t-Student model
        double var;
        for(size_t i=0; i<N; i++)
        {
            var = etsig[i]*etsig[i] +jit*jit;

            logL += std::lgamma(0.5*(nu + 1.)) - std::lgamma(0.5*nu)
                    - 0.5*log(M_PI*nu) - 0.5*log(var)
                    - 0.5*(nu + 1.)*log(1. + pow(et[i] - mu[i], 2)/var/nu);
        }

    }

    else{
        // The following code calculates the log likelihood
        // in the case of a Gaussian likelihood
        double var;
        for(size_t i=0; i<N; i++)
        {
            var = etsig[i]*etsig[i] + jit*jit;

            logL += - halflog2pi - 0.5*log(var)
                    - 0.5*(pow(et[i] - mu[i], 2)/var);
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


void ETmodel::print(std::ostream& out) const
{
    // output precision
    out.setf(ios::fixed,ios::floatfield);
    out.precision(8);

    out<<jitter<<'\t';


    out.precision(24);
    if (ephemeris >= 1) out << ephem1 << '\t';
    if (ephemeris >= 2) out << ephem2 << '\t';
    if (ephemeris == 3) out << ephem3 << '\t';
    out.precision(8);

    //auto data = get_data();

    if(known_object){ // KO mode!
        for (auto P: KO_P) out << P << "\t";
        for (auto K: KO_K) out << K << "\t";
        for (auto phi: KO_phi) out << phi << "\t";
        for (auto e: KO_e) out << e << "\t";
        for (auto w: KO_w) out << w << "\t";
    }

    planets.print(out);

    out << ' ' << staleness << ' ';

    if (studentt)
        out << '\t' << nu << '\t';

}


string ETmodel::description() const
{
    string desc;
    string sep = "   ";

    desc += "jitter   ";

    if (ephemeris >= 1) desc += "ephem1" + sep;
    if (ephemeris >= 2) desc += "ephem2" + sep;
    if (ephemeris == 3) desc += "ephem3" + sep;

    //auto data = get_data();

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
    
    return desc;
}

/**
 * Save the options of the current model in a INI file.
 * 
*/
void ETmodel::save_setup() {

    std::fstream fout("kima_model_setup.txt", std::ios::out);
    fout << std::boolalpha;

    time_t rawtime;
    time (&rawtime);
    fout << ";" << ctime(&rawtime) << endl;

    fout << "[kima]" << endl;

    fout << "model: " << "ETmodel" << endl << endl;
    fout << "fix: " << fix << endl;
    fout << "npmax: " << npmax << endl << endl;

    fout << "ephemeris: " << ephemeris << endl;
    fout << "known_object: " << known_object << endl;
    fout << "n_known_object: " << n_known_object << endl;
    fout << "studentt: " << studentt << endl;
    fout << endl;

    fout << endl;

    fout << "[data]" << endl;
    fout << "file: " << data._datafile << endl;
    fout << "skip: " << data._skip << endl;


    fout.precision(15);
    fout << "M0_epoch: " << data.M0_epoch << endl;
    fout.precision(6);

    fout << endl;

    fout << "[priors.general]" << endl;
    fout << "Jprior: " << *Jprior << endl;

    if (ephemeris >= 1) fout << "ephem1_prior: " << *ephem1_prior << endl;
    if (ephemeris >= 2) fout << "ephem2_prior: " << *ephem2_prior << endl;
    if (ephemeris == 3) fout << "ephem3_prior: " << *ephem3_prior << endl;

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
            fout << "eprior_" << i << ": " << *KO_eprior[i] << endl;
            fout << "phiprior_" << i << ": " << *KO_phiprior[i] << endl;
            fout << "wprior_" << i << ": " << *KO_wprior[i] << endl;
        }
    }

    fout << endl;
    fout.close();
}


using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;

auto ETMODEL_DOC = R"D(
Implements a sum-of-Keplerians model for eclipse timing data where the number of Keplerians can be free. These are light-travel time ETVs.

Args:
    fix (bool):
        whether the number of Keplerians should be fixed
    npmax (int):
        maximum number of Keplerians
    data (ETData):
        the Eclipse timing data
)D";

class ETmodel_publicist : public ETmodel
{
    public:
        using ETmodel::fix;
        using ETmodel::npmax;
        using ETmodel::ephemeris;
        using ETmodel::data;
        //
        using ETmodel::studentt;
        using ETmodel::star_mass;
};


NB_MODULE(ETmodel, m) {
    // bind ETconditionalPrior so it can be returned
    bind_ETConditionalPrior(m);

    nb::class_<ETmodel>(m, "ETmodel", "")
        .def(nb::init<bool&, int&, ETData&>(), "fix"_a, "npmax"_a, "data"_a,  
             ETMODEL_DOC
        )
        //
        .def_rw("fix", &ETmodel_publicist::fix,
                "whether the number of Keplerians is fixed")
        .def_rw("npmax", &ETmodel_publicist::npmax,
                "maximum number of Keplerians")
        .def_ro("data", &ETmodel_publicist::data,
                "the data")

        //
        .def_rw("ephemeris", &ETmodel_publicist::ephemeris,
                "order of the ephemeris used")

        .def_rw("studentt", &ETmodel_publicist::studentt,
                "use a Student-t distribution for the likelihood (instead of Gaussian)")

        // KO mode
        .def_prop_ro("known_object", [](ETmodel &m) { return m.get_known_object(); },
                     "whether the model includes (better) known extra Keplerian curve(s)")
        .def_prop_ro("n_known_object", [](ETmodel &m) { return m.get_n_known_object(); },
                     "how many known objects")

        //
        .def_rw("star_mass", &ETmodel_publicist::star_mass,
                "stellar mass [Msun]")


        // priors
        .def_prop_rw("Jprior",
            [](ETmodel &m) { return m.Jprior; },
            [](ETmodel &m, distribution &d) { m.Jprior = d; },
            "Prior for the extra white noise (jitter)")

        .def_prop_rw("ephem1_prior",
            [](ETmodel &m) { return m.ephem1_prior; },
            [](ETmodel &m, distribution &d) { m.ephem1_prior = d; },
            "Prior for the linear ephemeris")
        .def_prop_rw("ephem2_prior",
            [](ETmodel &m) { return m.ephem2_prior; },
            [](ETmodel &m, distribution &d) { m.ephem2_prior = d; },
            "Prior for the quadratic term of the ephemeris")
        .def_prop_rw("ephem3_prior",
            [](ETmodel &m) { return m.ephem3_prior; },
            [](ETmodel &m, distribution &d) { m.ephem3_prior = d; },
            "Prior for the cubic term of the ephemeris")

        .def_prop_rw("nu_prior",
            [](ETmodel &m) { return m.nu_prior; },
            [](ETmodel &m, distribution &d) { m.nu_prior = d; },
            "Prior for the degrees of freedom of the Student-t likelihood")

        // known object priors
        // ? should these setters check if known_object is true?
        .def_prop_rw("KO_Pprior",
                     [](ETmodel &m) { return m.KO_Pprior; },
                     [](ETmodel &m, std::vector<distribution>& vd) { m.KO_Pprior = vd; },
                     "Prior for KO orbital period")
        .def_prop_rw("KO_Kprior",
                     [](ETmodel &m) { return m.KO_Kprior; },
                     [](ETmodel &m, std::vector<distribution>& vd) { m.KO_Kprior = vd; },
                     "Prior for KO semi-amplitude")
        .def_prop_rw("KO_eprior",
                     [](ETmodel &m) { return m.KO_eprior; },
                     [](ETmodel &m, std::vector<distribution>& vd) { m.KO_eprior = vd; },
                     "Prior for KO eccentricity")
        .def_prop_rw("KO_wprior",
                     [](ETmodel &m) { return m.KO_wprior; },
                     [](ETmodel &m, std::vector<distribution>& vd) { m.KO_wprior = vd; },
                     "Prior for KO argument of periastron")
        .def_prop_rw("KO_phiprior",
                     [](ETmodel &m) { return m.KO_phiprior; },
                     [](ETmodel &m, std::vector<distribution>& vd) { m.KO_phiprior = vd; },
                     "Prior for KO mean anomaly(ies)")

        // conditional object
        .def_prop_rw("conditional",
                     [](ETmodel &m) { return m.get_conditional_prior(); },
                     [](ETmodel &m, RVConditionalPrior& c) { /* does nothing */ });
}
