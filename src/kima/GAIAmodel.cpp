#include "GAIAmodel.h"

using namespace std;
// using namespace Eigen;
using namespace DNest4;
using namespace nijenhuis;
using namespace brandt;

#define TIMING false

const double halflog2pi = 0.5*log(2.*M_PI);


void GAIAmodel::initialize_from_data(GAIAData& data)
{   
    // resize GAIA model vector
    mu.resize(data.N());
    
    
    // set default conditional priors that depend on data
    auto conditional = planets.get_conditional_prior();
    
    conditional->set_default_priors(data);
}

void GAIAmodel::set_known_object(size_t n)
{
    known_object = true;
    n_known_object = n;

    KO_Pprior.resize(n);
    KO_a0prior.resize(n);
    KO_eprior.resize(n);
    KO_phiprior.resize(n);
    KO_omegaprior.resize(n);
    KO_cosiprior.resize(n);
    KO_Omegaprior.resize(n);
}

/* set default priors if the user didn't change them */

void GAIAmodel::setPriors()  // BUG: should be done by only one thread!
{   
    if (thiele_innes)
    {
        auto conditional = planets.get_conditional_prior();
        conditional->use_thiele_innes();
    }
     
    if (!Jprior)
        Jprior = make_prior<ModifiedLogUniform>(0.01,10.);
    
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
            if (!KO_Pprior[i] || !KO_a0prior[i] || !KO_eprior[i] || !KO_phiprior[i] || !KO_omegaprior[i] || !KO_cosiprior[i] || !KO_Omegaprior[i])
                throw std::logic_error("When known_object=true, please set priors for each (KO_Pprior, KO_Kprior, KO_eprior, KO_phiprior, KO_wprior, KO_cosiprior, KO_Omprior)");
        }
    }

    if (studentt)
        nu_prior = make_prior<LogUniform>(2, 1000);

}


void GAIAmodel::from_prior(RNG& rng)
{
    // preliminaries
    setPriors();
    save_setup();

    planets.from_prior(rng);
    planets.consolidate_diff();
    
    jitter = Jprior->generate(rng);
    
    da = da_prior->generate(rng);
    dd = dd_prior->generate(rng);
    mua = mua_prior->generate(rng);
    mud = mud_prior->generate(rng);
    plx = plx_prior->generate(rng);

    
    if (known_object) { // KO mode!
        KO_P.resize(n_known_object);
        KO_a0.resize(n_known_object);
        KO_e.resize(n_known_object);
        KO_phi.resize(n_known_object);
        KO_omega.resize(n_known_object);
        KO_cosi.resize(n_known_object);
        KO_Omega.resize(n_known_object);

        for (int i=0; i<n_known_object; i++){
            KO_P[i] = KO_Pprior[i]->generate(rng);
            KO_a0[i] = KO_a0prior[i]->generate(rng);
            KO_e[i] = KO_eprior[i]->generate(rng);
            KO_phi[i] = KO_phiprior[i]->generate(rng);
            KO_omega[i] = KO_omegaprior[i]->generate(rng);
            KO_cosi[i] = KO_cosiprior[i]->generate(rng);
            KO_Omega[i] = KO_Omegaprior[i]->generate(rng);
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
void GAIAmodel::calculate_mu()
{

    // Get the epochs from the data
    size_t N = data.N();
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
        mu.assign(mu.size(), 0.);
        
        staleness = 0;
        
        for(size_t i=0; i<mu.size(); i++)
        {
            mu[i] += (da + mua * (data.t[i]-data.M0_epoch)) * sin(data.psi[i]) + (dd + mud * (data.t[i]-data.M0_epoch)) * cos(data.psi[i]) + plx*data.pf[i];
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
    double P, a0, phi, ecc, omega, Omega, cosi, Tp;
    double A, B, F, G, X, Y;
    for(size_t j=0; j<components.size(); j++)
    {
        P = components[j][0];
        phi = components[j][1];
        ecc = components[j][2];
        if(thiele_innes)
        {
            A = components[j][3];
            B = components[j][4];
            F = components[j][5];
            G = components[j][6];
        }
        else
        {
            a0 = components[j][3];
            omega = components[j][4];
            cosi = components[j][5];
            Omega = components[j][6];
            
            A = a0*(cos(omega) * cos(Omega) - sin(omega) * sin(Omega) * cosi);
            B = a0*(cos(omega) * sin(Omega) + sin(omega) * cos(Omega) * cosi);
            F = -a0*(sin(omega) * cos(Omega) + cos(omega) * sin(Omega) * cosi);
            G = -a0*(sin(omega) * sin(Omega) - cos(omega) * cos(Omega) * cosi);
        }
        
        auto wk = brandt::keplerian_gaia(data.t,data.psi, A, B, F, G, ecc, P, phi, data.M0_epoch);
        for(size_t i=0; i<N; i++)
            mu[i] += wk[i];

//         for(size_t i=0; i<N; i++)
//         {
//             ti = data.t[i];
//             
//             if(!thiele_innes)
//             {
//                 A = a0*(cos(omega) * cos(Omega) - sin(omega) * sin(Omega) * cosi);
//                 B = a0*(cos(omega) * sin(Omega) - sin(omega) * cos(Omega) * cosi);
//                 F = -a0*(sin(omega) * cos(Omega) - cos(omega) * sin(Omega) * cosi);
//                 G = -a0*(sin(omega) * sin(Omega) - cos(omega) * cos(Omega) * cosi);
//             }
//             
//             Tp = data.M0_epoch - (P * phi) / (2. * M_PI);
//             tie(X,Y) = nijenhuis::ellip_rectang(ti, P, ecc, Tp);
//             
//             wk = (B*X + G*Y)*sin(data.psi[i]) + (A*X + F*Y)*cos(data.psi[i]);
//             mu[i] += wk;
//         }
    }

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Model eval took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif

}


void GAIAmodel::remove_known_object()
{
    double wk, ti, Tp;
    double A, B, F, G, X, Y;
    // cout << "in remove_known_obj: " << KO_P[1] << endl;
    for(int j=0; j<n_known_object; j++)
    {
        
        
        
        
        for(size_t i=0; i<data.N(); i++)
        {
            ti = data.t[i];
            
            A = KO_a0[j]*(cos(KO_omega[j]) * cos(KO_Omega[j]) - sin(KO_omega[j]) * sin(KO_Omega[j]) * KO_cosi[j]);
            B = KO_a0[j]*(cos(KO_omega[j]) * sin(KO_Omega[j]) - sin(KO_omega[j]) * cos(KO_Omega[j]) * KO_cosi[j]);
            F = -KO_a0[j]*(sin(KO_omega[j]) * cos(KO_Omega[j]) - cos(KO_omega[j]) * sin(KO_Omega[j]) * KO_cosi[j]);
            G = -KO_a0[j]*(sin(KO_omega[j]) * sin(KO_Omega[j]) - cos(KO_omega[j]) * cos(KO_Omega[j]) * KO_cosi[j]);
            
            Tp = data.M0_epoch-(KO_P[j]*KO_phi[j])/(2.*M_PI);
            
            tie(X,Y) = nijenhuis::ellip_rectang(ti, KO_P[j], KO_e[j], Tp);
            
            wk =(B*X + G*Y)*sin(data.psi[i]) + (A*X + F*Y)*cos(data.psi[i]);
            mu[i] -= wk;
        }
    }
}


void GAIAmodel::add_known_object()
{
    double wk, ti, Tp;
    double A, B, F, G, X, Y;
    for(int j=0; j<n_known_object; j++)
    {
        for(size_t i=0; i<data.N(); i++)
        {
            ti = data.t[i];
            
            A = KO_a0[j]*(cos(KO_omega[j]) * cos(KO_Omega[j]) - sin(KO_omega[j]) * sin(KO_Omega[j]) * KO_cosi[j]);
            B = KO_a0[j]*(cos(KO_omega[j]) * sin(KO_Omega[j]) - sin(KO_omega[j]) * cos(KO_Omega[j]) * KO_cosi[j]);
            F = -KO_a0[j]*(sin(KO_omega[j]) * cos(KO_Omega[j]) - cos(KO_omega[j]) * sin(KO_Omega[j]) * KO_cosi[j]);
            G = -KO_a0[j]*(sin(KO_omega[j]) * sin(KO_Omega[j]) - cos(KO_omega[j]) * cos(KO_Omega[j]) * KO_cosi[j]);
            
            Tp = data.M0_epoch-(KO_P[j]*KO_phi[j])/(2.*M_PI);
            
            tie(X,Y) = nijenhuis::ellip_rectang(ti,  KO_P[j], KO_e[j], Tp);
            
            wk =(B*X + G*Y)*sin(data.psi[i]) + (A*X + F*Y)*cos(data.psi[i]);
            mu[i] += wk;
        }
    }
}

double GAIAmodel::perturb(RNG& rng)
{
    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif
    

    double logH = 0.;

    if(rng.rand() <= 0.75 && npmax > 0) // perturb planet parameters
    {
        logH += planets.perturb(rng);
        planets.consolidate_diff();
        calculate_mu();
    }
    else if(rng.rand() <= 0.4) // perturb jitter(s) + known_object
    {
        
        Jprior->perturb(jitter, rng);
        
        if (studentt)
            nu_prior->perturb(nu, rng);


        if (known_object)
        {
            remove_known_object();

            for (int i=0; i<n_known_object; i++){
                KO_Pprior[i]->perturb(KO_P[i], rng);
                KO_a0prior[i]->perturb(KO_a0[i], rng);
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
        //subtract ephemeris
        for(size_t i=0; i<mu.size(); i++)
        {
            mu[i] += -(da + mua * (data.t[i]-data.M0_epoch)) * sin(data.psi[i]) - (dd + mud * (data.t[i]-data.M0_epoch)) * cos(data.psi[i]) - plx*data.pf[i];
        }
        // propose new parameters
        da_prior->perturb(da, rng);
        dd_prior->perturb(dd, rng);
        mua_prior->perturb(mua, rng);
        mud_prior->perturb(mud, rng);
        plx_prior->perturb(plx, rng);

        //add ephemeris back in
        for(size_t i=0; i<mu.size(); i++)
        {
            mu[i] += (da + mua * (data.t[i]-data.M0_epoch)) * sin(data.psi[i]) + (dd + mud * (data.t[i]-data.M0_epoch)) * cos(data.psi[i]) + plx*data.pf[i];;
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
double GAIAmodel::log_likelihood() const
{
    size_t N = data.N();
    const auto& w = data.get_w();
    const auto& wsig = data.get_wsig();

    double logL = 0.;


    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif
    
    double jit = jitter;

    if (studentt){
        // The following code calculates the log likelihood 
        // in the case of a t-Student model
        double var;
        for(size_t i=0; i<N; i++)
        {
            var = wsig[i]*wsig[i] +jit*jit;

            logL += std::lgamma(0.5*(nu + 1.)) - std::lgamma(0.5*nu)
                    - 0.5*log(M_PI*nu) - 0.5*log(var)
                    - 0.5*(nu + 1.)*log(1. + pow(w[i] - mu[i], 2)/var/nu);
        }

    }

    else{
        // The following code calculates the log likelihood
        // in the case of a Gaussian likelihood
        double var;
        for(size_t i=0; i<N; i++)
        {
            var = wsig[i]*wsig[i] + jit*jit;

            logL += - halflog2pi - 0.5*log(var)
                    - 0.5*(pow(w[i] - mu[i], 2)/var);
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


void GAIAmodel::print(std::ostream& out) const
{
    // output precision
    out.setf(ios::fixed,ios::floatfield);
    out.precision(8);

    out<<jitter<<'\t';


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
        for (auto a: KO_a0) out << a << "\t";
        for (auto phi: KO_phi) out << phi << "\t";
        for (auto e: KO_e) out << e << "\t";
        for (auto w: KO_omega) out << w << "\t";
        for (auto cosi: KO_cosi) out << cosi << "\t";
        for (auto Om: KO_Omega) out << Om << "\t";
    }

    planets.print(out);

    out << ' ' << staleness << ' ';

    if (studentt)
        out << '\t' << nu << '\t';

}


string GAIAmodel::description() const
{
    string desc;
    string sep = "   ";

    desc += "jitter   ";

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
            desc += "KO_a0" + std::to_string(i) + sep;
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
        if(thiele_innes)
        {
            for(int i = 0; i < maxpl; i++) desc += "A" + std::to_string(i) + sep;
            for(int i = 0; i < maxpl; i++) desc += "B" + std::to_string(i) + sep;
            for(int i = 0; i < maxpl; i++) desc += "F" + std::to_string(i) + sep;
            for(int i = 0; i < maxpl; i++) desc += "G" + std::to_string(i) + sep;
        }
        else
        {
            for(int i = 0; i < maxpl; i++) desc += "a0" + std::to_string(i) + sep;
            for(int i = 0; i < maxpl; i++) desc += "w" + std::to_string(i) + sep;
            for(int i = 0; i < maxpl; i++) desc += "cosi" + std::to_string(i) + sep;
            for(int i = 0; i < maxpl; i++) desc += "W" + std::to_string(i) + sep;
        }
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
void GAIAmodel::save_setup() {
    std::fstream fout("kima_model_setup.txt", std::ios::out);
    fout << std::boolalpha;

    time_t rawtime;
    time (&rawtime);
    fout << ";" << ctime(&rawtime) << endl;

    fout << "[kima]" << endl;

    fout << "model: " << "GAIAmodel" << endl << endl;
    fout << "fix: " << fix << endl;
    fout << "npmax: " << npmax << endl << endl;

    fout << "known_object: " << known_object << endl;
    fout << "n_known_object: " << n_known_object << endl;
    fout << "studentt: " << studentt << endl;
    fout << endl;

    fout << endl;

    fout << "[data]" << endl;
    fout << "file: " << data._datafile << endl;
    fout << "skip: " << data._skip << endl;
    fout << "units: " << data._units << endl;

    // fout << "files: ";
    // for (auto f: data._datafile)
    //     fout << f << ",";
    // fout << endl;

    fout.precision(15);
    fout << "M0_epoch: " << data.M0_epoch << endl;
    fout.precision(6);

    fout << endl;

    fout << "[priors.general]" << endl;
    fout << "Jprior: " << *Jprior << endl;

    fout << "da_prior: " << *da_prior << endl;
    fout << "dd_prior: " << *dd_prior << endl;
    fout << "mua_prior: " << *mua_prior << endl;
    fout << "mud_prior: " << *mud_prior << endl;
    fout << "parallax_prior: " << *plx_prior << endl;

    if (studentt)
        fout << "nu_prior: " << *nu_prior << endl;

    if (planets.get_max_num_components()>0){
        auto conditional = planets.get_conditional_prior();

        fout << endl << "[priors.planets]" << endl;
        fout << "Pprior: " << *conditional->Pprior << endl;
        fout << "phiprior: " << *conditional->phiprior << endl;
        fout << "eprior: " << *conditional->eprior << endl;
        if(thiele_innes)
        {
            fout << "Aprior: " << *conditional->Aprior << endl;
            fout << "Bprior: " << *conditional->Bprior << endl;
            fout << "Fprior: " << *conditional->Fprior << endl;
            fout << "Gprior: " << *conditional->Gprior << endl;
//             fout << "Gprior: " << *conditional->Gprior << endl;
        }
        else
        {
            fout << "a0prior: " << *conditional->a0prior << endl;
            fout << "omegaprior: " << *conditional->omegaprior << endl;
            fout << "cosiprior: " << *conditional->cosiprior << endl;
            fout << "Omegaprior: " << *conditional->Omegaprior << endl;
        }
    }

    if (known_object) {
        fout << endl << "[priors.known_object]" << endl;
        for(int i=0; i<n_known_object; i++){
            fout << "Pprior_" << i << ": " << *KO_Pprior[i] << endl;
            fout << "aprior_" << i << ": " << *KO_a0prior[i] << endl;
            fout << "eprior_" << i << ": " << *KO_eprior[i] << endl;
            fout << "phiprior_" << i << ": " << *KO_phiprior[i] << endl;
            fout << "wprior_" << i << ": " << *KO_omegaprior[i] << endl;
            fout << "cosiprior_" << i << ": " << *KO_cosiprior[i] << endl;
            fout << "Omprior_" << i << ": " << *KO_Omegaprior[i] << endl;
        }
    }

    fout << endl;
    fout.close();
}

using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;

auto GAIAMODEL_DOC = R"D(
Analysis of Gaia epoch astrometry. Implements a sum-of-Keplerians model where the number of Keplerians can be free.
This model assumes white, uncorrelated noise. Known objects are given priors for geometric elements, free planet search 
has the choice of geometric or Thiele-Innes elements.

Args:
    fix (bool, default=True):
        whether the number of Keplerians should be fixed
    npmax (int, default=0):
        maximum number of Keplerians
    data (GAIAdata):
        the astrometric data
)D";

class GAIAmodel_publicist : public GAIAmodel
{
    public:
        using GAIAmodel::studentt;
//         using GAIAmodel::star_mass;
//         using GAIAmodel::enforce_stability;
//         using GAIAmodel::known_object;
//         using GAIAmodel::n_known_object;
        using GAIAmodel::thiele_innes;
};


NB_MODULE(GAIAmodel, m) {
    // bind RVConditionalPrior so it can be returned
    bind_GAIAConditionalPrior(m);

    nb::class_<GAIAmodel>(m, "GAIAmodel")
        .def(nb::init<bool&, int&, GAIAData&>(), "fix"_a, "npmax"_a, "data"_a, GAIAMODEL_DOC)
        //

        .def_rw("studentt", &GAIAmodel_publicist::studentt,
                "use a Student-t distribution for the likelihood (instead of Gaussian)")
        .def_rw("thiele_innes", &GAIAmodel_publicist::thiele_innes, 
                "use the thiele-innes coefficients rather than geometric")

//         //KO mode
//         .def_rw("known_object", &GAIAmodel_publicist::known_object,
//                 "whether to include (better) known extra Keplerian curve(s)")
//         .def_rw("n_known_object", &GAIAmodel_publicist::n_known_object,
//                 "how many known objects")
        // KO mode
        .def("set_known_object", &GAIAmodel::set_known_object)
        .def_prop_ro("known_object", [](GAIAmodel &m) { return m.get_known_object(); },
                     "whether the model includes (better) known extra Keplerian curve(s)")
        .def_prop_ro("n_known_object", [](GAIAmodel &m) { return m.get_n_known_object(); },
                     "how many known objects")


        // priors
        .def_prop_rw("Jprior",
            [](GAIAmodel &m) { return m.Jprior; },
            [](GAIAmodel &m, distribution &d) { m.Jprior = d; },
            "Prior for the extra white noise (jitter)")
        .def_prop_rw("nu_prior",
            [](GAIAmodel &m) { return m.nu_prior; },
            [](GAIAmodel &m, distribution &d) { m.nu_prior = d; },
            "Prior for the degrees of freedom of the Student-t likelihood")
        .def_prop_rw("da_prior",
            [](GAIAmodel &m) { return m.da_prior; },
            [](GAIAmodel &m, distribution &d) { m.da_prior = d; },
            "Prior for the offset in right-ascension (mas)")
        .def_prop_rw("dd_prior",
            [](GAIAmodel &m) { return m.dd_prior; },
            [](GAIAmodel &m, distribution &d) { m.dd_prior = d; },
            "Prior for the the offset in declination (mas)")
        .def_prop_rw("mua_prior",
            [](GAIAmodel &m) { return m.mua_prior; },
            [](GAIAmodel &m, distribution &d) { m.mua_prior = d; },
            "Prior for the proper-motion in right-ascension")
        .def_prop_rw("mud_prior",
            [](GAIAmodel &m) { return m.mud_prior; },
            [](GAIAmodel &m, distribution &d) { m.mud_prior = d; },
            "Prior for the proper-motion in declination")
        .def_prop_rw("parallax_prior",
            [](GAIAmodel &m) { return m.plx_prior; },
            [](GAIAmodel &m, distribution &d) { m.plx_prior = d; },
            "Prior for the parallax")

        // known object priors
        // ? should these setters check if known_object is true?
        .def_prop_rw("KO_Pprior",
                     [](GAIAmodel &m) { return m.KO_Pprior; },
                     [](GAIAmodel &m, std::vector<distribution>& vd) { m.KO_Pprior = vd; },
                     "Prior for KO orbital period(s)")
        .def_prop_rw("KO_a0prior",
                     [](GAIAmodel &m) { return m.KO_a0prior; },
                     [](GAIAmodel &m, std::vector<distribution>& vd) { m.KO_a0prior = vd; },
                     "Prior for KO photocentre semi-major-axis(es)")
        .def_prop_rw("KO_eprior",
                     [](GAIAmodel &m) { return m.KO_eprior; },
                     [](GAIAmodel &m, std::vector<distribution>& vd) { m.KO_eprior = vd; },
                     "Prior for KO eccentricity(ies)")
        .def_prop_rw("KO_omegaprior",
                     [](GAIAmodel &m) { return m.KO_omegaprior; },
                     [](GAIAmodel &m, std::vector<distribution>& vd) { m.KO_omegaprior = vd; },
                     "Prior for KO argument(s) of periastron")
        .def_prop_rw("KO_phiprior",
                     [](GAIAmodel &m) { return m.KO_phiprior; },
                     [](GAIAmodel &m, std::vector<distribution>& vd) { m.KO_phiprior = vd; },
                     "Prior for KO mean anomaly(ies)")
        .def_prop_rw("KO_cosiprior",
                     [](GAIAmodel &m) { return m.KO_cosiprior; },
                     [](GAIAmodel &m, std::vector<distribution>& vd) { m.KO_cosiprior = vd; },
                     "Prior for cosine of KO inclination(s)")
        .def_prop_rw("KO_Omegaprior",
                     [](GAIAmodel &m) { return m.KO_Omegaprior; },
                     [](GAIAmodel &m, std::vector<distribution>& vd) { m.KO_Omegaprior = vd; },
                     "Prior for KO longitude(s) of ascending node")


        // conditional object
        .def_prop_rw("conditional",
                     [](GAIAmodel &m) { return m.get_conditional_prior(); },
                     [](GAIAmodel &m, GAIAConditionalPrior& c) { /* does nothing */ });
}


