#include "GAIAmodel.h"

using namespace std;
using namespace Eigen;
using namespace DNest4;

#define TIMING false

const double halflog2pi = 0.5*log(2.*M_PI);


void GAIAmodel::initialize_from_data(GAIAData& data)
{
    jitters.resize(data.number_instruments);
    
    // resize GAIA model vector
    mu.resize(data.N());

    // set default conditional priors that depend on data
    auto conditional = planets.get_conditional_prior();
    conditional->set_default_priors(data);
}

/* set default priors if the user didn't change them */

void GAIAmodel::setPriors()  // BUG: should be done by only one thread!
{
    auto data = get_data();
    
     if (!Jprior)
        Jprior = make_prior<ModifiedLogUniform>(0.1,100.);
    
    if (!da_prior)
        da_prior = make_prior<Gaussian>(0.0,pow(10,1));
    if (!dd_prior)
        dd_prior = make_prior<Gaussian>(0.0,pow(10,1));
    if (!mua_prior)
        mua_prior = make_prior<Gaussian>(0.0,pow(10,1));
    if (!mud_prior)
        mud_prior = make_prior<Gaussian>(0.0,pow(10,1));
    if (!plx_prior)
        plx_prior = make_prior<LogUniform>(0.01,1000.);
        
    if (known_object) { // KO mode!
        // if (n_known_object == 0) cout << "Warning: `known_object` is true, but `n_known_object` is set to 0";
        for (int i = 0; i < n_known_object; i++){
            if (!KO_Pprior[i] || !KO_aprior[i] || !KO_eprior[i] || !KO_phiprior[i] || !KO_wprior[i] || !KO_cosiprior[i] || !KO_Omprior[i])
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

    auto data = get_data();
    
    if (known_object) { // KO mode!
        KO_P.resize(n_known_object);
        KO_a.resize(n_known_object);
        KO_e.resize(n_known_object);
        KO_phi.resize(n_known_object);
        KO_w.resize(n_known_object);
        KO_cosi.resize(n_known_object);
        KO_Om.resize(n_known_object);

        for (int i=0; i<n_known_object; i++){
            KO_P[i] = KO_Pprior[i]->generate(rng);
            KO_a[i] = KO_aprior[i]->generate(rng);
            KO_e[i] = KO_eprior[i]->generate(rng);
            KO_phi[i] = KO_phiprior[i]->generate(rng);
            KO_w[i] = KO_wprior[i]->generate(rng);
            KO_cosi[i] = KO_cosiprior[i]->generate(rng);
            KO_Om[i] = KO_Omprior[i]->generate(rng);
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
    auto data = get_data();
    // Get the epochs from the data
    const vector<double>& t = data.get_t();
    const vector<double>& psi = data.get_psi();
    const vector<double>& pf = data.get_pf();

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
            mu[i] += (da + mua * (t[i]-data.M0_epoch)) * sin(psi[i]) + (dd + mud * (t[i]-data.M0_epoch)) * cos(psi[i]) + plx*pf[i];
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
    double P, a, phi, ecc, omega, Omega, cosi, Tp;
    double A, B, F, G, X, Y;
    for(size_t j=0; j<components.size(); j++)
    {
        P = components[j][0];
        phi = components[j][1];
        ecc = components[j][2];
        A = components[j][3];
        B = components[j][4];
        F = components[j][5];
        G = components[j][6];

        for(size_t i=0; i<t.size(); i++)
        {
            ti = t[i];
            
            //A = a*(cos(omega) * cos(Omega) - sin(omega) * sin(Omega) * cosi);
            //B = a*(cos(omega) * sin(Omega) - sin(omega) * cos(Omega) * cosi);
            //F = -a*(sin(omega) * cos(Omega) - cos(omega) * sin(Omega) * cosi);
            //G = -a*(sin(omega) * sin(Omega) - cos(omega) * cos(Omega) * cosi);
            
            Tp = data.M0_epoch - (P * phi) / (2. * M_PI);
            tie(X,Y) = nijenhuis::ellip_rectang(ti, P, ecc, Tp);
            
            wk = (B*X + G*Y)*sin(psi[i]) + (A*X + F*Y)*cos(psi[i]);
            mu[i] += wk;
        }
    }

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Model eval took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif

}


void GAIAmodel::remove_known_object()
{
    auto data = get_data();
    auto t = data.get_t();
    auto psi = data.get_psi();
    double wk, ti, Tp;
    double A, B, F, G, X, Y;
    // cout << "in remove_known_obj: " << KO_P[1] << endl;
    for(int j=0; j<n_known_object; j++)
    {
        for(size_t i=0; i<t.size(); i++)
        {
            ti = t[i];
            
            A = KO_a[j]*(cos(KO_w[j]) * cos(KO_Om[j]) - sin(KO_w[j]) * sin(KO_Om[j]) * KO_cosi[j]);
            B = KO_a[j]*(cos(KO_w[j]) * sin(KO_Om[j]) - sin(KO_w[j]) * cos(KO_Om[j]) * KO_cosi[j]);
            F = -KO_a[j]*(sin(KO_w[j]) * cos(KO_Om[j]) - cos(KO_w[j]) * sin(KO_Om[j]) * KO_cosi[j]);
            G = -KO_a[j]*(sin(KO_w[j]) * sin(KO_Om[j]) - cos(KO_w[j]) * cos(KO_Om[j]) * KO_cosi[j]);
            
            Tp = data.M0_epoch-(KO_P[j]*KO_phi[j])/(2.*M_PI);
            
            tie(X,Y) = nijenhuis::ellip_rectang(ti, KO_P[j], KO_e[j], Tp);
            
            wk =(B*X + G*Y)*sin(psi[i]) + (A*X + F*Y)*cos(psi[i]);
            mu[i] -= wk;
        }
    }
}


void GAIAmodel::add_known_object()
{
    auto data = get_data();
    auto t = data.get_t();
    auto psi = data.get_psi();
    double wk, ti, Tp;
    double A, B, F, G, X, Y;
    for(int j=0; j<n_known_object; j++)
    {
        for(size_t i=0; i<t.size(); i++)
        {
            ti = t[i];
            
            A = KO_a[j]*(cos(KO_w[j]) * cos(KO_Om[j]) - sin(KO_w[j]) * sin(KO_Om[j]) * KO_cosi[j]);
            B = KO_a[j]*(cos(KO_w[j]) * sin(KO_Om[j]) - sin(KO_w[j]) * cos(KO_Om[j]) * KO_cosi[j]);
            F = -KO_a[j]*(sin(KO_w[j]) * cos(KO_Om[j]) - cos(KO_w[j]) * sin(KO_Om[j]) * KO_cosi[j]);
            G = -KO_a[j]*(sin(KO_w[j]) * sin(KO_Om[j]) - cos(KO_w[j]) * cos(KO_Om[j]) * KO_cosi[j]);
            
            Tp = data.M0_epoch-(KO_P[j]*KO_phi[j])/(2.*M_PI);
            
            tie(X,Y) = nijenhuis::ellip_rectang(ti,  KO_P[j], KO_e[j], Tp);
            
            wk =(B*X + G*Y)*sin(psi[i]) + (A*X + F*Y)*cos(psi[i]);
            mu[i] += wk;
        }
    }
}

double GAIAmodel::perturb(RNG& rng)
{
    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif
    

    auto data = get_data();
    const vector<double>& t = data.get_t();
    const vector<double>& psi = data.get_psi();
    const vector<double>& pf = data.get_pf();
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
                KO_aprior[i]->perturb(KO_a[i], rng);
                KO_eprior[i]->perturb(KO_e[i], rng);
                KO_phiprior[i]->perturb(KO_phi[i], rng);
                KO_wprior[i]->perturb(KO_w[i], rng);
                KO_cosiprior[i]->perturb(KO_cosi[i], rng);
                KO_Omprior[i]->perturb(KO_Om[i], rng);
            }

            add_known_object();
        }
        
    }
    else //perturb background solution
    {
        //subtract ephemeris
        for(size_t i=0; i<mu.size(); i++)
        {
            mu[i] += -(da + mua * (t[i]-data.M0_epoch)) * sin(psi[i]) - (dd + mud * (t[i]-data.M0_epoch)) * cos(psi[i]) - plx*pf[i];
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
            mu[i] += (da + mua * (t[i]-data.M0_epoch)) * sin(psi[i]) + (dd + mud * (t[i]-data.M0_epoch)) * cos(psi[i]) + plx*pf[i];;
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
    const auto data = get_data();
    int N = data.N();
    auto w = data.get_w();
    auto wsig = data.get_wsig();

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
        for (auto a: KO_a) out << a << "\t";
        for (auto phi: KO_phi) out << phi << "\t";
        for (auto e: KO_e) out << e << "\t";
        for (auto w: KO_w) out << w << "\t";
        for (auto cosi: KO_cosi) out << cosi << "\t";
        for (auto Om: KO_Om) out << Om << "\t";
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
            desc += "KO_a" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_phi" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_ecc" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_w" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_cosi" + std::to_string(i) + sep;
        for(int i=0; i<n_known_object; i++) 
            desc += "KO_Om" + std::to_string(i) + sep;
    }

    desc += "ndim" + sep + "maxNp" + sep;

    desc += "Np" + sep;

    int maxpl = planets.get_max_num_components();
    if (maxpl > 0) {
        for(int i = 0; i < maxpl; i++) desc += "P" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "phi" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "ecc" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "A" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "B" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "F" + std::to_string(i) + sep;
        for(int i = 0; i < maxpl; i++) desc += "G" + std::to_string(i) + sep;
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
    auto data = get_data();
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

    fout << "files: ";
    for (auto f: data.datafile)
        fout << f << ",";
    fout << endl;

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
        fout << "Aprior: " << *conditional->Aprior << endl;
        fout << "Bprior: " << *conditional->Bprior << endl;
        fout << "Fprior: " << *conditional->Fprior << endl;
        fout << "Gprior: " << *conditional->Gprior << endl;
    }

    if (known_object) {
        fout << endl << "[priors.known_object]" << endl;
        for(int i=0; i<n_known_object; i++){
            fout << "Pprior_" << i << ": " << *KO_Pprior[i] << endl;
            fout << "aprior_" << i << ": " << *KO_aprior[i] << endl;
            fout << "eprior_" << i << ": " << *KO_eprior[i] << endl;
            fout << "phiprior_" << i << ": " << *KO_phiprior[i] << endl;
            fout << "wprior_" << i << ": " << *KO_wprior[i] << endl;
            fout << "cosiprior_" << i << ": " << *KO_cosiprior[i] << endl;
            fout << "Omprior_" << i << ": " << *KO_Omprior[i] << endl;
        }
    }

    fout << endl;
    fout.close();
}

