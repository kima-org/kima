#include "GAIAmodel.h"

using namespace std;
// using namespace Eigen;
using namespace DNest4;
using namespace nijenhuis;
using namespace brandt;

#define TIMING false

const double halflog2pi = 0.5*log(2.*M_PI);
const double daytoyear = 1/365.25;
const double daytoyear2 = 1/(365.25*365.25);
const double daytoyear3 = 1/(365.25*365.25*365.25);


void GAIAmodel::initialize_from_data(GAIAdata& data)
{   
    // resize GAIA model vector
    mu.resize(data.N());
    
    
    // set default conditional priors that depend on data
    auto conditional = planets.get_conditional_prior();
    
    conditional->set_default_priors(data);
}

void GAIAmodel::set_baseline_model(size_t n)
{
    n_baseline_params = n;
    if (n==5){
        acceleration = false;
        jerk = false;
    }
    else if (n==7){
        acceleration = true;
        jerk = false;
    }
    else if (n==9){
        acceleration = true;
        jerk = true;
    }
    else {
        throw std::logic_error("When setting a baseline solution please choose one of 5 (standard), 7 (+acceleration) or 9 (+jerk) for the number of parameters.");
    }
}

void GAIAmodel::set_scan_dep_signal(size_t n)
{
    scan_dep_signal = true;
    n_scan_dep_components = n;

    Ak.resize(n);
    thetak.resize(n);

    Ak_prior.resize(n);
    thetak_prior.resize(n);
}

void GAIAmodel::set_known_object(size_t n)
{
    known_object = true;
    n_known_object = n;

    KO_Pprior.resize(n);
    KO_aprior.resize(n);
    KO_eprior.resize(n);
    KO_phiprior.resize(n);
    KO_wprior.resize(n);
    KO_cosiprior.resize(n);
    KO_Wprior.resize(n);

    KO_P.resize(n);
    KO_a.resize(n);
    KO_e.resize(n);
    KO_phi.resize(n);
    KO_w.resize(n);
    KO_cosi.resize(n);
    KO_W.resize(n);
}

void GAIAmodel::set_transiting_planet(size_t n)
{
    transiting_planet = true;
    n_transiting_planet = n;
    // planet_perturb_prob = 0.5;
    // jitKO_perturb_prob = 0.3;

    TR_Pprior.resize(n);
    TR_aprior.resize(n);
    TR_eprior.resize(n);
    TR_Tcprior.resize(n);
    TR_wprior.resize(n);
    TR_cosiprior.resize(n);
    TR_Wprior.resize(n);

    TR_P.resize(n);
    TR_a.resize(n);
    TR_e.resize(n);
    TR_Tc.resize(n);
    TR_w.resize(n);
    TR_cosi.resize(n);
    TR_W.resize(n);
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

    if (scan_dep_signal)
    {
        for (int i = 0; i < n_scan_dep_components; i++)
        {
            if (!Ak_prior[i])
                Ak_prior[i] = make_prior<ModifiedLogUniform>(0.05,10.);
            if (!thetak_prior[i])
                thetak_prior[i] = make_prior<Uniform>(0,2.*M_PI/(i*2 + 3));
        }
    }
    // Alter default offset prior if there is an acceleration
    double dp = 0;
    if (acceleration)
        dp = 2;
    
    if (!da_prior)
        da_prior = make_prior<Gaussian>(0.0,pow(10,dp));
    if (!dd_prior)
        dd_prior = make_prior<Gaussian>(0.0,pow(10,dp));
    if (!mua_prior)
        mua_prior = make_prior<Gaussian>(0.0,pow(10,2));
    if (!mud_prior)
        mud_prior = make_prior<Gaussian>(0.0,pow(10,2));
    if (!plx_prior)
        plx_prior = make_prior<LogUniform>(1.,100.);
    if (acceleration) {
        if (!accela_prior)
            accela_prior = make_prior<Gaussian>(0.0,2);
        if (!acceld_prior)
            acceld_prior = make_prior<Gaussian>(0.0,2);
        if (jerk) {
            if (!jerka_prior)
                jerka_prior = make_prior<Gaussian>(0.0,10);
            if (!jerkd_prior)
                jerkd_prior = make_prior<Gaussian>(0.0,10);
        }
    }

    if (known_object) { // KO mode!
        // if (n_known_object == 0) cout << "Warning: `known_object` is true, but `n_known_object` is set to 0";
        for (int i = 0; i < n_known_object; i++){
            if (!KO_Pprior[i] || !KO_aprior[i] || !KO_eprior[i] || !KO_phiprior[i] || !KO_Wprior[i] || !KO_cosiprior[i] || !KO_Wprior[i])
            {
                std::string p = "KO_Pprior, KO_aprior, KO_eprior, KO_phiprior, KO_wprior, KO_cosiprior, KO_Wprior";
                std::string msg = "When known_object=true, must set priors for each of " + p;
                throw std::logic_error(msg);
            }
        }
    }

    if (transiting_planet) {
        for (size_t i = 0; i < n_transiting_planet; i++) {
            if (!TR_Pprior[i] || !TR_aprior[i] || !TR_eprior[i] || !TR_Tcprior[i] || !TR_wprior[i] || !TR_cosiprior[i] || !TR_Wprior[i])
            {
                std::string p = "TR_Pprior, TR_aprior, TR_eprior, TR_Tcprior, TR_wprior, TR_cosiprior, TR_Wprior";
                std::string msg = "When transiting_planet=true, must set priors for each of " + p;
                throw std::logic_error(msg);
            }
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

    if (scan_dep_signal)
    {
        for (int i=0; i<n_scan_dep_components; i++){
            Ak[i] = Ak_prior[i]->generate(rng);
            thetak[i] = thetak_prior[i]->generate(rng);
        }
    }
    
    da = da_prior->generate(rng);
    dd = dd_prior->generate(rng);
    mua = mua_prior->generate(rng);
    mud = mud_prior->generate(rng);
    plx = plx_prior->generate(rng);

    if (acceleration){
        accela = accela_prior->generate(rng);
        acceld = acceld_prior->generate(rng);
        if (jerk){
            jerka = jerka_prior->generate(rng);
            jerkd = jerkd_prior->generate(rng);
        }
    }

    
    if (known_object) { // KO mode!

        for (int i=0; i<n_known_object; i++){
            KO_P[i] = KO_Pprior[i]->generate(rng);
            KO_a[i] = KO_aprior[i]->generate(rng);
            KO_e[i] = KO_eprior[i]->generate(rng);
            KO_phi[i] = KO_phiprior[i]->generate(rng);
            KO_w[i] = KO_wprior[i]->generate(rng);
            KO_cosi[i] = KO_cosiprior[i]->generate(rng);
            KO_W[i] = KO_Wprior[i]->generate(rng);
        }
    }

    if (transiting_planet) {
        for (int i = 0; i < n_transiting_planet; i++)
        {
            TR_P[i] = TR_Pprior[i]->generate(rng);
            TR_a[i] = TR_aprior[i]->generate(rng);
            TR_e[i] = TR_eprior[i]->generate(rng);
            TR_Tc[i] = TR_Tcprior[i]->generate(rng);
            TR_w[i] = TR_wprior[i]->generate(rng);
            TR_cosi[i] = TR_cosiprior[i]->generate(rng);
            TR_W[i] = TR_Wprior[i]->generate(rng);
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
            mu[i] += (da + mua * daytoyear * (data.t[i]-data.M0_epoch)) * sin(data.psi[i]) + (dd + mud * daytoyear * (data.t[i]-data.M0_epoch)) * cos(data.psi[i]) + plx*data.pf[i];
            if (acceleration){
                mu[i] += (accela * daytoyear2 * pow((data.t[i]-data.M0_epoch),2) / 2) * sin(data.psi[i]) + (acceld * daytoyear2 * pow((data.t[i]-data.M0_epoch),2) / 2) * cos(data.psi[i]);
                if (jerk){
                    mu[i] += (jerka * daytoyear3 * pow((data.t[i]-data.M0_epoch),3) / 6) * sin(data.psi[i]) + (jerkd * daytoyear3 * pow((data.t[i]-data.M0_epoch),3) / 6) * cos(data.psi[i]);
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


    double P, a0, phi, ecc, omega, Omega, cosi;
    double A, B, F, G; // X, Y;
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
        
        auto wk = brandt::keplerian_gaia(data.t, data.psi, A, B, F, G, ecc, P, phi, data.M0_epoch);
        for(size_t i=0; i<N; i++)
            mu[i] += wk[i];
    }

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Model eval took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif

}


void GAIAmodel::remove_known_object()
{
    size_t N = data.N();
    double wk, ti, Tp;
    double A, B, F, G, X, Y;
    for(int j=0; j<n_known_object; j++)
    {
        A = KO_a[j]*(cos(KO_w[j]) * cos(KO_W[j]) - sin(KO_w[j]) * sin(KO_W[j]) * KO_cosi[j]);
        B = KO_a[j]*(cos(KO_w[j]) * sin(KO_W[j]) + sin(KO_w[j]) * cos(KO_W[j]) * KO_cosi[j]);
        F = -KO_a[j]*(sin(KO_w[j]) * cos(KO_W[j]) + cos(KO_w[j]) * sin(KO_W[j]) * KO_cosi[j]);
        G = -KO_a[j]*(sin(KO_w[j]) * sin(KO_W[j]) - cos(KO_w[j]) * cos(KO_W[j]) * KO_cosi[j]);
        
        auto wk = brandt::keplerian_gaia(data.t,data.psi, A, B, F, G, KO_e[j], KO_P[j], KO_phi[j], data.M0_epoch);
        for (size_t i = 0; i < N; i++)
            mu[i] -= wk[i];
    }
}


void GAIAmodel::add_known_object()
{
    size_t N = data.N();
    double wk, ti, Tp;
    double A, B, F, G, X, Y;
    for (int j = 0; j < n_known_object; j++)
    {
        A = KO_a[j]*(cos(KO_w[j]) * cos(KO_W[j]) - sin(KO_w[j]) * sin(KO_W[j]) * KO_cosi[j]);
        B = KO_a[j]*(cos(KO_w[j]) * sin(KO_W[j]) + sin(KO_w[j]) * cos(KO_W[j]) * KO_cosi[j]);
        F = -KO_a[j]*(sin(KO_w[j]) * cos(KO_W[j]) + cos(KO_w[j]) * sin(KO_W[j]) * KO_cosi[j]);
        G = -KO_a[j]*(sin(KO_w[j]) * sin(KO_W[j]) - cos(KO_w[j]) * cos(KO_W[j]) * KO_cosi[j]);
        
        auto wk = brandt::keplerian_gaia(data.t,data.psi, A, B, F, G, KO_e[j], KO_P[j], KO_phi[j], data.M0_epoch);
        for (size_t i = 0; i < N; i++)
            mu[i] += wk[i];
    }
}


void GAIAmodel::remove_transiting_planet()
{
    size_t N = data.N();
    const double twopi = 2.0 * M_PI;
    for (int j = 0; j < n_transiting_planet; j++) {
        double w = TR_w[j], e = TR_e[j], P = TR_P[j], Tc = TR_Tc[j];

        double A = TR_a[j]*(cos(TR_w[j]) * cos(TR_W[j]) - sin(TR_w[j]) * sin(TR_W[j]) * TR_cosi[j]);
        double B = TR_a[j]*(cos(TR_w[j]) * sin(TR_W[j]) + sin(TR_w[j]) * cos(TR_W[j]) * TR_cosi[j]);
        double F = -TR_a[j]*(sin(TR_w[j]) * cos(TR_W[j]) + cos(TR_w[j]) * sin(TR_W[j]) * TR_cosi[j]);
        double G = -TR_a[j]*(sin(TR_w[j]) * sin(TR_W[j]) - cos(TR_w[j]) * cos(TR_W[j]) * TR_cosi[j]);
        
        // Tc -> phi
        double f_c = M_PI / 2.0 - w;
        double E_c = 2.0 * atan2(sqrt(1.0 - e) * sin(f_c/2.0),
                                 sqrt(1.0 + e) * cos(f_c/2.0));
        double M_c = E_c - e * sin(E_c);
        double phi = M_c + twopi * (data.M0_epoch - Tc) / P;
        phi = fmod(phi, twopi);
        if (phi < 0) phi += twopi;

        auto wk = brandt::keplerian_gaia(data.t, data.psi, A, B, F, G, e, P, phi, data.M0_epoch);
        for (size_t i = 0; i < N; i++)
            mu[i] -= wk[i];
    }
}

void GAIAmodel::add_transiting_planet()
{
    size_t N = data.N();
    const double twopi = 2.0 * M_PI;
    for (int j = 0; j < n_transiting_planet; j++) {
        double w = TR_w[j], e = TR_e[j], P = TR_P[j], Tc = TR_Tc[j];

        double A = TR_a[j]*(cos(TR_w[j]) * cos(TR_W[j]) - sin(TR_w[j]) * sin(TR_W[j]) * TR_cosi[j]);
        double B = TR_a[j]*(cos(TR_w[j]) * sin(TR_W[j]) + sin(TR_w[j]) * cos(TR_W[j]) * TR_cosi[j]);
        double F = -TR_a[j]*(sin(TR_w[j]) * cos(TR_W[j]) + cos(TR_w[j]) * sin(TR_W[j]) * TR_cosi[j]);
        double G = -TR_a[j]*(sin(TR_w[j]) * sin(TR_W[j]) - cos(TR_w[j]) * cos(TR_W[j]) * TR_cosi[j]);
        
        // Tc -> phi
        double f_c = M_PI / 2.0 - w;
        double E_c = 2.0 * atan2(sqrt(1.0 - e) * sin(f_c/2.0),
                                 sqrt(1.0 + e) * cos(f_c/2.0));
        double M_c = E_c - e * sin(E_c);
        double phi = M_c + twopi * (data.M0_epoch - Tc) / P;
        phi = fmod(phi, twopi);
        if (phi < 0) phi += twopi;

        auto wk = brandt::keplerian_gaia(data.t, data.psi, A, B, F, G, e, P, phi, data.M0_epoch);
        for (size_t i = 0; i < N; i++)
            mu[i] += wk[i];
    }
}


double GAIAmodel::perturb(RNG& rng)
{
    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif
    

    double logH = 0.;
    double prob_add = 0;
    if (known_object){
        prob_add += 0.2;
    }
    if (scan_dep_signal){
        prob_add +=0.2;
    }

    if(rng.rand() <= 0.5 && npmax > 0) // perturb planet parameters
    {
        logH += planets.perturb(rng);
        planets.consolidate_diff();
        calculate_mu();
    }
    else if(rng.rand() <= 0.2 + prob_add) // perturb jitter(s) + known_object
    {
        
        Jprior->perturb(jitter, rng);
        
        if (studentt)
            nu_prior->perturb(nu, rng);


        if (known_object)
        {
            remove_known_object();

            for (int i=0; i<n_known_object; i++){
                logH += KO_Pprior[i]->perturb(KO_P[i], rng);
                logH += KO_aprior[i]->perturb(KO_a[i], rng);
                logH += KO_eprior[i]->perturb(KO_e[i], rng);
                logH += KO_phiprior[i]->perturb(KO_phi[i], rng);
                logH += KO_wprior[i]->perturb(KO_w[i], rng);
                logH += KO_cosiprior[i]->perturb(KO_cosi[i], rng);
                logH += KO_Wprior[i]->perturb(KO_W[i], rng);
            }

            add_known_object();
        }

        if (transiting_planet)
        {
            remove_transiting_planet();

            for (int i = 0; i < n_transiting_planet; i++)
            {
                logH += TR_Pprior[i]->perturb(TR_P[i], rng);
                logH += TR_aprior[i]->perturb(TR_a[i], rng);
                logH += TR_eprior[i]->perturb(TR_e[i], rng);
                logH += TR_Tcprior[i]->perturb(TR_Tc[i], rng);
                logH += TR_wprior[i]->perturb(TR_w[i], rng);
                logH += TR_cosiprior[i]->perturb(TR_cosi[i], rng);
                logH += TR_Wprior[i]->perturb(TR_W[i], rng);
            }

            add_transiting_planet();
        }

    }
    else //perturb baseline solution
    {
        // propose new parameters
        logH += da_prior->perturb(da, rng);
        logH += dd_prior->perturb(dd, rng);
        logH += mua_prior->perturb(mua, rng);
        logH += mud_prior->perturb(mud, rng);
        logH += plx_prior->perturb(plx, rng);

        if (acceleration)
        {
            logH += accela_prior->perturb(accela, rng);
            logH += acceld_prior->perturb(acceld, rng);
            if (jerk)
            {
                logH += jerka_prior->perturb(jerka, rng);
                logH += jerkd_prior->perturb(jerkd, rng);
            }
        }

        if (scan_dep_signal) {
            for (int j = 0; j < n_scan_dep_components; j++)
            {
                logH += Ak_prior[j]->perturb(Ak[j], rng);
                logH += thetak_prior[j]->perturb(thetak[j], rng);
            }
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
        double c_nu = std::lgamma(0.5*(nu + 1.)) - std::lgamma(0.5*nu) - 0.5*log(M_PI*nu);
        for(size_t i=0; i<N; i++)
        {
            var = wsig[i]*wsig[i] +jit*jit;

            logL += c_nu - 0.5*log(var) - 0.5*(nu + 1.)*log(1. + pow(w[i] - mu[i], 2)/var/nu);
        }

    }

    else{
        // The following code calculates the log likelihood
        // in the case of a Gaussian likelihood
        double var;
        for(size_t i=0; i<N; i++)
        {
            var = wsig[i]*wsig[i] + jit*jit;

            logL += - halflog2pi - 0.5*log(var) - 0.5*(pow(w[i] - mu[i], 2)/var);
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

    if (acceleration){
        out << accela << '\t';
        out << acceld << '\t';
        if (jerk){
            out << jerka << '\t';
            out << jerkd << '\t';
        }
    }
    
    out.precision(8);

    //auto data = get_data();
    if (scan_dep_signal){
        for (auto A: Ak) out << A << "\t";
        for (auto theta: thetak) out << theta << "\t";
    }

    if(known_object){ // KO mode!
        for (auto P: KO_P) out << P << "\t";
        for (auto phi: KO_phi) out << phi << "\t";
        for (auto e: KO_e) out << e << "\t";
        for (auto a: KO_a) out << a << "\t";
        for (auto w: KO_w) out << w << "\t";
        for (auto cosi: KO_cosi) out << cosi << "\t";
        for (auto Om: KO_W) out << Om << "\t";
    }

    if(transiting_planet){
        for (auto P: TR_P)   out << P  << "\t";
        for (auto a: TR_a)   out << a  << "\t";
        for (auto Tc: TR_Tc) out << Tc << "\t";
        for (auto e: TR_e)   out << e  << "\t";
        for (auto w: TR_w)   out << w  << "\t";
        for (auto cosi: TR_cosi) out << cosi << "\t";
        for (auto Om: TR_W) out << Om << "\t";
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

    if (acceleration){
        desc += "accela" + sep;
        desc += "acceld" + sep;
        if (jerk){
            desc += "jerka" + sep;
            desc += "jerkd" + sep;
        }
    }

    //auto data = get_data();
    if (scan_dep_signal){
        for (int i=0; i<n_scan_dep_components; i++)
            desc += "A"+std::to_string(i*2 + 3) + sep;
        for (int i=0; i<n_scan_dep_components; i++)
            desc += "theta"+std::to_string(i*2 + 3) + sep;
    }

    auto printi = [&](const size_t n, const string& name )
    {
        for(size_t i = 0; i < n; i++) 
        {
            desc += name + std::to_string(i + index_from) + sep;
        }
    };

    if(known_object) { // KO mode!
        printi(n_known_object, "KO_P");
        printi(n_known_object, "KO_phi");
        printi(n_known_object, "KO_ecc");
        printi(n_known_object, "KO_a");
        printi(n_known_object, "KO_w");
        printi(n_known_object, "KO_cosi");
        printi(n_known_object, "KO_W");
    }

    if(transiting_planet) {
        printi(n_transiting_planet, "TR_P");
        printi(n_transiting_planet, "TR_a");
        printi(n_transiting_planet, "TR_Tc");
        printi(n_transiting_planet, "TR_ecc");
        printi(n_transiting_planet, "TR_w");
        printi(n_transiting_planet, "TR_cosi");
        printi(n_transiting_planet, "TR_W");
    }

    desc += "ndim" + sep + "maxNp" + sep;

    desc += "Np" + sep;

    int maxpl = planets.get_max_num_components();
    if (maxpl > 0) {
        printi(maxpl, "P");
        printi(maxpl, "phi");
        printi(maxpl, "ecc");
        if(thiele_innes)
        {
            printi(maxpl, "A");
            printi(maxpl, "B");
            printi(maxpl, "F");
            printi(maxpl, "G");
        }
        else
        {
            printi(maxpl, "a");
            printi(maxpl, "w");
            printi(maxpl, "cosi");
            printi(maxpl, "W");
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
    fout << std::boolalpha << std::fixed;
    fout.precision(15);

    fout << "; " << timestamp() << endl << endl;

    fout << "[kima]" << endl;

    fout << "model: " << "GAIAmodel" << endl << endl;
    fout << "fix: " << fix << endl;
    fout << "npmax: " << npmax << endl << endl;

    fout << "known_object: " << known_object << endl;
    fout << "n_known_object: " << n_known_object << endl;
    fout << "transiting_planet: " << transiting_planet << endl;
    fout << "n_transiting_planet: " << n_transiting_planet << endl;

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

    fout << "M0_epoch: " << data.M0_epoch << endl;

    fout << endl;

    fout << "[priors.general]" << endl;
    fout << "Jprior: " << *Jprior << endl;

    fout << "da_prior: " << *da_prior << endl;
    fout << "dd_prior: " << *dd_prior << endl;
    fout << "mua_prior: " << *mua_prior << endl;
    fout << "mud_prior: " << *mud_prior << endl;
    fout << "parallax_prior: " << *plx_prior << endl;

    if (acceleration){
        fout << "accela_prior: " << *accela_prior << endl;
        fout << "acceld_prior: " << *acceld_prior << endl;
        if (jerk){
            fout << "jerka _prior: " << *jerka_prior << endl;
            fout << "jerkd _prior: " << *jerkd_prior << endl;
        }
    }

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
            fout << "aprior: " << *conditional->aprior << endl;
            fout << "wprior: " << *conditional->wprior << endl;
            fout << "cosiprior: " << *conditional->cosiprior << endl;
            fout << "Wprior: " << *conditional->Wprior << endl;
        }
    }

    if (scan_dep_signal){
        fout << endl << "[priors.scan_dep_signal]" << endl;
        for(int i=0; i<n_scan_dep_components; i++){
            fout << "Ak_prior_" << i << ": " << *Ak_prior[i] << endl;
            fout << "thetak_prior_" << i << ": " << *thetak_prior[i] << endl;
        }
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
            fout << "Wprior_" << i << ": " << *KO_Wprior[i] << endl;
        }
    }

    if (transiting_planet) {
        fout << endl << "[priors.transiting_planet]" << endl;
        for (int i = 0; i < n_transiting_planet; i++)
        {
            fout << "Pprior_" << i << ": " << *TR_Pprior[i] << endl;
            fout << "aprior_" << i << ": " << *TR_aprior[i] << endl;
            fout << "eprior_" << i << ": " << *TR_eprior[i] << endl;
            fout << "Tcprior_" << i << ": " << *TR_Tcprior[i] << endl;
            fout << "wprior_" << i << ": " << *TR_wprior[i] << endl;
            fout << "cosiprior_" << i << ": " << *TR_cosiprior[i] << endl;
            fout << "Wprior_" << i << ": " << *TR_Wprior[i] << endl;
        }
    }

    fout << endl;
    fout.close();
}

using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;

auto GAIAMODEL_DOC = R"D(
Analysis of Gaia epoch astrometry. Implements a sum-of-Keplerians model where the number of Keplerians can be free.
This model assumes white, uncorrelated noise. Known objects are given priors for geometric elements, free planet search 
has the choice of geometric or Thiele-Innes elements. An option to fit for a scan-angle dependent signal is included.

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
        using GAIAmodel::index_from;
        using GAIAmodel::fix;
        using GAIAmodel::npmax;
        using GAIAmodel::data;
        using GAIAmodel::star_mass;
        using GAIAmodel::RA;
        using GAIAmodel::DEC;
//         using GAIAmodel::enforce_stability;
//         using GAIAmodel::known_object;
//         using GAIAmodel::n_known_object;
        using GAIAmodel::thiele_innes;
};


NB_MODULE(GAIAmodel, m) {
    // bind ConditionalPrior so it can be returned
    bind_GAIAConditionalPrior(m);

    nb::class_<GAIAmodel>(m, "GAIAmodel")
        .def(nb::init<bool&, int&, GAIAdata&>(), "fix"_a, "npmax"_a, "data"_a, GAIAMODEL_DOC)
        //
        .def_rw("fix", &GAIAmodel_publicist::fix,
                "whether the number of Keplerians is fixed")
        .def_rw("npmax", &GAIAmodel_publicist::npmax,
                "maximum number of Keplerians")
        .def_ro("data", &GAIAmodel_publicist::data,
                "the data")

        .def_rw("studentt", &GAIAmodel_publicist::studentt,
                "use a Student-t distribution for the likelihood (instead of Gaussian)")
        //
        .def_rw("index_from", &GAIAmodel_publicist::index_from,
                "what indexing convention to use for the labelling of keplerians, defaults to 1.")
        //
        .def_rw("thiele_innes", &GAIAmodel_publicist::thiele_innes, 
                "use the thiele-innes coefficients rather than geometric")
        .def_rw("star_mass", &GAIAmodel_publicist::star_mass,
                "the mass of the central star (Msun)")
        .def_rw("RA", &GAIAmodel_publicist::RA,
                "Right Ascension of the target star (degrees)")
        .def_rw("DEC", &GAIAmodel_publicist::DEC,
                "Declination of the target star (degrees)")

        .def("set_scan_dep_signal", &GAIAmodel::set_scan_dep_signal,
                "set whether the model includes a model for potential scan-angle dependent signals")
        .def_prop_ro("scan_dep_signal", [](GAIAmodel &m) { return m.get_scan_dep_signal(); },
                     "whether the model includes a model for potential scan-angle dependent signals that could bias towards certain frequencies")
        .def_prop_ro("n_scan_dep_components", [](GAIAmodel &m) { return m.get_n_scan_dep_components(); },
                     "how many components of scan-angle harmonics are included")

        .def("set_baseline_model", &GAIAmodel::set_baseline_model,
                "set the number of parameters for the baseline astrometric solution, either 5 (the default astrometric solution), 7, or 9 (which include acceleration and jerk terms)")
        .def_prop_ro("n_baseline_params", [](GAIAmodel &m) { return m.get_n_baseline_params(); },
                "how many baseline astrometric parameters are included the model")
//         //KO mode
//         .def_rw("known_object", &GAIAmodel_publicist::known_object,
//                 "whether to include (better) known extra Keplerian curve(s)")
//         .def_rw("n_known_object", &GAIAmodel_publicist::n_known_object,
//                 "how many known objects")
        // KO mode
        .def("set_known_object", &GAIAmodel::set_known_object,
            "set how many known objects to include")
        .def_prop_ro("known_object", [](GAIAmodel &m) { return m.get_known_object(); },
                     "whether the model includes (better) known extra Keplerian curve(s)")
        .def_prop_ro("n_known_object", [](GAIAmodel &m) { return m.get_n_known_object(); },
                     "how many known objects")

        // transiting planets
        .def("set_transiting_planet", &GAIAmodel::set_transiting_planet)
        .def_prop_ro("transiting_planet", [](GAIAmodel &m) { return m.get_transiting_planet(); },
                     "whether the model includes transiting planet(s)")
        .def_prop_ro("n_transiting_planet", [](GAIAmodel &m) { return m.get_n_transiting_planet(); },
                     "how many transiting planets")


        // priors
        .def_prop_rw("Jprior",
            [](GAIAmodel &m) { return m.Jprior; },
            [](GAIAmodel &m, distribution &d) { m.Jprior = d; },
            "Prior for the extra white noise (jitter)")
        .def_prop_rw("nu_prior",
            [](GAIAmodel &m) { return m.nu_prior; },
            [](GAIAmodel &m, distribution &d) { m.nu_prior = d; },
            "Prior for the degrees of freedom of the Student-t likelihood")
        .def_prop_rw("Ak_prior",
            [](GAIAmodel &m) { return m.Ak_prior; },
            [](GAIAmodel &m, std::vector<distribution>& vd) { m.Ak_prior = vd; },
            "Prior for the amplitudes of scan-angle dependent signals")
        .def_prop_rw("thetak_prior",
            [](GAIAmodel &m) { return m.thetak_prior; },
            [](GAIAmodel &m, std::vector<distribution>& vd) { m.thetak_prior = vd; },
            "Prior for the phase of scan-angle dependent signals")
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
            "Prior for the proper-motion in right-ascension (mas/yr)")
        .def_prop_rw("mud_prior",
            [](GAIAmodel &m) { return m.mud_prior; },
            [](GAIAmodel &m, distribution &d) { m.mud_prior = d; },
            "Prior for the proper-motion in declination (mas/yr)")
        .def_prop_rw("parallax_prior",
            [](GAIAmodel &m) { return m.plx_prior; },
            [](GAIAmodel &m, distribution &d) { m.plx_prior = d; },
            "Prior for the parallax")
        .def_prop_rw("accela_prior",
            [](GAIAmodel &m) { return m.accela_prior; },
            [](GAIAmodel &m, distribution &d) { m.accela_prior = d; },
            "Prior for the proper-acceleration in right-ascension (mas/yr^2)")
        .def_prop_rw("acceld_prior",
            [](GAIAmodel &m) { return m.acceld_prior; },
            [](GAIAmodel &m, distribution &d) { m.acceld_prior = d; },
            "Prior for the proper-acceleration in declination (mas/yr^2)")
        .def_prop_rw("jerka_prior",
            [](GAIAmodel &m) { return m.jerka_prior; },
            [](GAIAmodel &m, distribution &d) { m.jerka_prior = d; },
            "Prior for the proper-jerk in right-ascension (mas/yr^3)")
        .def_prop_rw("jerkd_prior",
            [](GAIAmodel &m) { return m.jerkd_prior; },
            [](GAIAmodel &m, distribution &d) { m.jerkd_prior = d; },
            "Prior for the proper-jerk in declination (mas/yr^3)")    

        // known object priors
        // ? should these setters check if known_object is true?
        .def_prop_rw("KO_Pprior",
                     [](GAIAmodel &m) { return m.KO_Pprior; },
                     [](GAIAmodel &m, std::vector<distribution>& vd) { m.KO_Pprior = vd; },
                     "Prior for KO orbital period(s)")
        .def_prop_rw("KO_aprior",
                     [](GAIAmodel &m) { return m.KO_aprior; },
                     [](GAIAmodel &m, std::vector<distribution>& vd) { m.KO_aprior = vd; },
                     "Prior for KO photocentre semi-major-axis(es)")
        .def_prop_rw("KO_eprior",
                     [](GAIAmodel &m) { return m.KO_eprior; },
                     [](GAIAmodel &m, std::vector<distribution>& vd) { m.KO_eprior = vd; },
                     "Prior for KO eccentricity(ies)")
        .def_prop_rw("KO_wprior",
                     [](GAIAmodel &m) { return m.KO_wprior; },
                     [](GAIAmodel &m, std::vector<distribution>& vd) { m.KO_wprior = vd; },
                     "Prior for KO argument(s) of periastron")
        .def_prop_rw("KO_phiprior",
                     [](GAIAmodel &m) { return m.KO_phiprior; },
                     [](GAIAmodel &m, std::vector<distribution>& vd) { m.KO_phiprior = vd; },
                     "Prior for KO mean anomaly(ies)")
        .def_prop_rw("KO_cosiprior",
                     [](GAIAmodel &m) { return m.KO_cosiprior; },
                     [](GAIAmodel &m, std::vector<distribution>& vd) { m.KO_cosiprior = vd; },
                     "Prior for cosine of KO inclination(s)")
        .def_prop_rw("KO_Wprior",
                     [](GAIAmodel &m) { return m.KO_Wprior; },
                     [](GAIAmodel &m, std::vector<distribution>& vd) { m.KO_Wprior = vd; },
                     "Prior for KO longitude(s) of ascending node")


        // conditional object
        .def_prop_rw("conditional",
                     [](GAIAmodel &m) { return m.get_conditional_prior(); },
                     [](GAIAmodel &m, GAIAConditionalPrior& c) { /* does nothing */ });
}


