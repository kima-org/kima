#include "TRANSITmodel.h"

#define TIMING false
#define CHECK_WINDOWS false  // debug: compare windowed vs full transit model

const double halflog2pi = 0.5*log(2.*M_PI);


void TRANSITmodel::initialize_from_data(PHOTdata& data)
{
    mu.resize(data.N());
    mu_bkg.resize(data.N());
    t_mid = data.get_t_middle();  // cached: it is O(N) to compute

    t_sorted = std::is_sorted(data.get_t().begin(), data.get_t().end());
    if (!t_sorted)
        std::cerr << "kima: TRANSITmodel: times are not sorted, "
                  << "transit windows disabled (the model will be slower)" << std::endl;

    // set default conditional priors that depend on data
    auto conditional = planets.get_conditional_prior();
    conditional->set_default_priors(data);
}

void TRANSITmodel::get_ld(double& u1, double& u2) const
{
    if (kipping_ld) {
        const double sq = std::sqrt(ld1);
        u1 = 2.0 * sq * ld2;
        u2 = sq * (1.0 - 2.0 * ld2);
    } else {
        u1 = ld1;
        u2 = ld2;
    }
}

/// physical limb darkening: u1 > 0, u1 + u2 < 1, u1 + 2 u2 > 0
/// (always true for the Kipping parametrisation with q1, q2 in [0, 1])
bool TRANSITmodel::ld_is_valid() const
{
    double u1, u2;
    get_ld(u1, u2);
    return (u1 > 0.0) && (u1 + u2 < 1.0) && (u1 + 2.0 * u2 > 0.0);
}

/// a/Rs = (G rho P^2 / 3 pi)^(1/3), from Kepler's third law (any eccentricity,
/// planet mass neglected). rho in g/cm^3, P in days.
double TRANSITmodel::a_over_rs(const std::vector<double>& c) const
{
    if (!use_stellar_density)
        return c[3];
    constexpr double G_cgs = 6.6743e-8;  // cm^3 g^-1 s^-2
    const double P = c[0] * 86400.0;
    return std::cbrt(G_cgs * rho * P * P / (3.0 * M_PI));
}

/* set default priors if the user didn't change them */
void TRANSITmodel::setPriors()  // BUG: should be done by only one thread!
{
    if (!Cprior)
        Cprior = make_prior<Gaussian>(data.get_flux_mean(), data.get_flux_std());

    if (!Jprior)
        Jprior = make_prior<Uniform>(0, data.get_flux_std());

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

    if (!ld1_prior)
        ld1_prior = make_prior<Uniform>(0.0, 1.0);
    if (!ld2_prior)
        ld2_prior = make_prior<Uniform>(0.0, 1.0);

    if (studentt && !nu_prior)
        nu_prior = make_prior<LogUniform>(2, 1000);

    if (use_stellar_density) {
        if (!rho_prior)
            rho_prior = make_prior<LogUniform>(0.1, 40.0);  // g/cm^3, broad
        // a/Rs is derived, so "fix" it in each planet's parameters
        planets.get_conditional_prior()->aprior = make_prior<Fixed>(1.0);
    }

}


void TRANSITmodel::from_prior(RNG& rng)
{
    // preliminaries
    setPriors();
    save_setup();

    planets.from_prior(rng);

    background = Cprior->generate(rng);
    jitter = Jprior->generate(rng);

    if(trend)
    {
        if (degree >= 1) slope = slope_prior->generate(rng);
        if (degree >= 2) quadr = quadr_prior->generate(rng);
        if (degree == 3) cubic = cubic_prior->generate(rng);
    }

    // limb darkening (common to all planets)
    do {
        ld1 = ld1_prior->generate(rng);
        ld2 = ld2_prior->generate(rng);
    } while (!ld_is_valid());

    if (use_stellar_density)
        rho = rho_prior->generate(rng);

    if (studentt)
        nu = nu_prior->generate(rng);

    calculate_mu(true);
    calculate_mu_bkg();
}

/// Any point where the planet overlaps the stellar disk has d < 1 + Rp. Since
/// d >= r |cos(w + f)| and r >= a(1 - e), this requires
///     |f - f_transit| < asin((1 + Rp) / (a (1 - e)))
/// and the mean anomaly changes at most by (1-e^2)^(3/2) / (1-e)^2 per unit f,
/// which gives a (conservative) bound on the time from mid-transit.
bool TRANSITmodel::transit_halfwidth(double P, double Rp, double a, double e, double& half) const
{
    const double x = (1.0 + Rp) / (a * (1.0 - e));
    if (!(x < 1.0))
        return false;
    const double s = std::sqrt(1.0 - e * e);
    half = window_margin * P / (2.0 * M_PI) * std::asin(x) * (s * s * s) / ((1.0 - e) * (1.0 - e));
    return half < 0.5 * P;
}

void TRANSITmodel::add_transit(const std::vector<double>& c, double sign)
{
    // c = {P, t0, Rp, a, inc, ecc, omega}
    double u1, u2;
    get_ld(u1, u2);

    const auto& t = data.get_t();
    const double P = c[0], t0 = c[1], Rp = c[2], a = a_over_rs(c);
    const double inc = c[4], ecc = c[5], omega = c[6];

    double half;
    if (use_transit_windows && t_sorted && !t.empty() &&
        transit_halfwidth(P, Rp, a, ecc, half))
    {
        // gather the points inside the windows around each mid-transit
        // (half < P/2, so windows never overlap)
        std::vector<size_t> idx;
        std::vector<double> tw;

        const long n0 = static_cast<long>(std::ceil((t.front() - half - t0) / P));
        const long n1 = static_cast<long>(std::floor((t.back() + half - t0) / P));
        for (long n = n0; n <= n1; n++)
        {
            const double tc = t0 + n * P;
            auto lo = std::lower_bound(t.begin(), t.end(), tc - half);
            auto hi = std::upper_bound(lo, t.end(), tc + half);
            for (auto it = lo; it != hi; ++it) {
                idx.push_back(static_cast<size_t>(it - t.begin()));
                tw.push_back(*it);
            }
        }

        if (!tw.empty()) {
            auto ds = rsky_new(tw, t0, P, a, inc, ecc, omega);
            auto f = quadratic_ld(ds, u1, u2, Rp);
            for (size_t k = 0; k < idx.size(); k++)
                mu[idx[k]] += sign * (f[k] - 1.0);
        }

        #if CHECK_WINDOWS
        {
            auto ds_full = rsky_new(t, t0, P, a, inc, ecc, omega);
            auto f_full = quadratic_ld(ds_full, u1, u2, Rp);
            std::vector<double> f_win(t.size(), 1.0);
            if (!tw.empty()) {
                auto ds = rsky_new(tw, t0, P, a, inc, ecc, omega);
                auto f = quadratic_ld(ds, u1, u2, Rp);
                for (size_t k = 0; k < idx.size(); k++) f_win[idx[k]] = f[k];
            }
            double maxdiff = 0.0;
            for (size_t i = 0; i < t.size(); i++)
                maxdiff = std::max(maxdiff, std::abs(f_full[i] - f_win[i]));
            if (maxdiff > 1e-12)
                std::cerr << "kima: transit window misses signal: max diff = "
                          << maxdiff << " (P=" << P << ", e=" << ecc << ")" << std::endl;
        }
        #endif
    }
    else  // fallback and evaluate all the points
    {
        auto ds = rsky_new(t, c[1], c[0], a_over_rs(c), c[4], c[5], c[6]);
        auto f = quadratic_ld(ds, u1, u2, c[2]);
        for (size_t i = 0; i < t.size(); i++)
            mu[i] += sign * (f[i] - 1.0);
    }
}

/**
 * @brief Calculate the transit signal of all planets
 *
 * Incremental updates are used when possible: planets that were removed by the
 * last proposal are subtracted and the ones that were added are added. 
 * @param full  force the recalculation from scratch (e.g. if limb darkening changed)
 */
void TRANSITmodel::calculate_mu(bool full)
{
    const bool update = !full &&
                        (planets.get_added().size() < planets.get_components().size()) &&
                        (staleness <= 10);

    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    if (update) {
        staleness++;
        for (const auto& c : planets.get_removed())
            add_transit(c, -1.0);
        for (const auto& c : planets.get_added())
            add_transit(c, +1.0);
    }
    else {
        staleness = 0;
        mu.assign(mu.size(), 0.0);
        for (const auto& c : planets.get_components())
            add_transit(c, +1.0);
    }

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Model eval took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif

}

void TRANSITmodel::calculate_mu_bkg()
{
    const size_t N = data.N();
    const auto& t = data.get_t();

    mu_bkg.assign(N, background);
    if (trend) {
        for (size_t i = 0; i < N; i++) {
            const double dt = t[i] - t_mid;
            mu_bkg[i] += slope * dt + quadr * dt * dt + cubic * dt * dt * dt;
        }
    }
}


double TRANSITmodel::perturb(RNG& rng)
{
    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    double logH = 0.;

    if(npmax > 0 && rng.rand() <= planet_perturb_prob) // perturb planet parameters
    {
        logH += planets.perturb(rng);
        calculate_mu();
    }
    else
    {
        const double r = rng.rand();

        if (r <= 0.333) // noise parameters
        {
            logH += Jprior->perturb(jitter, rng);
            if (studentt)
                logH += nu_prior->perturb(nu, rng);
        }
        else if (r <= 0.666) // limb darkening / stellar density: change every transit
        {
            if (use_stellar_density && rng.rand() <= 0.5)
                logH += rho_prior->perturb(rho, rng);
            else {
                logH += ld1_prior->perturb(ld1, rng);
                logH += ld2_prior->perturb(ld2, rng);
            }
            calculate_mu(true);
        }
        else // baseline
        {
            logH += Cprior->perturb(background, rng);
            if (trend) {
                if (degree >= 1) logH += slope_prior->perturb(slope, rng);
                if (degree >= 2) logH += quadr_prior->perturb(quadr, rng);
                if (degree == 3) logH += cubic_prior->perturb(cubic, rng);
            }
            calculate_mu_bkg();
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
    const size_t N = data.N();
    const auto& y = data.get_y();
    const auto& sig = data.get_sig();

    if (!ld_is_valid())
        return -std::numeric_limits<double>::infinity();

    // the planet's orbit must be outside the star
    for (const auto& c : planets.get_components())
        if (a_over_rs(c) <= 1.0 + c[2])
            return -std::numeric_limits<double>::infinity();

    double logL = 0.;
    const double jit2 = jitter * jitter;

    #if TIMING
    auto begin = std::chrono::high_resolution_clock::now();  // start timing
    #endif

    if (studentt) {
        // The following code calculates the log likelihood 
        // in the case of a t-Student model
        const double c_nu = std::lgamma(0.5 * (nu + 1.)) - std::lgamma(0.5 * nu)
                            - 0.5 * log(M_PI * nu);
        for (size_t i = 0; i < N; i++)
        {
            const double var = sig[i]*sig[i] + jit2;
            const double r = y[i] - mu[i] - mu_bkg[i];
            logL += c_nu - 0.5*log(var) - 0.5*(nu + 1.)*log(1.0 + r * r / var / nu);
        }
    }
    else{
        // The following code calculates the log likelihood
        // in the case of a Gaussian likelihood
        for (size_t i = 0; i < N; i++)
        {
            const double var = sig[i] * sig[i] + jit2;
            const double r = y[i] - mu[i] - mu_bkg[i];
            logL += -halflog2pi - 0.5 * log(var) - 0.5 * (r * r / var);
        }
    }

    #if TIMING
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Likelihood took " << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()*1E-6 << " ms" << std::endl;
    #endif

    if(std::isnan(logL) || std::isinf(logL))
        logL = -std::numeric_limits<double>::infinity();

    return logL;
}


void TRANSITmodel::print(std::ostream& out) const
{
    // output precision
    out.setf(ios::fixed,ios::floatfield);
    out.precision(8);

    out << jitter << '\t';

    if (trend)
    {
        out.precision(15);
        if (degree >= 1) out << slope << '\t';
        if (degree >= 2) out << quadr << '\t';
        if (degree == 3) out << cubic << '\t';
        out.precision(8);
    }

    out << ld1 << '\t' << ld2 << '\t';
    if (use_stellar_density)
        out << rho << '\t';

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

    if (kipping_ld) desc += "q1" + sep + "q2" + sep;
    else            desc += "u1" + sep + "u2" + sep;

    if (use_stellar_density) desc += "rho" + sep;

    desc += "ndim" + sep + "maxNp" + sep;
    if(false) // hyperpriors
        desc += "muP" + sep + "wP" + sep + "muK";

    desc += "Np" + sep;

    int maxpl = planets.get_max_num_components();
    if (maxpl > 0) {
        for (const char* name : {"P", "tc", "Rp", "a", "inc", "ecc", "w"})
            for (int i = 1; i <= maxpl; i++)
                desc += name + std::to_string(i) + sep;
    }

    desc += "staleness" + sep;

    if (studentt)
        desc += "nu" + sep;
    
    desc += "vsys"; // TODO: should be called something else

    return desc;
}

/**
 * Save the options of the current model in a INI file.
 * 
*/
void TRANSITmodel::save_setup() {
	std::fstream fout("kima_model_setup.txt", std::ios::out);
    fout << std::boolalpha << std::fixed;
    fout.precision(15);

    fout << "; " << timestamp() << endl << endl;

    fout << "[kima]" << endl;

    fout << "model: " << "TRANSITmodel" << endl << endl;
    fout << "fix: " << fix << endl;
    fout << "npmax: " << npmax << endl << endl;

    fout << "hyperpriors: " << false << endl;
    fout << "trend: " << trend << endl;
    fout << "degree: " << degree << endl;
    fout << "studentt: " << studentt << endl;
    fout << "kipping_ld: " << kipping_ld << endl;
    fout << "use_stellar_density: " << use_stellar_density << endl;
    fout << "use_transit_windows: " << use_transit_windows << endl;
    fout << endl << endl;

    fout << "[data]" << endl;
    fout << "file: " << data._datafile << endl;
    fout << "units: " << data._units << endl;
    fout << "skip: " << data._skip << endl;

    fout << "files: ";
    for (auto f: data._datafiles)
        fout << f << ",";
    fout << endl;

    fout << "M0_epoch: " << data.M0_epoch << endl;

    fout << endl;

    fout << "[priors.general]" << endl;
    fout << "Cprior: " << *Cprior << endl;
    fout << "Jprior: " << *Jprior << endl;
    if (trend){
        if (degree >= 1) fout << "slope_prior: " << *slope_prior << endl;
        if (degree >= 2) fout << "quadr_prior: " << *quadr_prior << endl;
        if (degree == 3) fout << "cubic_prior: " << *cubic_prior << endl;
    }
    fout << (kipping_ld ? "q1_prior: " : "u1_prior: ") << *ld1_prior << endl;
    fout << (kipping_ld ? "q2_prior: " : "u2_prior: ") << *ld2_prior << endl;
    
    if (use_stellar_density)
        fout << "rho_prior: " << *rho_prior << endl;
    
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
        using TRANSITmodel::kipping_ld;
        using TRANSITmodel::use_stellar_density;
        using TRANSITmodel::use_transit_windows;
        using TRANSITmodel::window_margin;
};


NB_MODULE(TRANSITmodel, m) {
    // registers TRANSITConditionalPrior so it can be returned
    bind_TRANSITConditionalPrior(m);

    nb::class_<TRANSITmodel>(m, "TRANSITmodel")
        .def(nb::init<bool&, int&, PHOTdata&>(), "fix"_a, "npmax"_a, "data"_a, TRANSITMODEL_DOC)
        //
        .def_rw("fix", &TRANSITmodel::fix, "whether the number of Keplerians is fixed")
        .def_rw("npmax", &TRANSITmodel::npmax, "maximum number of Keplerians")
        .def_ro("data", &TRANSITmodel::data, "the data")
        // 
        .def_rw("directory", &TRANSITmodel::directory, "directory where the model ran")

        // 
        .def_rw("trend", &TRANSITmodel_publicist::trend,
                "whether the model includes a polynomial trend")
        .def_rw("degree", &TRANSITmodel_publicist::degree,
                "degree of the polynomial trend")
        //
        .def_rw("studentt", &TRANSITmodel_publicist::studentt,
                "use a Student-t distribution for the likelihood (instead of Gaussian)")
        //
        .def_rw("kipping_ld", &TRANSITmodel_publicist::kipping_ld,
                "sample limb darkening as (q1, q2) of Kipping (2013); if False, as (u1, u2)")

        .def_rw("use_stellar_density", &TRANSITmodel_publicist::use_stellar_density,
                "derive a/Rs of all planets from a common stellar density (g/cm^3)")

        .def_rw("use_transit_windows", &TRANSITmodel_publicist::use_transit_windows,
                "only evaluate the transit model near the transits (exact speed-up)")
        .def_rw("window_margin", &TRANSITmodel_publicist::window_margin,
                "safety factor (>= 1) for the half-width of the transit windows")

        // to un/pickle TRANSITmodel
        .def("__getstate__", [](const TRANSITmodel &m) { return std::make_tuple(m.fix, m.npmax, m.data); })
        .def("__setstate__", 
             [](TRANSITmodel &m, const std::tuple<bool, int, PHOTdata&> &state)
             {
                new (&m) TRANSITmodel(std::get<0>(state), std::get<1>(state), std::get<2>(state));
             })

        // priors
        .def_prop_rw("Cprior",
            [](TRANSITmodel &m) { return m.Cprior; },
            [](TRANSITmodel &m, distribution &d) { m.Cprior = d; },
            "Prior for the baseline flux")
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

        .def_prop_rw("ld1_prior",
            [](TRANSITmodel &m) { return m.ld1_prior; },
            [](TRANSITmodel &m, distribution &d) { m.ld1_prior = d; },
            "Prior for the first limb-darkening parameter (q1 or u1)")
        .def_prop_rw("ld2_prior",
            [](TRANSITmodel &m) { return m.ld2_prior; },
            [](TRANSITmodel &m, distribution &d) { m.ld2_prior = d; },
            "Prior for the second limb-darkening parameter (q2 or u2)")

        .def_prop_rw("rho_prior",
            [](TRANSITmodel &m) { return m.rho_prior; },
            [](TRANSITmodel &m, distribution &d) { m.rho_prior = d; },
            "Prior for the stellar density in g/cm^3 (if use_stellar_density)")

        .def_prop_rw("nu_prior",
            [](TRANSITmodel &m) { return m.nu_prior; },
            [](TRANSITmodel &m, distribution &d) { m.nu_prior = d; },
            "Prior for the degrees of freedom of the Student-t likelihood")

            // conditional object
        .def_prop_rw("conditional",
                     [](TRANSITmodel &m) { return m.get_conditional_prior(); },
                     [](TRANSITmodel &m, TRANSITConditionalPrior& c) { /* does nothing */ });
}