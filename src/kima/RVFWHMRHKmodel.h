#pragma once

#include <vector>
#include <memory>
#include <exception>

#include "DNest4.h"
#include "Data.h"
#include "ConditionalPrior.h"
#include "utils.h"
#include "kepler.h"
#include "AMDstability.h"
#include "default_priors.h"
#include "GP.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

using namespace std;
using namespace DNest4;

// for nanobind
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
namespace nb = nanobind;
using namespace nb::literals;
#include "nb_shared.h"

class  RVFWHMRHKmodel
{
    protected:
        /// Fix the number of planets? (by default, yes)
        bool fix {true};

        /// Maximum number of planets (by default 1)
        int npmax {1};

        /// whether the model includes a polynomial trend
        bool trend {false};
        /// degree of the polynomial trend
        int degree {0};

        /// stellar mass (in units of Msun)
        double star_mass = 1.0;

        /// whether to enforce AMD-stability
        bool enforce_stability = false;

        // share some of the hyperparameters?
        /// Whether $\eta_2$ is shared between RVs and FWHM
        bool share_eta2 {true};
        /// Whether $\eta_3$ is shared between RVs and FWHM
        bool share_eta3 {true};
        /// Whether $\eta_4$ is shared between RVs and FWHM
        bool share_eta4 {true};

        /// Whether to include a periodic GP kernel for the magnetic cycle
        bool magnetic_cycle_kernel {false};

        RVData data;

    private:
        Eigen::VectorXd sig_copy;  // copy of RV uncertainties for the GP covariance
        Eigen::VectorXd sig_fwhm_copy;  // copy of FWHM uncertainties for the GP covariance
        Eigen::VectorXd sig_rhk_copy;  // copy of R'HK uncertainties for the GP covariance

        DNest4::RJObject<KeplerianConditionalPrior> planets =
            DNest4::RJObject<KeplerianConditionalPrior>(5, npmax, fix, KeplerianConditionalPrior());

        double bkg, bkg_fwhm, bkg_rhk;

        std::vector<double> offsets; // between instruments
        std::vector<double> jitters; // for each instrument

        double slope, quadr=0.0, cubic=0.0;
        double jitter, jitter_fwhm, jitter_rhk;
        double nu;

        // Parameters for the known object, if set
        // double KO_P, KO_K, KO_e, KO_phi, KO_w;
        std::vector<double> KO_P;
        std::vector<double> KO_K;
        std::vector<double> KO_e;
        std::vector<double> KO_phi;
        std::vector<double> KO_w;

        // Parameters for the transiting planet, if set
        std::vector<double> TR_P;
        std::vector<double> TR_K;
        std::vector<double> TR_e;
        std::vector<double> TR_Tc;
        std::vector<double> TR_w;

        // Parameters for the Gaussian process
        double eta1, eta2, eta3, eta4;
        double eta1_fw, eta2_fw, eta3_fw, eta4_fw;
        double eta1_rhk, eta2_rhk, eta3_rhk, eta4_rhk;
        double zeta1, zeta2;

        double eta5, eta5_fw, eta5_rhk, eta6, eta7;

        // The signal
        std::vector<double> mu; // = std::vector<double>(data.N());
        std::vector<double> mu_fwhm; // = std::vector<double>(data.N());
        std::vector<double> mu_rhk; // = std::vector<double>(data.N());

        // The covariance matrix for the data
        Eigen::MatrixXd C; // {data.N(), data.N()};
        Eigen::MatrixXd C_fwhm; // {data.N(), data.N()};
        Eigen::MatrixXd C_rhk; // {data.N(), data.N()};

        void calculate_mu();
        void calculate_mu_fwhm();
        void calculate_mu_rhk();

        void add_known_object();
        void remove_known_object();
        void add_transiting_planet();
        void remove_transiting_planet();

        int is_stable() const;

        unsigned int staleness;


    public:
        RVFWHMRHKmodel() {};
        RVFWHMRHKmodel(bool fix, int npmax, RVData& data) : fix(fix), npmax(npmax), data(data) {
            initialize_from_data(data);
        };

        void initialize_from_data(RVData& data);

        // priors for parameters *not* belonging to the planets
        using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;

        /// Prior for the systemic velocity.
        distribution Cprior;
        distribution Cfwhm_prior;
        distribution Crhk_prior;

        /// Prior for the extra white noise (jitter).
        distribution Jprior;
        distribution Jfwhm_prior;
        distribution Jrhk_prior;

        /// Prior for the slope
        distribution slope_prior;
        /// Prior for the quadratic coefficient of the trend
        distribution quadr_prior;
        /// Prior for the cubic coefficient of the trend
        distribution cubic_prior;

        /// (Common) prior for the between-instruments offsets.
        distribution offsets_prior;
        distribution offsets_fwhm_prior;
        distribution offsets_rhk_prior;
        std::vector<distribution> individual_offset_prior; // {(size_t) data.number_instruments - 1};
        std::vector<distribution> individual_offset_fwhm_prior; // {(size_t) data.number_instruments - 1};
        std::vector<distribution> individual_offset_rhk_prior; // {(size_t) data.number_instruments - 1};

        /* KO mode! */

        /// include (better) known extra Keplerian curve(s)?
        bool known_object {false};
        bool get_known_object() { return known_object; }

        /// how many known objects
        size_t n_known_object {0};
        size_t get_n_known_object() { return n_known_object; }

        void set_known_object(size_t known_object);

        /// Prior for the KO orbital period(s)
        std::vector<distribution> KO_Pprior;
        /// Prior for the KO semi-amplitude(s)
        std::vector<distribution> KO_Kprior;
        /// Prior for the KO eccentricity(ies)
        std::vector<distribution> KO_eprior;
        /// Prior for the KO mean anomaly(ies)
        std::vector<distribution> KO_phiprior;
        /// Prior for the KO argument(s) of pericenter
        std::vector<distribution> KO_wprior;

        /* Transiting planets! */

        /// include known extra Keplerian curve(s) for transiting planet(s)?
        bool transiting_planet {false};
        bool get_transiting_planet() { return transiting_planet; }

        /// how many known objects
        size_t n_transiting_planet {0};
        size_t get_n_transiting_planet() { return n_transiting_planet; }

        void set_transiting_planet(size_t transiting_planet);

        /// Prior for the TR orbital period(s)
        std::vector<distribution> TR_Pprior;
        /// Prior for the TR semi-amplitude(s)
        std::vector<distribution> TR_Kprior;
        /// Prior for the TR eccentricity(ies)
        std::vector<distribution> TR_eprior;
        /// Prior for the TR time(s) of transit
        std::vector<distribution> TR_Tcprior;
        /// Prior for the TR argument(s) of pericenter
        std::vector<distribution> TR_wprior;


        KernelType kernel {qp};

        // priors for the GP hyperparameters
        /// Prior for $\eta_1$, the GP "amplitude"
        distribution eta1_prior;
        /// Prior for $\eta_2$, the GP correlation timescale
        distribution eta2_prior;
        /// Prior for $\eta_3$, the GP period
        distribution eta3_prior;
        /// Prior for $\eta_4$, the recurrence timescale
        distribution eta4_prior;

        // same for the FWHM
        /// Same as $\eta_1$ but for the FWHM
        distribution eta1_fwhm_prior;
        /// Same as $\eta_2$ but for the FWHM
        distribution eta2_fwhm_prior;
        /// Same as $\eta_3$ but for the FWHM
        distribution eta3_fwhm_prior;
        /// Same as $\eta_4$ but for the FWHM
        distribution eta4_fwhm_prior;

        // same for the RHK
        /// Same as $\eta_1$ but for the RHK
        distribution eta1_rhk_prior;
        /// Same as $\eta_2$ but for the RHK
        distribution eta2_rhk_prior;
        /// Same as $\eta_3$ but for the RHK
        distribution eta3_rhk_prior;
        /// Same as $\eta_4$ but for the RHK
        distribution eta4_rhk_prior;

        /// Prior for $\eta_5$, the "amplitude" of the magnetic cycle kernel
        distribution eta5_prior;
        distribution eta5_fwhm_prior;
        distribution eta5_rhk_prior;
        /// Prior for $\eta_6$, the period of the magnetic cycle kernel
        distribution eta6_prior;
        /// Prior for $\eta_7$, the recurrence timescale of the magnetic cycle kernel
        distribution eta7_prior;

        KeplerianConditionalPrior* get_conditional_prior() {
            return planets.get_conditional_prior();
        }

        /// @brief Generate a point from the prior.
        void from_prior(DNest4::RNG& rng);

        /// @brief Set the default priors
        void setPriors();

        /// @brief Save the setup of this model
        void save_setup();

        /// @brief Do Metropolis-Hastings proposals.
        double perturb(DNest4::RNG& rng);

        /// @brief Build the covariance matrix of the RVs
        void calculate_C();
        /// @brief Build the covariance matrix of the FWHM
        void calculate_C_fwhm();
        /// @brief Build the covariance matrix of the R'HK
        void calculate_C_rhk();

        /// @brief log-likelihood function
        double log_likelihood() const;

        // Print parameters to stream
        void print(std::ostream& out) const;

        // Return string with column information
        std::string description() const;

        // Directory where the model runs
        std::string directory = "";

};

