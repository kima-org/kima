#pragma once

#include <vector>
#include <memory>
#include "DNest4.h"
#include "Data.h"
#include "ConditionalPrior.h"
#include "utils.h"
#include "kepler.h"
#include "postkepler.h"
#include "AMDstability.h"
#include "default_priors.h"

using namespace std;
using namespace DNest4;

// for nanobind
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
namespace nb = nanobind;
using namespace nb::literals;
#include "nb_shared.h"



class KIMA_API BINARIESmodel
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
    
        /// use a Student-t distribution for the likelihood (instead of Gaussian)
        bool studentt {false};
    
        /// include (better) known extra Keplerian curve(s)? (KO mode!)
        bool known_object {true};
        /// how many known objects
        int n_known_object {1};
    
        /// stellar masses and radii (in units of Msun)
        double star_mass = 1.0;
        double star_radius = 0.0; //if not specified set to zero
        double binary_mass = 0.0; //if not specified set to zero
        double binary_radius = 0.0; //if not specified set to zero
    
        /// whether to enforce AMD-stability
        bool enforce_stability = false;
        
        ///whether to perform the GR and tidal corrections (Tidal in particular is computationally expensive)
        bool relativistic_correction = false;
        bool tidal_correction = false;
        
        ///Is the binary a double lined binary with RV data on both stars
        bool double_lined = false;

        /// Is the binary eclipsing (i.e. should the cosi be set to 0)
        bool eclipsing = true;

        /// Whether to sample using the mean longitude rather than mean anomaly at epoch
        ///for use when the binary is close to circular for better sampling
        bool use_binary_longitude {false};


        RVData data;
    private:
        

        DNest4::RJObject<KeplerianConditionalPrior> planets =
            DNest4::RJObject<KeplerianConditionalPrior>(5, npmax, fix, KeplerianConditionalPrior());

        double bkg, bkg2;

        //primary
        std::vector<double> offsets; // between instruments
        std::vector<double> jitters; // for each instrument
              
        //secondary
        std::vector<double> offsets_2; // between instruments
        std::vector<double> jitters_2; // for each instrument
        
        
        //std::vector<double> betas = // "slopes" for each indicator
              //std::vector<double>(get_data().number_indicators);

        double slope, quadr=0.0, cubic=0.0;
        double extra_sigma, extra_sigma_2;
        double nu;
        double bin_phi = 0.0;
        

        // Parameters for the known object, if set
        // double KO_P, KO_K, KO_e, KO_phi, KO_w, KO_wdot;
        std::vector<double> KO_P;
        std::vector<double> KO_K;
        std::vector<double> KO_q;
        std::vector<double> KO_e;
        std::vector<double> KO_phi;
        std::vector<double> KO_w;
        std::vector<double> KO_wdot;
        std::vector<double> KO_cosi;

        // The signal
        std::vector<double> mu;// the RV model
        std::vector<double> mu_2;// the RV model for secondary
        void calculate_mu();
        void calculate_mus();
        void add_known_object();
        void add_known_object_sb2();
        void remove_known_object();
        void remove_known_object_sb2();
        
        int is_stable() const;

        unsigned int staleness;

        //void setPriors();
        //void save_setup();

    public:
        
        BINARIESmodel() {};
        BINARIESmodel(bool fix, int npmax, RVData& data) : data(data), fix(fix), npmax(npmax) {
            initialize_from_data(data);
        };

        void initialize_from_data(RVData& data);


        // priors for parameters *not* belonging to the planets
        using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;
        /// Prior for the systemic velocity.
        distribution Cprior;
        /// Prior for the extra white noise (jitter).
        distribution Jprior;
        /// Prior for the slope (used if `trend = true`).
        distribution slope_prior;
        distribution  quadr_prior;
        distribution  cubic_prior;
        /// (Common) prior for the between-instruments offsets.
        distribution  offsets_prior;
        std::vector<distribution>   individual_offset_prior ;


        // priors for KO mode!
        std::vector<distribution> KO_Pprior {(size_t) n_known_object};
        std::vector<distribution> KO_Kprior {(size_t) n_known_object};
        std::vector<distribution> KO_qprior {(size_t) n_known_object};
        std::vector<distribution> KO_eprior {(size_t) n_known_object};
        std::vector<distribution> KO_phiprior {(size_t) n_known_object};
        std::vector<distribution> KO_wprior {(size_t) n_known_object};
        std::vector<distribution> KO_wdotprior {(size_t) n_known_object};
        std::vector<distribution> KO_cosiprior {(size_t) n_known_object};


        distribution nu_prior;
        
        KeplerianConditionalPrior* get_conditional_prior() {
            return planets.get_conditional_prior();
        }
        void set_conditional_prior(const KeplerianConditionalPrior &conditional) {
            planets = DNest4::RJObject<KeplerianConditionalPrior>(5, npmax, fix, conditional);
        }
        
        /// @brief Generate a point from the prior.
        void from_prior(DNest4::RNG& rng);

        /// @brief Set the default priors
        void setPriors();

        /// @brief Save the setup of this model
        void save_setup();

        /// @brief Do Metropolis-Hastings proposals.
        double perturb(DNest4::RNG& rng);

        /// @brief log-likelihood function
        double log_likelihood() const;

        // Print parameters to stream
        void print(std::ostream& out) const;

        // Return string with column information
        std::string description() const;

        // Directory where the model runs
        std::string directory = "";

};

