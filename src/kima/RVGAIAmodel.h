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

using namespace std;
using namespace DNest4;

// for nanobind
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
namespace nb = nanobind;
using namespace nb::literals;
#include "nb_shared.h"


class KIMA_API RVGAIAmodel
{
    protected:
    
        /// Fix the number of planets? (by default, yes)
        bool fix {true};
        /// Maximum number of planets (by default 1)
        int npmax {1};
        
        /// use a Student-t distribution for the likelihood (instead of Gaussian)
        bool studentt {false};
    
        // include (better) known extra Keplerian curve(s)? (KO mode!)
//         bool known_object {false};
        // how many known objects
//         int n_known_object {0};

        /// stellar mass (in units of Msun)
        double star_mass {1.0};
        
        /// whether the model includes a polynomial trend
        bool trend {false};
        /// degree of the polynomial trend
        int degree {0};

        /// RA and DEC of target (in degrees)
        double RA = 0.0;
        double DEC = 0.0;

        /// include in the model linear correlations with indicators
        bool indicator_correlations = false;
        
        GAIAdata GAIA_data;
        RVData RV_data;
    
    private:
    
        DNest4::RJObject<RVGAIAConditionalPrior> planets =
            DNest4::RJObject<RVGAIAConditionalPrior>(7, npmax, fix, RVGAIAConditionalPrior());
        
        std::vector<double> offsets; // between instruments
            //   std::vector<double>(0, data.number_instruments - 1);
        std::vector<double> jitters; // for each instrument
            //   std::vector<double>(data.number_instruments);

        std::vector<double> betas; // "slopes" for each indicator
            //   std::vector<double>(data.number_indicators);

        double background;
        double slope, quadr=0.0, cubic=0.0;
        double nu_RV;
        double jitter_RV;

        double nu_GAIA;
        double jitter_GAIA;
        
        double da; 
        double dd;
        double mua;
        double mud;
        double plx;

        // Parameters for accelerations if using
        double accela;
        double acceld;
        double jerka;
        double jerkd;

        // Parameters of the scan-angle bias modelling if set
        std::vector<double> Ak;
        std::vector<double> thetak;

        // Parameters for the known object, if set. Use geometric parameters rather than thiele_innes
        // double KO_P, KO_K, KO_e, KO_phi, KO_w;
        std::vector<double> KO_P;
        std::vector<double> KO_a0;
        std::vector<double> KO_e;
        std::vector<double> KO_phi;
        std::vector<double> KO_omega;
        std::vector<double> KO_cosi;
        std::vector<double> KO_Omega;

        // Vectors to hold the masses of objects inside the orbit of each body
        std::vector<double> KO_Mints;
        std::vector<double> Mints;

        // The signal
        std::vector<double> mu_GAIA;// = the astrometric model
                            //std::vector<double>(GaiaData::get_instance().N());
        
        std::vector<double> mu_RV;// = the radial velocity model
                            //std::vector<double>(GaiaData::get_instance().N());
                            
        void calculate_mu();
        void get_interior_masses();
        void add_known_object();
        void remove_known_object();

        unsigned int staleness;


    public:
        RVGAIAmodel() {};
        RVGAIAmodel(bool fix, int npmax, GAIAdata& GAIA_data, RVData& RV_data) 
        : GAIA_data(GAIA_data), RV_data(RV_data), fix(fix), npmax(npmax) {
            initialize_from_data(GAIA_data, RV_data);
        };

        void initialize_from_data(GAIAdata& GAIAdata, RVData& RVdata);
        
        // priors for parameters *not* belonging to the planets
        using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;
        // Prior for the extra white noise (jitter) for GAIA data.
        distribution J_GAIA_prior;
        /// prior for student-t degree of freedom for GAIA data
        distribution nu_GAIA_prior;
        // Prior for the extra white noise (jitter) for RV data.
        distribution J_RV_prior;
        /// prior for student-t degree of freedom for RV data
        distribution nu_RV_prior;

        // ///prior for central star mass, removed to sample in K instead
        // distribution star_mass_prior; 
        
        // priors for RV solution
        /// Prior for the systemic velocity.
        distribution Cprior;
        /// Prior for the slope
        distribution slope_prior;
        /// Prior for the quadratic coefficient of the trend
        distribution quadr_prior;
        /// Prior for the cubic coefficient of the trend
        distribution cubic_prior;
        /// (Common) prior for the between-instrument offsets.
        distribution offsets_prior;
        std::vector<distribution> individual_offset_prior;
        // { (size_t) data.number_instruments - 1 };
        /// no doc.
        distribution betaprior;

        ///Whether to include an astrometric model of the along-scan bias from a close binary (see Holl et al. 2023)
        bool al_scan_bias {false};
        bool get_al_scan_bias() { return al_scan_bias; }
        ///number of components of the al_scan bias model up to 3?
        size_t al_scan_bias_components {0};
        size_t get_al_scan_bias_components() { return al_scan_bias_components; }
        /// set the number of components
        void set_al_scan_bias(size_t al_scan_bias_components);


        std::vector<distribution> Ak_prior;
        std::vector<distribution> thetak_prior;

        ///Whether to use an astrometric acceleration solution (i.e. 7-parameter or 9-parameter rather than the default 5-parameter solution)
        bool acceleration {false};
        bool jerk {false};
        size_t n_background_params {5};
        size_t get_n_background_params() { return n_background_params; }
        void set_background_solution(size_t n_background_params);
        
        distribution accela_prior;
        distribution acceld_prior;
        distribution jerka_prior;
        distribution jerkd_prior;
        
        //priors for astrometric solution
        distribution da_prior;
        distribution dd_prior;
        distribution mua_prior;
        distribution mud_prior;
        distribution plx_prior;
        
        bool known_object {false};
        bool get_known_object() { return known_object; }

        /// how many known objects
        size_t n_known_object {0};
        size_t get_n_known_object() { return n_known_object; }

        void set_known_object(size_t known_object);
        // priors for KO mode!
        std::vector<distribution> KO_Pprior;
        std::vector<distribution> KO_a0prior;
        std::vector<distribution> KO_eprior;
        std::vector<distribution> KO_phiprior;
        std::vector<distribution> KO_omegaprior;
        std::vector<distribution> KO_cosiprior;
        std::vector<distribution> KO_Omegaprior;
//         distribution KO_a0prior {(size_t) n_known_object};
//         distribution KO_eprior {(size_t) n_known_object};
//         distribution KO_phiprior {(size_t) n_known_object};
//         distribution KO_omegaprior {(size_t) n_known_object};
//         distribution KO_cosiprior {(size_t) n_known_object};
//         distribution KO_Omegaprior {(size_t) n_known_object};
        
        

        RVGAIAConditionalPrior* get_conditional_prior() {
            return planets.get_conditional_prior();
        }
        void set_conditional_prior(const RVGAIAConditionalPrior &conditional) {
            planets = DNest4::RJObject<RVGAIAConditionalPrior>(7, npmax, fix, conditional);
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

