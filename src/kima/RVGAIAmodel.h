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
        
        /// whether the model includes a polynomial trend
        bool trend {false};
        /// degree of the polynomial trend
        int degree {0};
        
        /// stellar mass (in units of Msun)
        double star_mass = 1.0;
        /// include in the model linear correlations with indicators
        bool indicator_correlations = false;
        
        GAIAData GAIAdata;
        RVData RVdata;
    
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
        
        double da; 
        double dd;
        double mua;
        double mud;
        double plx;
        
        double nu_GAIA;
        double jitter_GAIA;

        // Parameters for the known object, if set. Use geometric parameters rather than thiele_innes
        // double KO_P, KO_K, KO_e, KO_phi, KO_w;
        std::vector<double> KO_P;
        std::vector<double> KO_M;
        std::vector<double> KO_e;
        std::vector<double> KO_phi;
        std::vector<double> KO_omega;
        std::vector<double> KO_cosi;
        std::vector<double> KO_Omega;

        // The signal
        std::vector<double> mu_GAIA;// = the astrometric model
                            //std::vector<double>(GaiaData::get_instance().N());
        
        std::vector<double> mu_RV;// = the astrometric model
                            //std::vector<double>(GaiaData::get_instance().N());
                            
        void calculate_mu();
        void add_known_object();
        void remove_known_object();

        unsigned int staleness;


    public:
        RVGAIAmodel() {};
        RVGAIAmodel(bool fix, int npmax, GAIAData& GAIAdata, RVData& RVdata) : GAIAdata(GAIAdata), RVdata(RVdata), fix(fix), npmax(npmax) {
            initialize_from_data(GAIAdata, RVdata);
        };

        void initialize_from_data(GAIAData& GAIAdata, RVData& RVdata);
        
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
        std::vector<distribution> KO_Mprior;
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

};

