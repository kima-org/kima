#pragma once

#include <vector>
#include <memory>
#include "DNest4.h"
#include "Data.h"
#include "ConditionalPrior.h"
#include "utils.h"
#include "kepler.h"
#include "AMDstability.h"

using namespace std;
using namespace DNest4;

// for nanobind
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
namespace nb = nanobind;
using namespace nb::literals;
#include "nb_shared.h"


class KIMA_API GAIAmodel
{
    protected:
        /// use a Student-t distribution for the likelihood (instead of Gaussian)
        bool studentt {false};
        
        /// Fix the number of planets? (by default, yes)
        bool fix {true};
        /// Maximum number of planets (by default 1)
        int npmax {1};
    
        /// include (better) known extra Keplerian curve(s)? (KO mode!)
        ///bool known_object {false};
        /// how many known objects
        ///int n_known_object {0};
        
        ///Whether to use thiele_innes parametrisation
        bool thiele_innes {false};
        
        /// stellar mass (in units of Msun)
        double star_mass = 1.0;
        
    
    private:
    
        GAIAData data;
        

        DNest4::RJObject<GAIAConditionalPrior> planets =
            DNest4::RJObject<GAIAConditionalPrior>(7, npmax, fix, GAIAConditionalPrior());
            

        double da; 
        double dd;
        double mua;
        double mud;
        double plx;
        
        double nu;
        double jitter;

        // Parameters for the known object, if set. Use geometric parameters rather than thiele_innes
        // double KO_P, KO_K, KO_e, KO_phi, KO_w;
        std::vector<double> KO_P;
        std::vector<double> KO_a0;
        std::vector<double> KO_e;
        std::vector<double> KO_phi;
        std::vector<double> KO_omega;
        std::vector<double> KO_cosi;
        std::vector<double> KO_Omega;

        // The signal
        std::vector<double> mu;// = the astrometric model
                            //std::vector<double>(GaiaData::get_instance().N());
                            
        void calculate_mu();
        void add_known_object();
        void remove_known_object();
        

        unsigned int staleness;


    public:
        GAIAmodel() {};
        GAIAmodel(bool fix, int npmax, GAIAData& data) : data(data), fix(fix), npmax(npmax) {
            initialize_from_data(data);
        };

        void initialize_from_data(GAIAData& data);

        // priors for parameters *not* belonging to the planets
        using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;
        // Prior for the extra white noise (jitter).
        distribution Jprior;
        /// prior for student-t degree of freedom
        distribution nu_prior;
        
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
        
        

        GAIAConditionalPrior* get_conditional_prior() {
            return planets.get_conditional_prior();
        }
        void set_conditional_prior(const GAIAConditionalPrior &conditional) {
            planets = DNest4::RJObject<GAIAConditionalPrior>(7, npmax, fix, conditional);
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

