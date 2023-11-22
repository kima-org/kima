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
    
        /// include (better) known extra Keplerian curve(s)? (KO mode!)
        bool known_object {true};
        /// how many known objects
        int n_known_object {1};
        
        ///Whether to use thiele-innes parametrisation
        bool thiele-innes {false};
        
    
    private:
    
        GAIAdata data;
        /// Fix the number of planets? (by default, yes)
        bool fix {true};
        /// Maximum number of planets (by default 1)
        int npmax {1};

        DNest4::RJObject<GAIAConditionalPrior> planets =
            DNest4::RJObject<GAIAConditionalPrior>(7, npmax, fix, GAIAConditionalPrior());

        double da; 
        double dd;
        double mua;
        double mud;
        double plx;
        
        double nu;
        double jitter;

        // Parameters for the known object, if set. Use geometric parameters rather than Thiele-Innes
        // double KO_P, KO_K, KO_e, KO_phi, KO_w;
        std::vector<double> KO_P;
        std::vector<double> KO_a;
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
        
        double star_mass = 1.0;  // [Msun]

        unsigned int staleness;


    public:
        GAIAmodel();
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
        
        // priors for KO mode!
        distribution KO_Pprior {(size_t) n_known_object};
        distribution KO_aprior {(size_t) n_known_object};
        distribution KO_eprior {(size_t) n_known_object};
        distribution KO_phiprior {(size_t) n_known_object};
        distributionKO_wprior {(size_t) n_known_object};
        distribution KO_cosiprior {(size_t) n_known_object};
        distribution KO_Omprior {(size_t) n_known_object};
        
        //priors for astrometric solution
        distribution da_prior;
        distribution dd_prior;
        distribution mua_prior;
        distribution mud_prior;
        distribution plx_prior;

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

