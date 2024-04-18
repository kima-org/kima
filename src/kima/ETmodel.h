#pragma once

#include <vector>
#include <memory>
#include "DNest4.h"
#include "Data.h"
#include "ConditionalPrior.h"
#include "utils.h"
#include "kepler.h"

using namespace std;
using namespace DNest4;

// for nanobind
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
namespace nb = nanobind;
using namespace nb::literals;
#include "nb_shared.h"



class KIMA_API ETmodel
{
    protected:
        /// Fix the number of planets? (by default, yes)
        bool fix {true};

        /// Maximum number of planets (by default 1)
        int npmax {1};
        
        /// type of ephemeris
        int ephemeris {1};

        /// use a Student-t distribution for the likelihood (instead of Gaussian)
        bool studentt {false};
        
        double star_mass = 1.0;  // [Msun]
        
        ETData data;
    
    private:

        DNest4::RJObject<ETConditionalPrior> planets =
            DNest4::RJObject<ETConditionalPrior>(5, npmax, fix, ETConditionalPrior());

        double ephem1, ephem2=0.0, ephem3=0.0;
        double nu;
        double jitter;

        // Parameters for the known object, if set
        // double KO_P, KO_K, KO_e, KO_phi, KO_w;
        std::vector<double> KO_P;
        std::vector<double> KO_K;
        std::vector<double> KO_e;
        std::vector<double> KO_phi;
        std::vector<double> KO_w;

        // The signal
        std::vector<double> mu ;
        void calculate_mu();
        void add_known_object();
        void remove_known_object();
        
        

        unsigned int staleness;

    public:
        
        ETmodel() {};
        ETmodel(bool fix, int npmax, ETData& data) : fix(fix), npmax(npmax), data(data) {
            initialize_from_data(data);
        };

        void initialize_from_data(ETData& data);

        using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;
        // priors for parameters *not* belonging to the planets
        /// Prior for the extra white noise (jitter).
        distribution Jprior;
        /// Prior for the stellar jitter (common to all instruments)
        distribution stellar_jitter_prior;
        /// Prior for the slope
        distribution ephem1_prior;
        /// Prior for the quadratic coefficient of the trend
        distribution ephem2_prior;
        /// Prior for the cubic coefficient of the trend
        distribution ephem3_prior;


        /* KO mode! */

        /// include (better) known extra Keplerian curve(s)?
        bool known_object {false};
        bool get_known_object() { return known_object; }

        /// how many known objects
        size_t n_known_object {0};
        size_t get_n_known_object() { return n_known_object; }
        

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


        /// Prior for the degrees of freedom $\nu$ of the Student t likelihood
        distribution nu_prior;
        
        ETConditionalPrior* get_conditional_prior() {
            return planets.get_conditional_prior();
        }
        void set_conditional_prior(const ETConditionalPrior &conditional) {
            planets = DNest4::RJObject<ETConditionalPrior>(5, npmax, fix, conditional);
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

