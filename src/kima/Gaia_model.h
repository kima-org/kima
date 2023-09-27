#pragma once

#include <vector>
#include <memory>
#include "ConditionalPrior.h"
#include "RJObject/RJObject.h"
#include "RNG.h"
#include "DNest4.h"
#include "Data.h"
#include "kepler.h"
#include "AMDstability.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "celerite/celerite.h"


/// include a (better) known extra Keplerian curve? (KO mode!)
extern const bool known_object;
extern const int n_known_object;

/// use a Student-t distribution for the likelihood (instead of Gaussian)
extern const bool studentt;


class Gaia_model
{
    private:
        /// Fix the number of planets? (by default, yes)
        bool fix {true};
        /// Maximum number of planets (by default 1)
        int npmax {1};

        DNest4::RJObject<GaiaConditionalPrior> planets =
            DNest4::RJObject<GaiaConditionalPrior>(7, npmax, fix, GaiaConditionalPrior());

        double da; 
        double dd;
        double mua;
        double mud;
        double par;
        
        double nu;
        double jitter;

        // Parameters for the known object, if set
        // double KO_P, KO_K, KO_e, KO_phi, KO_w;
        std::vector<double> KO_P;
        std::vector<double> KO_a;
        std::vector<double> KO_e;
        std::vector<double> KO_phi;
        std::vector<double> KO_w;
        std::vector<double> KO_cosi;
        std::vector<double> KO_Om;

        // The signal
        std::vector<double> mu = // the astrometric model
                            std::vector<double>(GaiaData::get_instance().N());
        void calculate_mu();
        void add_known_object();
        void remove_known_object();
        
        double star_mass = 1.0;  // [Msun]

        unsigned int staleness;

        void setPriors();
        void save_setup();

    public:
        Gaia_model();

        void initialise() {};

        // priors for parameters *not* belonging to the planets
        // Prior for the extra white noise (jitter).
        std::shared_ptr<DNest4::ContinuousDistribution> Jprior;
        
        // priors for KO mode!
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_Pprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_aprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_eprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_phiprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_wprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_cosiprior {(size_t) n_known_object};
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_Omprior {(size_t) n_known_object};
        
        std::shared_ptr<DNest4::ContinuousDistribution> da_prior;
        std::shared_ptr<DNest4::ContinuousDistribution> dd_prior;
        std::shared_ptr<DNest4::ContinuousDistribution> mua_prior;
        std::shared_ptr<DNest4::ContinuousDistribution> mud_prior;
        std::shared_ptr<DNest4::ContinuousDistribution> par_prior;

        std::shared_ptr<DNest4::ContinuousDistribution> nu_prior;

        // change the name of std::make_shared :)
        /**
         * @brief Assign a prior distribution.
         * 
         * This function defines, initializes, and assigns a prior distribution.
         * Possible distributions are ...
         * 
         * For example:
         * 
         * @code{.cpp}
         *          Cprior = make_prior<Uniform>(0, 1);
         * @endcode
         * 
         * @tparam T     ContinuousDistribution
         * @tparam Args  
         * @param args   Arguments for constructor of distribution
         * @return std::shared_ptr<T> 
        */
        template <class T, class... Args>
        std::shared_ptr<T> make_prior(Args&&... args)
        {
            return std::make_shared<T>(args...);
        }

        // create an alias for ETData::get_instance()
        static GaiaData& get_data() { return GaiaData::get_instance(); }

        /// @brief Generate a point from the prior.
        void from_prior(DNest4::RNG& rng);

        /// @brief Do Metropolis-Hastings proposals.
        double perturb(DNest4::RNG& rng);

        // Likelihood function
        double log_likelihood() const;

        // Print parameters to stream
        void print(std::ostream& out) const;

        // Return string with column information
        std::string description() const;

};

