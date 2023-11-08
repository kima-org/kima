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


class  OutlierRVmodel
{
    protected:
        /// whether the model includes a polynomial trend
        bool trend {false};
        /// degree of the polynomial trend
        int degree {0};

        /// use a Student-t distribution for the likelihood (instead of Gaussian)
        bool studentt {false};
    
        /// include (better) known extra Keplerian curve(s)? (KO mode!)
        bool known_object {false};
        /// how many known objects
        int n_known_object {0};

        /// stellar mass (in units of Msun)
        double star_mass = 1.0;

        /// whether to enforce AMD-stability
        bool enforce_stability = false;

    private:
        RVData data;// = RVData::get_instance();

        /// Fix the number of planets? (by default, yes)
        bool fix {true};

        /// Maximum number of planets (by default 1)
        int npmax {1};

        DNest4::RJObject<RVConditionalPrior> planets =
            DNest4::RJObject<RVConditionalPrior>(5, npmax, fix, RVConditionalPrior());

        double background;


        std::vector<double> offsets; // between instruments
            //   std::vector<double>(0, data.number_instruments - 1);
        std::vector<double> jitters; // for each instrument
            //   std::vector<double>(data.number_instruments);

        std::vector<double> betas; // "slopes" for each indicator
            //   std::vector<double>(data.number_indicators);

        double slope, quadr=0.0, cubic=0.0;
        double extra_sigma;
        double nu;

        double Q;
        double outlier_background;
        double outlier_sigma;

        // Parameters for the known object, if set
        // double KO_P, KO_K, KO_e, KO_phi, KO_w;
        std::vector<double> KO_P;
        std::vector<double> KO_K;
        std::vector<double> KO_e;
        std::vector<double> KO_phi;
        std::vector<double> KO_w;

        // The signal
        std::vector<double> mu; // = std::vector<double>(data.N());

        void calculate_mu();
        void add_known_object();
        void remove_known_object();

        int is_stable() const;

        unsigned int staleness;


    public:
        OutlierRVmodel() {};
        OutlierRVmodel(bool fix, int npmax, RVData& data) : fix(fix), npmax(npmax), data(data) {
            initialize_from_data(data);
        };

        void initialize_from_data(RVData& data);

        // priors for parameters *not* belonging to the planets
        using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;
        /// Prior for the systemic velocity.
        distribution Cprior;
        /// Prior for the extra white noise (jitter).
        distribution Jprior;
        /// Prior for the slope
        distribution slope_prior;
        /// Prior for the quadratic coefficient of the trend
        distribution quadr_prior;
        /// Prior for the cubic coefficient of the trend
        distribution cubic_prior;
        /// (Common) prior for the between-instrument offsets.
        distribution offsets_prior;
        std::vector<distribution> individual_offset_prior;
        /// no doc.
        distribution betaprior;

        // priors for KO mode!
        /// Prior for the KO orbital period(s)
        std::vector<distribution> KO_Pprior {(size_t) n_known_object};
        /// Prior for the KO semi-amplitude(s)
        std::vector<distribution> KO_Kprior {(size_t) n_known_object};
        /// Prior for the KO eccentricity(ies)
        std::vector<distribution> KO_eprior {(size_t) n_known_object};
        /// Prior for the KO mean anomaly(ies)
        std::vector<distribution> KO_phiprior {(size_t) n_known_object};
        /// Prior for the KO argument(s) of pericenter
        std::vector<distribution> KO_wprior {(size_t) n_known_object};

        /// Prior for the degrees of freedom $\nu$ of the Student t likelihood
        distribution nu_prior;

        /// Priors for outlier model
        distribution outlier_mean_prior;
        distribution outlier_sigma_prior;
        distribution outlier_Q_prior;

        RVConditionalPrior* get_conditional_prior() {
            return planets.get_conditional_prior();
        }
        void set_conditional_prior(const RVConditionalPrior &conditional) {
            planets = DNest4::RJObject<RVConditionalPrior>(5, npmax, fix, conditional);
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

