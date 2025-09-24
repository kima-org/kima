#pragma once

#include <vector>
#include <memory>
#include "DNest4.h"
#include "Data.h"
#include "ConditionalPrior.h"
#include "utils.h"
#include "kmath.h"
#include "kepler.h"
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

enum ApodType { apod_gaussian, apod_boxcar, apod_plateau };

class KIMA_API ApodizedRVmodel
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
    
        /// stellar mass (in units of Msun)
        double star_mass = 1.0;

        /// whether to enforce AMD-stability
        bool enforce_stability = false;

        /// whether to remove the label switching degeneracy on the period
        bool remove_label_switching_degeneracy = false;

        /// include in the model linear correlations with indicators
        bool indicator_correlations = false;

        bool jitter_propto_indicator = false;
        int jitter_propto_indicator_index = 0;

        RVData data;

    private:

        DNest4::RJObject<ApodizedKeplerianConditionalPrior> planets =
            DNest4::RJObject<ApodizedKeplerianConditionalPrior>(8, npmax, fix, ApodizedKeplerianConditionalPrior());

        double background;


        std::vector<double> offsets; // between instruments
            //   std::vector<double>(0, data.number_instruments - 1);
        std::vector<double> jitters; // for each instrument
            //   std::vector<double>(data.number_instruments);

        std::vector<double> betas; // "slopes" for each indicator
            //   std::vector<double>(data.number_indicators);

        double slope, quadr=0.0, cubic=0.0;
        double jitter, stellar_jitter, jitter_propto_indicator_slope;
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

        // The signal
        std::vector<double> mu;
        // The apodization function
        std::vector<double> apod;

        double planet_perturb_prob = 0.75;
        double jitKO_perturb_prob = 0.5;

        void calculate_mu();
        void add_known_object();
        void remove_known_object();
        void add_transiting_planet();
        void remove_transiting_planet();

        // // Solve the label switching degeneracy by mapping
		// // the orbital period P to the hypertriangle where
		// // P(K) >= P(K−1) >= ... >= P(1)
		void solve_label_switching(RNG& rng);

        int is_stable() const;

        unsigned int staleness;

        /**
         * Returns the indices that would sort an array.
         * @param array input array
         * @return indices w.r.t sorted array
        */
        vector<size_t> argsort(const vector<double> &array) {
            vector<size_t> indices(array.size());
            iota(indices.begin(), indices.end(), 0);
            sort(indices.begin(), indices.end(),
                    [&array](size_t left, size_t right) -> bool {
                        // sort indices according to corresponding array element
                        return array[left] < array[right];
                    });
            return indices;
        }


    public:
        ApodizedRVmodel() {};
        ApodizedRVmodel(bool fix, int npmax, RVData& data) : fix(fix), npmax(npmax), data(data) {
            initialize_from_data(data);
        };

        void initialize_from_data(RVData& data);

        using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;
        // priors for parameters *not* belonging to the planets

        /// Prior for the systemic velocity.
        distribution Cprior;
        /// Prior for the extra white noise (jitter).
        // [[deprecated("The default prior for the jitter will change in the future.")]]
        distribution Jprior;
        /// Prior for the stellar jitter (common to all instruments)
        distribution stellar_jitter_prior;
        /// Prior for ...
        distribution jitter_slope_prior;
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
        /// (Common) prior for the activity indicator coefficients
        distribution beta_prior;

        // Type of apodization
        ApodType apodization { apod_gaussian };

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


        /// Prior for the degrees of freedom $\nu$ of the Student t likelihood
        distribution nu_prior;

        ApodizedKeplerianConditionalPrior* get_conditional_prior() {
            return planets.get_conditional_prior();
        }
        void set_conditional_prior(const ApodizedKeplerianConditionalPrior &conditional) {
            planets = DNest4::RJObject<ApodizedKeplerianConditionalPrior>(8, npmax, fix, conditional);
        }

        void set_loguniform_prior_Np() {
            auto conditional = planets.get_conditional_prior();
            planets = DNest4::RJObject<ApodizedKeplerianConditionalPrior>(8, npmax, fix, *conditional,
                                                           DNest4::PriorType::log_uniform);
        };

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

