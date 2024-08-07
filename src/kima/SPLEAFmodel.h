#pragma once

#include <vector>
#include <memory>
#include "DNest4.h"
#include "Data.h"
#include "ConditionalPrior.h"
#include "utils.h"
#include "kepler.h"
#include "AMDstability.h"
#include "spleaf.h"

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

class  SPLEAFmodel
{
    private:
        RVData data;// = RVData::get_instance();

        /// Fix the number of planets? (by default, yes)
        bool fix {true};

        /// Maximum number of planets (by default 1)
        int npmax {1};

        /// Whether the model is for multiple timeseries (or just RVs)
        bool multi_series {false};
        
        /// Number of time series
        size_t nseries {0};

        DNest4::RJObject<RVConditionalPrior> planets =
            DNest4::RJObject<RVConditionalPrior>(5, npmax, fix, RVConditionalPrior());

        double background;

        /// whether the model includes a linear trend
        bool trend {false};
        /// and its degree
        int degree {0};

        /// include (better) known extra Keplerian curve(s)? (KO mode!)
        bool known_object {false};
        int n_known_object {0};

        std::vector<double> offsets; // between instruments
            //   std::vector<double>(0, data.number_instruments - 1);
        std::vector<double> jitters; // for each instrument
            //   std::vector<double>(data.number_instruments);

        // std::vector<double> betas; // "slopes" for each indicator
        //     //   std::vector<double>(data.number_indicators);

        double slope, quadr=0.0, cubic=0.0;
        double extra_sigma;
        double nu;

        // Parameters for the known object, if set
        // double KO_P, KO_K, KO_e, KO_phi, KO_w;
        std::vector<double> KO_P;
        std::vector<double> KO_K;
        std::vector<double> KO_e;
        std::vector<double> KO_phi;
        std::vector<double> KO_w;

        // the RV Keplerian signal
        std::vector<double> mu; // = std::vector<double>(data.N());
        // residuals
        VectorXd residuals;
        // the SPLEAF covariance model
        // Error* err;
        Term* _kernel;
        MultiSeriesKernel ms;
        Cov cov;

        void calculate_mu();
        void add_known_object();
        void remove_known_object();

        double star_mass = 1.0;  // [Msun]
        int is_stable() const;
        bool enforce_stability = false;

        unsigned int staleness;


    public:
        SPLEAFmodel() {};
        SPLEAFmodel(bool fix, int npmax, RVData& data, Term& kernel, bool multi_series)
            : data(data), fix(fix), npmax(npmax), multi_series(multi_series)
        {
            _kernel = &kernel;
            initialize_from_data(data, *_kernel);
        };

        void initialize_from_data(RVData& data, Term& kernel);

        // getter and setter for trend
        bool get_trend() const { return trend; };
        void set_trend(bool t) { trend = t; };
        // getter and setter for degree
        double get_degree() const { return degree; };
        void set_degree(double d) { degree = d; };

        // priors for parameters *not* belonging to the planets
        /// Prior for the systemic velocity.
        std::shared_ptr<DNest4::ContinuousDistribution> Cprior;
        /// Prior for the extra white noise (jitter).
        std::shared_ptr<DNest4::ContinuousDistribution> Jprior;
        /// Prior for the slope
        std::shared_ptr<DNest4::ContinuousDistribution> slope_prior;
        /// Prior for the quadratic coefficient of the trend
        std::shared_ptr<DNest4::ContinuousDistribution> quadr_prior;
        /// Prior for the cubic coefficient of the trend
        std::shared_ptr<DNest4::ContinuousDistribution> cubic_prior;
        /// (Common) prior for the between-instruments offsets.
        std::shared_ptr<DNest4::ContinuousDistribution> offsets_prior;
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> individual_offset_prior;
        // { (size_t) data.number_instruments - 1 };
        /// no doc.
        // std::shared_ptr<DNest4::ContinuousDistribution> betaprior;

        // priors for KO mode!
        /// Prior for the KO orbital period(s)
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_Pprior {(size_t) n_known_object};
        /// Prior for the KO semi-amplitude(s)
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_Kprior {(size_t) n_known_object};
        /// Prior for the KO eccentricity(ies)
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_eprior {(size_t) n_known_object};
        /// Prior for the KO mean anomaly(ies)
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_phiprior {(size_t) n_known_object};
        /// Prior for the KO argument(s) of pericenter
        std::vector<std::shared_ptr<DNest4::ContinuousDistribution>> KO_wprior {(size_t) n_known_object};

        // priors for the GP hyperparameters
        /// Prior for $\eta_1$, the GP "amplitude"
        std::shared_ptr<DNest4::ContinuousDistribution> eta1_prior;
        /// Prior for $\eta_2$, the GP correlation timescale
        std::shared_ptr<DNest4::ContinuousDistribution> eta2_prior;
        /// Prior for $\eta_3$, the GP period
        std::shared_ptr<DNest4::ContinuousDistribution> eta3_prior;
        /// Prior for $\eta_4$, the recurrence timescale
        std::shared_ptr<DNest4::ContinuousDistribution> eta4_prior;


        // /// @brief an alias for RVData::get_instance()
        // static RVData& get_data() { return RVData::get_instance(); }
        // Term* get_kernel() {
        //     return kernel;
        // }


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
        double log_likelihood();

        // Print parameters to stream
        void print(std::ostream& out) const;

        // Return string with column information
        std::string description() const;

        // Directory where the model runs
        std::string directory = "";

};

