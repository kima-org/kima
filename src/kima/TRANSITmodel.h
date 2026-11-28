#pragma once

#include <vector>
#include <memory>
#include "DNest4.h"
#include "Data.h"
#include "ConditionalPrior.h"
#include "utils.h"
#include "kepler.h"
#include "transits.h"
#include "default_priors.h"

using namespace std;
using namespace DNest4;

// for nanobind
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
namespace nb = nanobind;
using namespace nb::literals;
#include "nb_shared.h"


class  TRANSITmodel
{
    public:
        /// Fix the number of planets? (by default, yes)
        bool fix {true};

        /// Maximum number of planets (by default 1)
        int npmax {1};

        PHOTdata data;

    protected:
        /// whether the model includes a polynomial trend
        bool trend {false};
        /// degree of the polynomial trend
        int degree {0};

        /// use a Student-t distribution for the likelihood (instead of Gaussian)
        bool studentt {false};

        /// Sample limb-darkening in the Kipping (2013) (q1, q2) parametrisation
        /// (default). If false, ld1 and ld2 are the quadratic coefficients u1, u2
        bool kipping_ld {true};

        /// Use a stellar-density prior to set a/Rs of every planet (Kepler's
        /// third law) instead of sampling a/Rs independently for each planet
        bool use_stellar_density {false};

        /// only evaluate the transit model for the points that can be in
        /// transit (windows around each mid-transit). Exact, up to the safety
        /// margin below. Requires the times to be sorted.
        bool use_transit_windows {true};
        /// safety factor (>= 1) applied to the window half-width
        double window_margin {1.05};

        /// probability of perturbing the planet parameters
        double planet_perturb_prob = 0.75;

    private:

        DNest4::RJObject<TRANSITConditionalPrior> planets =
            DNest4::RJObject<TRANSITConditionalPrior>(7, npmax, fix, TRANSITConditionalPrior());

        double background;
        double jitter;
        double slope, quadr=0.0, cubic=0.0;
        double nu;
        /// limb darkening: (q1, q2) if kipping_ld, else (u1, u2)
        double ld1 = 0.0, ld2 = 0.0;

        /// stellar density [g/cm^3], common to all planets
        double rho = 1.0;

        double t_mid = 0.0;  // cached reference epoch for the trend
        bool t_sorted = true;  // are the times sorted? (needed for the windows)

        // The planet signal (sum of (flux - 1)) and the baseline
        std::vector<double> mu, mu_bkg;

        void calculate_mu(bool full = false);
        void calculate_mu_bkg();
        /// add (sign=+1) or remove (sign=-1) the transit of one component
        void add_transit(const std::vector<double>& component, double sign);
        /// quadratic limb-darkening coefficients u1, u2
        void get_ld(double& u1, double& u2) const;
        bool ld_is_valid() const;
        /// half-width [days] of the window around a mid-transit outside of
        /// which the planet cannot overlap the stellar disk. Returns false if
        /// no useful (< P/2) window exists, in which case all points are used
        bool transit_halfwidth(double P, double Rp, double a, double e, double& half) const;
        /// a/Rs for a planet of period P [days]: from rho if
        /// use_stellar_density, otherwise the value stored in the component
        double a_over_rs(const std::vector<double>& component) const;

        unsigned int staleness {0};


    public:
        TRANSITmodel() {};
        TRANSITmodel(bool fix, int npmax, PHOTdata& data) : fix(fix), npmax(npmax), data(data)
        {
            initialize_from_data(data);
        };

        void initialize_from_data(PHOTdata& data);

        using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;
        // priors for parameters *not* belonging to the planets
        
        /// Prior for the baseline flux level
        distribution Cprior;
        /// Prior for the extra white noise (jitter).
        distribution Jprior;
        /// Prior for the slope
        distribution slope_prior;
        /// Prior for the quadratic coefficient of the trend
        distribution quadr_prior;
        /// Prior for the cubic coefficient of the trend
        distribution cubic_prior;

        /// Prior for q1 (or u1)
        distribution ld1_prior;
        /// Prior for q2 (or u2)
        distribution ld2_prior;

        /// Prior for the stellar density [g/cm^3] (used if use_stellar_density)
        distribution rho_prior;

        /// Prior for the degrees of freedom $\nu$ of the Student t likelihood
        distribution nu_prior;

        
        TRANSITConditionalPrior* get_conditional_prior() {
            return planets.get_conditional_prior();
        }
        void set_conditional_prior(const TRANSITConditionalPrior &conditional) {
            planets = DNest4::RJObject<TRANSITConditionalPrior>(7, npmax, fix, conditional);
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

