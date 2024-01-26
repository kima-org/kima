#pragma once

#include <memory>
#include <cmath>
#include <typeinfo>
#include "DNest4.h"
#include "Data.h"

// for nanobind
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
namespace nb = nanobind;
using namespace nb::literals;


class RVConditionalPrior : public DNest4::ConditionalPrior
{
	private:
		/// whether the model includes hyper-priors for the orbital period and
		/// semi-amplitude
		bool hyperpriors;
		// Parameters of bi-exponential hyper-distribution for log-periods
		double center, width;
		// Mean of exponential hyper-distribution for semi-amplitudes
		double muK;
		double perturb_hyperparameters(DNest4::RNG& rng);

	public:
		RVConditionalPrior();

		void set_default_priors(const RVData &data);

		// priors for all planet parameters
		using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;
		
		/// Prior for the orbital periods.
		distribution Pprior;
		/// Prior for the semi-amplitudes (in m/s).
		distribution Kprior;
		/// Prior for the eccentricities.
		distribution eprior;
		/// Prior for the phases.
		distribution phiprior;
		/// Prior for the .
		distribution wprior;

		// hyperpriors

		// turn on hyperpriors
		void use_hyperpriors();

		/// Prior for the log of the median orbital period
		distribution log_muP_prior;
		/// Prior for the diversity of orbital periods
		distribution wP_prior;
		/// Prior for the log of the mean semi-amplitude
		distribution log_muK_prior;


		/// Generate a point from the prior
		void from_prior(DNest4::RNG& rng);
		/// Get the log prob density at a position `vec`
		double log_pdf(const std::vector<double>& vec) const;
		/// Get parameter sample from a uniform sample (CDF)
		void from_uniform(std::vector<double>& vec) const;
		/// Get uniform sample from a parameter sample (inverse CDF)
		void to_uniform(std::vector<double>& vec) const;

		void print(std::ostream& out) const;
		static const int weight_parameter = 1;
};



class TRANSITConditionalPrior:public DNest4::ConditionalPrior
{
	private:
    	
		double perturb_hyperparameters(DNest4::RNG& rng);

	public:
		TRANSITConditionalPrior();

		void set_default_priors(const PHOTdata &data);

		// priors for all planet parameters
		using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;
		
		/// Prior for the orbital periods.
		distribution Pprior;
		/// Prior for the time of inferior conjunction
		distribution t0prior;
		/// Prior for the planet radius (in units of the stellar radius)
		distribution RPprior;
		/// Prior for the semi-major axis (in units of the stellar radius)
		distribution aprior;
		/// Prior for the inclination
		distribution incprior;
		/// Prior for the eccentricities.
		distribution eprior;
		/// Prior for the argument of periastron
		distribution wprior;

		/// Generate a point from the prior
		void from_prior(DNest4::RNG& rng);
		/// Get the log prob density at a position `vec`
		double log_pdf(const std::vector<double>& vec) const;
		/// Get parameter sample from a uniform sample (CDF)
		void from_uniform(std::vector<double>& vec) const;
		/// Get uniform sample from a parameter sample (inverse CDF)
		void to_uniform(std::vector<double>& vec) const;

		void print(std::ostream& out) const;
		static const int weight_parameter = 1;
};


class GAIAConditionalPrior:public DNest4::ConditionalPrior
{
 	private:
        
        

		double perturb_hyperparameters(DNest4::RNG& rng);

 	public:
		GAIAConditionalPrior();
		
		/// Use thiele_innes parameters
     	bool thiele_innes;

		void set_default_priors(const GAIAData &data);
		
		// priors for all planet parameters
		using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;
		
		/// Prior for the orbital periods.
		distribution Pprior;
		/// Prior for the eccentricities.
		distribution eprior;
		/// Prior for the phases (maybe do T0s instead?).
		distribution phiprior;
		/// Prior for the photocentre semi major axes (in ...).
		distribution a0prior;
		/// Prior for the arguments of periastron.
		distribution omegaprior;
		/// Prior for cos of the inclination.
		distribution cosiprior;
		/// Prior for the longitude of ascending node.
		distribution Omegaprior;
		
		///Priors for the thiele_innes parameters
		distribution Aprior;
		/// 
		distribution Bprior;
		/// 
		distribution Fprior;
		/// 
		distribution Gprior;
		
		distribution Xprior;
		
		// turn on hyperpriors
		void use_thiele_innes();


		/// Generate a point from the prior.
		void from_prior(DNest4::RNG& rng);
        /// Get the log prob density at a position `vec`
		double log_pdf(const std::vector<double>& vec) const;
		/// Get parameter sample from a uniform sample (CDF)
		void from_uniform(std::vector<double>& vec) const;
		/// Get uniform sample from a parameter sample (inverse CDF)
		void to_uniform(std::vector<double>& vec) const;

		void print(std::ostream& out) const;
		static const int weight_parameter = 1;

};

class RVGAIAConditionalPrior:public DNest4::ConditionalPrior
{
 	private:

		double perturb_hyperparameters(DNest4::RNG& rng);

 	public:
		RVGAIAConditionalPrior();

		void set_default_priors(const GAIAData &GAIAdata, RVData &RVdata);
		
		// priors for all planet parameters
		using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;
		
		/// Prior for the orbital periods.
		distribution Pprior;
		/// Prior for the eccentricities.
		distribution eprior;
		/// Prior for the phases (maybe do T0s instead?).
		distribution phiprior;
		/// Prior for the photocentre semi major axes (in ...).
		distribution Mprior;
		/// Prior for the arguments of periastron.
		distribution omegaprior;
		/// Prior for cos of the inclination.
		distribution cosiprior;
		/// Prior for the longitude of ascending node.
		distribution Omegaprior;
		


		/// Generate a point from the prior.
		void from_prior(DNest4::RNG& rng);
        /// Get the log prob density at a position `vec`
		double log_pdf(const std::vector<double>& vec) const;
		/// Get parameter sample from a uniform sample (CDF)
		void from_uniform(std::vector<double>& vec) const;
		/// Get uniform sample from a parameter sample (inverse CDF)
		void to_uniform(std::vector<double>& vec) const;

		void print(std::ostream& out) const;
		static const int weight_parameter = 1;

};




void bind_RVConditionalPrior(nb::module_ &m);
void bind_GAIAConditionalPrior(nb::module_ &m);
void bind_RVGAIAConditionalPrior(nb::module_ &m);

