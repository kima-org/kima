#include <sstream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;
using namespace nb::literals;

#include "DNest4.h"

#include "distributions/InverseGamma.h"
#include "distributions/InverseMoment.h"
#include "distributions/ExponentialRayleighMixture.h"
#include "distributions/GaussianMixture.h"
#include "distributions/BivariateGaussian.h"
#include "distributions/PowerLaw.h"
#include "distributions/Sine.h"

// the types of objects in the distributions state (for pickling)
using _state_type = std::tuple<double, double, double, double>;
