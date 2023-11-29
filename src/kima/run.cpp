#include <ctime>
#include <iostream>
#include <chrono>

#include <nanobind/nanobind.h>
// #include <nanobind/stl/string.h>
// #include <nanobind/stl/vector.h>
namespace nb = nanobind;
using namespace nb::literals;

#include "RVmodel.h"
#include "GPmodel.h"
#include "RVFWHMmodel.h"
#include "TRANSITmodel.h"
#include "OutlierRVmodel.h"
#include "BINARIESmodel.h"
#include "GAIAmodel.h"


auto RUN_DOC = R"D(
Run the DNest4 sampler with the given model

Args:
    m (model): the model
    steps (int, default=100): how many steps to run for
)D";


#define RUN_SIGNATURE(name) \
    [](name &m, int steps=100, unsigned int num_threads=1, unsigned int num_particles=1,                              \
                unsigned int new_level_interval=2000, unsigned int save_interval=100, unsigned int thread_steps=10,   \
                unsigned int max_num_levels=0, double lambda_=10.0, double beta=100.0,                                \
                double compression=exp(1.0), unsigned int seed=0, unsigned int print_thin=50)

#define RUN_BODY(name) \
    const auto opt = Options(num_particles, new_level_interval, save_interval,          \
                                thread_steps, max_num_levels, lambda_, beta, steps);    \
    Sampler<name> sampler(num_threads, compression, opt, true, false);                  \
    auto ns = static_cast<unsigned int>(sampler.size());                                \
    for (unsigned int i = 0; i < ns; i++)                                               \
    {                                                                                   \
        name *p = sampler.particle(i);                                                  \
        *p = m;                                                                         \
    }                                                                                   \
    if (seed == 0)                                                                      \
        seed = static_cast<unsigned int>(time(NULL));                                   \
    sampler.initialise(seed);                                                           \
    auto start = std::chrono::high_resolution_clock::now();                             \
    sampler.run(print_thin);                                                            \
    auto stop = std::chrono::high_resolution_clock::now();                              \
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);  \
    std::cout << "# Took " << duration.count() / 1000.0 << " seconds";

#define RUN_ARGS \
    "m"_a, "steps"_a=100, "num_threads"_a=1, "num_particles"_a=1,               \
    "new_level_interval"_a=2000, "save_interval"_a=100, "thread_steps"_a=10,    \
    "max_num_levels"_a=0, "lambda_"_a=10.0, "beta"_a=100.0,                     \
    "compression"_a=exp(1.0), "seed"_a=0, "print_thin"_a=50


NB_MODULE(Sampler, m)
{
    m.def("run", RUN_SIGNATURE(RVmodel) { RUN_BODY(RVmodel) }, RUN_ARGS, RUN_DOC);

    m.def("run", RUN_SIGNATURE(GPmodel) { RUN_BODY(GPmodel) }, RUN_ARGS, RUN_DOC);

    m.def("run", RUN_SIGNATURE(RVFWHMmodel) { RUN_BODY(RVFWHMmodel) }, RUN_ARGS, RUN_DOC);

    m.def("run", RUN_SIGNATURE(TRANSITmodel) { RUN_BODY(TRANSITmodel) }, RUN_ARGS, RUN_DOC);

    m.def("run", RUN_SIGNATURE(OutlierRVmodel) { RUN_BODY(OutlierRVmodel) }, RUN_ARGS, RUN_DOC);

    m.def("run", RUN_SIGNATURE(BINARIESmodel) { RUN_BODY(BINARIESmodel) }, RUN_ARGS, RUN_DOC);

    m.def("run", RUN_SIGNATURE(GAIAmodel) { RUN_BODY(GAIAmodel) }, RUN_ARGS, RUN_DOC);
}