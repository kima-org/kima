#include <ctime>

#include <nanobind/nanobind.h>
// #include <nanobind/stl/string.h>
// #include <nanobind/stl/vector.h>
namespace nb = nanobind;
using namespace nb::literals;

#include "RVmodel.h"
#include "GPmodel.h"
#include "RVFWHMmodel.h"
#include "SPLEAFmodel.h"
#include "OutlierRVmodel.h"


auto RUN_DOC = R"D(
Run the DNest4 sampler with the given model

Args:
    m (model): the model
    steps (int, default=100): how many steps to run for
)D";




NB_MODULE(Sampler, m)
{
    m.def(
        "run", [](RVmodel &m, int steps=100, int num_threads=1, int num_particles=1, 
                  int new_level_interval=2000, int save_interval=100, int thread_steps=10,
                  int max_num_levels=0, double lambda=10.0, double beta=100.0,
                  double compression=exp(1.0), unsigned int seed=0, unsigned int print_thin=50)
        {
            // setup the sampler options
            auto opt = Options(num_particles, new_level_interval, save_interval,
                               thread_steps, max_num_levels, lambda, beta, steps);
            // create the sampler
            Sampler<RVmodel> sampler(num_threads, compression, opt, true, false);
            // replace default particles with provided model
            for (size_t i = 0; i < sampler.size(); i++)
            {
                RVmodel *p = sampler.particle(i);
                *p = m;
            }
            if (seed == 0)
                seed = static_cast<unsigned int>(time(NULL));
            sampler.initialise(seed);
            sampler.run(print_thin);
        },
        "m"_a, "steps"_a=100, "num_threads"_a=1, "num_particles"_a=1,
        "new_level_interval"_a=2000, "save_interval"_a=100, "thread_steps"_a=10,
        "max_num_levels"_a=0, "lambda"_a=10.0, "beta"_a=100.0,
        "compression"_a=exp(1.0), "seed"_a=0, "print_thin"_a=50,
        RUN_DOC);

    m.def(
        "run", [](GPmodel &m, int steps=100, int num_threads=1, int num_particles=1, 
                  int new_level_interval=2000, int save_interval=100, int thread_steps=10,
                  int max_num_levels=0, double lambda=10.0, double beta=100.0,
                  double compression=exp(1.0), unsigned int seed=0, unsigned int print_thin=50)
        {
            // setup the sampler options
            auto opt = Options(num_particles, new_level_interval, save_interval,
                               thread_steps, max_num_levels, lambda, beta, steps);
            // create the sampler
            Sampler<GPmodel> sampler(num_threads, compression, opt, true, false);
            // replace default particles with provided model
            for (size_t i = 0; i < sampler.size(); i++)
            {
                GPmodel *p = sampler.particle(i);
                *p = m;
            }
            if (seed == 0)
                seed = static_cast<unsigned int>(time(NULL));
            sampler.initialise(seed);
            sampler.run(print_thin);
        },
        "m"_a, "steps"_a=100, "num_threads"_a=1, "num_particles"_a=1,
        "new_level_interval"_a=2000, "save_interval"_a=100, "thread_steps"_a=10,
        "max_num_levels"_a=0, "lambda"_a=10.0, "beta"_a=100.0,
        "compression"_a=exp(1.0), "seed"_a=0, "print_thin"_a=50,
        RUN_DOC);
    
    m.def(
        "run", [](RVFWHMmodel &m, int steps=100, int num_threads=1, int num_particles=1, 
                  int new_level_interval=2000, int save_interval=100, int thread_steps=10,
                  int max_num_levels=0, double lambda=10.0, double beta=100.0,
                  double compression=exp(1.0), unsigned int seed=0, unsigned int print_thin=50)
        {
            // setup the sampler options
            auto opt = Options(num_particles, new_level_interval, save_interval,
                               thread_steps, max_num_levels, lambda, beta, steps);
            // create the sampler
            Sampler<RVFWHMmodel> sampler(num_threads, compression, opt, true, false);
            // replace default particles with provided model
            for (size_t i = 0; i < sampler.size(); i++)
            {
                RVFWHMmodel *p = sampler.particle(i);
                *p = m;
            }
            if (seed == 0)
                seed = static_cast<unsigned int>(time(NULL));
            sampler.initialise(seed);
            sampler.run(print_thin);
        },
        "m"_a, "steps"_a=100, "num_threads"_a=1, "num_particles"_a=1,
        "new_level_interval"_a=2000, "save_interval"_a=100, "thread_steps"_a=10,
        "max_num_levels"_a=0, "lambda"_a=10.0, "beta"_a=100.0,
        "compression"_a=exp(1.0), "seed"_a=0, "print_thin"_a=50,
        RUN_DOC);

    m.def(
        "run", [](SPLEAFmodel &m, int steps=100, int num_threads=1, int num_particles=1, 
                  int new_level_interval=2000, int save_interval=100, int thread_steps=10,
                  int max_num_levels=0, double lambda=10.0, double beta=100.0,
                  double compression=exp(1.0), unsigned int seed=0, unsigned int print_thin=50)
        {
            // setup the sampler options
            auto opt = Options(num_particles, new_level_interval, save_interval,
                               thread_steps, max_num_levels, lambda, beta, steps);
            // create the sampler
            Sampler<SPLEAFmodel> sampler(num_threads, compression, opt, true, false);
            // replace default particles with provided model
            for (size_t i = 0; i < sampler.size(); i++)
            {
                SPLEAFmodel *p = sampler.particle(i);
                *p = m;
            }
            if (seed == 0)
                seed = static_cast<unsigned int>(time(NULL));
            sampler.initialise(seed);
            sampler.run(print_thin);
        },
        "m"_a, "steps"_a=100, "num_threads"_a=1, "num_particles"_a=1,
        "new_level_interval"_a=2000, "save_interval"_a=100, "thread_steps"_a=10,
        "max_num_levels"_a=0, "lambda"_a=10.0, "beta"_a=100.0,
        "compression"_a=exp(1.0), "seed"_a=0, "print_thin"_a=50,
        RUN_DOC);

    m.def(
        "run", [](OutlierRVmodel &m, int steps=100, int num_threads=1, int num_particles=1, 
                  int new_level_interval=2000, int save_interval=100, int thread_steps=10,
                  int max_num_levels=0, double lambda=10.0, double beta=100.0,
                  double compression=exp(1.0), unsigned int seed=0, unsigned int print_thin=50)
        {
            // setup the sampler options
            auto opt = Options(num_particles, new_level_interval, save_interval,
                               thread_steps, max_num_levels, lambda, beta, steps);
            // create the sampler
            Sampler<OutlierRVmodel> sampler(num_threads, compression, opt, true, false);
            // replace default particles with provided model
            for (size_t i = 0; i < sampler.size(); i++)
            {
                OutlierRVmodel *p = sampler.particle(i);
                *p = m;
            }
            if (seed == 0)
                seed = static_cast<unsigned int>(time(NULL));
            sampler.initialise(seed);
            sampler.run(print_thin);
        },
        "m"_a, "steps"_a=100, "num_threads"_a=1, "num_particles"_a=1,
        "new_level_interval"_a=2000, "save_interval"_a=100, "thread_steps"_a=10,
        "max_num_levels"_a=0, "lambda"_a=10.0, "beta"_a=100.0,
        "compression"_a=exp(1.0), "seed"_a=0, "print_thin"_a=50,
        RUN_DOC);

}