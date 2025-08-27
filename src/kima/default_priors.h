#pragma once

#include <iostream>
#include <map>
#include <string>
#include <unordered_map>

#include "DNest4.h"
#include "Data.h"
#include "utils.h"
#include "distributions.h"

using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;

// for nanobind
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
namespace nb = nanobind;
using namespace nb::literals;

class DefaultPriors
{
    protected:
        const RVData &data;
        std::unordered_map<std::string, distribution> default_mapping;
    
    public:
        DefaultPriors() = delete;
        DefaultPriors(const RVData &data);
        distribution get(std::string name);
        void print();
    
};
