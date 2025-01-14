#pragma once

#include <iostream>
#include <map>
#include <string>

#include "DNest4.h"
#include "Data.h"
#include "utils.h"
using distribution = std::shared_ptr<DNest4::ContinuousDistribution>;


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

// distribution get_default_prior(std::string name);