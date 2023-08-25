#pragma once

#include <glob.h>  // glob(), globfree()
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "loadtxt.hpp"

// from https://stackoverflow.com/a/8615450
std::vector<std::string> glob(const std::string& pattern);


class RVData {
  private:
    std::vector<double> t, y, sig;
    std::vector<int> obsi;
    std::vector<std::vector<double>> actind;


  public:
    RVData();
    /// Get the total number of RV points
    int N() const { return t.size(); }

    // to read data from one file, one instrument
    void load(const string filename, const string units, int skip = 2, 
              const string delimiter = " ",
              const vector<string> &indicators = vector<string>());

};