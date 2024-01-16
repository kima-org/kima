#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>  // memset()
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <exception>

#include "loadtxt.hpp"

#define VERBOSE false

using namespace std;

// for nanobind
#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;
using namespace nb::literals;
#include "nb_shared.h"


template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v using std::stable_sort instead
  // of std::sort to avoid unnecessary index re-orderings when v contains
  // elements of equal values 
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

double median(vector<double> v);


class  KIMA_API RVData {

  friend class RVmodel;
  friend class GPmodel;
  friend class RVFWHMmodel;
  friend class SPLEAFmodel;
  friend class OutlierRVmodel;
  friend class BINARIESmodel;
  friend class RVGAIAmodel;

  private:
    vector<double> t, y, sig, y2, sig2;
    vector<int> obsi;
    vector<vector<double>> actind;

  public:
    string _datafile;
    string _instrument;
    vector<string> _instruments;
    vector<string> _datafiles;
    string _units;
    int _skip;
    bool _multi;
    vector<string> _indicator_names;

    RVData() {};

    RVData(const vector<string>& filenames, const string& units="ms", int skip=0, int max_rows=0, 
           const string& delimiter=" ", const vector<string>& indicators=vector<string>());

    RVData(const string& filename, const string& units="ms", int skip=0, int max_rows=0, 
           const string& delimiter=" ", const vector<string>& indicators=vector<string>());

    RVData(const vector<double> t, const vector<double> y, const vector<double> sig,
           const string& units="ms", const string& instrument="");
    
    RVData(const vector<vector<double>> t, 
           const vector<vector<double>> y, 
           const vector<vector<double>> sig,
           const string& units="ms", const vector<string>& instruments={});

    friend ostream& operator<<(ostream& os, const RVData& d);

    // to read data from one file, one instrument
    void load(const string filename, const string units, int skip=0, int max_rows=0,
              const string delimiter=" ", const vector<string> &indicators=vector<string>());

    // to read data from one file, more than one instrument
    void load_multi(const string filename, const string units, int skip=0, int max_rows=0,
                    const string delimiter=" ", const vector<string> &indicators=vector<string>());

    // to read data from more than one file, more than one instrument
    void load_multi(vector<string> filenames, const string units, int skip=0, int max_rows=0,
                    const string delimiter=" ", const vector<string>& indicators=vector<string>());

    bool indicator_correlations;
    int number_indicators;
    int number_instruments;
    
    bool sb2 {false};

    /// docs for M0_epoch
    double M0_epoch;

    /// store medians of each instrument's RVs
    vector<double> medians;


    // to deprecate a function (C++14), put
    // [[deprecated("Replaced by bar, which has an improved interface")]]
    // before the definition


    // /// Check if the data was read correctly and all vectors are the right size
    // bool check_data(bool check_for_indicator_errors = true) const;

    /// Get the total number of RV points
    int N() const { return t.size(); }

    /// Get the array of times
    const vector<double>& get_t() const { return t; }
    /// Get the array of RVs
    const vector<double>& get_y() const { return y; }
    /// Get the array of errors
    const vector<double>& get_sig() const { return sig; }
    /// Get the array of secondary RVs
    const vector<double>& get_y2() const { return y2; }
    /// Get the array of secondary errors
    const vector<double>& get_sig2() const { return sig2; }

    /// Get the mininum (starting) time
    double get_t_min() const { return *min_element(t.begin(), t.end()); }
    /// Get the maximum (ending) time
    double get_t_max() const { return *max_element(t.begin(), t.end()); }
    /// Get the timespan
    double get_timespan() const { return get_t_max() - get_t_min(); }
    double get_t_span() const { return get_t_max() - get_t_min(); }
    /// Get the middle time
    double get_t_middle() const { return get_t_min() + 0.5 * get_timespan(); }

    /// Get the mininum RV
    double get_RV_min() const { return *min_element(y.begin(), y.end()); }
    /// Get the maximum RV
    double get_RV_max() const { return *max_element(y.begin(), y.end()); }
    /// Get the RV span (peak-to-peak)
    double get_RV_span() const { return get_RV_max() - get_RV_min(); };
    /// Get the maximum RV span
    double get_max_RV_span() const;
    /// Get the mean of the RVs
    double get_RV_mean() const;
    /// Get the variance of the RVs
    double get_RV_var() const;
    /// Get the standard deviation of the RVs
    double get_RV_std() const { return sqrt(get_RV_var()); }

    /// Get the RV variance, adjusted for multiple instruments
    double get_adjusted_RV_var() const;
    /// Get the RV standard deviation, adjusted for multiple instruments
    double get_adjusted_RV_std() const { return sqrt(get_adjusted_RV_var()); }

    /// Get the maximum slope allowed by the data
    double topslope() const;
    /// Order of magnitude of trend coefficient (of degree) given the data
    int get_trend_magnitude(int degree) const;

    /// Get the array of activity indictators
    const vector<vector<double>>& get_actind() const { return actind; }

    /// Get the array of instrument identifiers
    const vector<int>& get_obsi() const { return obsi; }

    /// Get the number of instruments
    int Ninstruments() const
    {
        set<int> s(obsi.begin(), obsi.end());
        return s.size();
    }

  //  private:
  //   // Singleton
  //   static RVData instance;

  //  public:
  //   static RVData& get_instance() { return instance; }
};




class  PHOTdata {

  friend class TRANSITmodel;


  private:
    vector<double> t, y, sig;
    // vector<int> obsi;
    // vector<vector<double>> actind;

  public:

    string _datafile;
    vector<string> _datafiles;
    string _units;
    int _skip;

    PHOTdata();
    PHOTdata(const string& filename, const string& units="ms", int skip=0, const string& delimiter=" ")
    {
      load(filename, units, skip, delimiter);
    }

    friend ostream& operator<<(ostream& os, const PHOTdata& d);

    // to read data from one file, one instrument
    void load(const string filename, const string units, int skip=0, 
              const string delimiter=" ");

    /// docs for M0_epoch
    double M0_epoch;

    /// Get the total number of RV points
    int N() const { return t.size(); }

    /// Get the array of times
    const vector<double>& get_t() const { return t; }
    /// Get the array of RVs
    const vector<double>& get_y() const { return y; }
    /// Get the array of errors
    const vector<double>& get_sig() const { return sig; }

    /// Get the mininum (starting) time
    double get_t_min() const { return *min_element(t.begin(), t.end()); }
    /// Get the maximum (ending) time
    double get_t_max() const { return *max_element(t.begin(), t.end()); }
    /// Get the timespan
    double get_timespan() const { return get_t_max() - get_t_min(); }
    double get_t_span() const { return get_t_max() - get_t_min(); }
    /// Get the middle time
    double get_t_middle() const { return get_t_min() + 0.5 * get_timespan(); }

    /// Get the mininum flux
    double get_flux_min() const { return *min_element(y.begin(), y.end()); }
    /// Get the maximum flux
    double get_flux_max() const { return *max_element(y.begin(), y.end()); }
    /// Get the flux span (peak-to-peak)
    double get_flux_span() const { return get_flux_max() - get_flux_min(); };
    /// Get the mean of the fluxs
    double get_flux_mean() const;
    /// Get the variance of the fluxs
    double get_flux_var() const;
    /// Get the standard deviation of the fluxs
    double get_flux_std() const { return sqrt(get_flux_var()); }

    /// Get the maximum slope allowed by the data
    double topslope() const;
    /// Order of magnitude of trend coefficient (of degree) given the data
    int get_trend_magnitude(int degree) const;

};

class KIMA_API GAIAData {

  friend class GAIAmodel;
  friend class RVGAIAmodel;

  private:
    vector<double> t, w, wsig, psi, pf;


  public:
    string _datafile;
    string _units;
    int _skip;

    GAIAData();
    GAIAData(const string& filename, const string& units="mas", int skip=0, int max_rows=0, 
            const string& delimiter=" ")
    {
      load(filename, units, skip, max_rows, delimiter);
    }

    friend ostream& operator<<(ostream& os, const GAIAData& d);

    // to read data from one file, one instrument
    void load(const string filename, const string units, int skip=0, int max_rows=0,
              const string delimiter=" ");


    /// docs for M0_epoch
    double M0_epoch;

    // to deprecate a function (C++14), put
    // [[deprecated("Replaced by bar, which has an improved interface")]]
    // before the definition


    // /// Check if the data was read correctly and all vectors are the right size
    // bool check_data(bool check_for_indicator_errors = true) const;

    /// Get the total number of Gaia points
    int N() const { return t.size(); }

    /// Get the array of times
    const vector<double>& get_t() const { return t; }
    /// Get the array of RVs
    const vector<double>& get_w() const { return w; }
    /// Get the array of errors
    const vector<double>& get_psi() const { return psi; }
    /// Get the array of secondary RVs
    const vector<double>& get_wsig() const { return wsig; }
    /// Get the array of secondary errors
    const vector<double>& get_pf() const { return pf; }

    /// Get the mininum (starting) time
    double get_t_min() const { return *min_element(t.begin(), t.end()); }
    /// Get the maximum (ending) time
    double get_t_max() const { return *max_element(t.begin(), t.end()); }
    /// Get the timespan
    double get_timespan() const { return get_t_max() - get_t_min(); }
    double get_t_span() const { return get_t_max() - get_t_min(); }
    /// Get the middle time
    double get_t_middle() const { return get_t_min() + 0.5 * get_timespan(); }


  //  private:
  //   // Singleton
  //   static RVData instance;

  //  public:
  //   static RVData& get_instance() { return instance; }
};

