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
#include <filesystem>
#include <climits>

#include "loadtxt.hpp"

#define VERBOSE false

using namespace std;
namespace fs = std::filesystem;

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

template <typename T>
T read_value(std::ifstream &file);

template <typename T>
T swap_endian(T u);


class KIMA_API RVData {

  friend class RVmodel;
  friend class GPmodel;
  friend class RVFWHMmodel;
  friend class SPLEAFmodel;
  friend class OutlierRVmodel;
  friend class BINARIESmodel;
  friend class RVGAIAmodel;
  friend class RVFWHMRHKmodel;
  friend class ApodizedRVmodel;
  friend class RVHGPMmodel;

  private:
    vector<double> t, y, sig, y2, sig2;
    vector<int> obsi;
    vector<vector<double>> actind;
    vector<vector<double>> normalized_actind;

  public:
    string _datafile;
    string _instrument;
    vector<string> _instruments;
    vector<string> _datafiles;
    string _units;
    int _skip;
    bool _multi;
    vector<string> _indicator_names;
    vector<double> _unique_times;
    vector<int> _inverse_indices;

    RVData() {};

    // read data from a list of files
    RVData(const vector<string>& filenames, const string& units="ms", int skip=0, int max_rows=0,
           const string& delimiter=" \t,", const vector<string>& indicators=vector<string>(), bool double_lined=false);

    // read data from a single file
    RVData(const string& filename, const string& units="ms", int skip=0, int max_rows=0, bool multi=false,
           const string& delimiter=" \t,", const vector<string>& indicators=vector<string>(), bool double_lined=false);

    // read data from arrays
    RVData(const vector<double> t, const vector<double> y, const vector<double> sig,
           const string& units="ms", const string& instrument="");
    
    // read data from arrays for multiple instruments
    RVData(const vector<vector<double>> t, 
           const vector<vector<double>> y, 
           const vector<vector<double>> sig,
           const string& units="ms", const vector<string>& instruments={});

    friend ostream& operator<<(ostream& os, const RVData& d);

    // to read data from one file, one instrument
    void load(const string filename, const string units, int skip=0, int max_rows=0,
              const string delimiter=" \t,", const vector<string> &indicators=vector<string>());

    // to read data from one file, more than one instrument
    void load_multi(const string filename, const string units, int skip=0, int max_rows=0,
                    const string delimiter=" \t,", const vector<string> &indicators=vector<string>());

    // to read data from more than one file, more than one instrument
    void load_multi(vector<string> filenames, const string units, int skip=0, int max_rows=0,
                    const string delimiter=" \t,", const vector<string>& indicators=vector<string>());

    bool indicator_correlations;
    int number_indicators;
    int number_instruments;
    
    bool sb2 {false};

    /// docs for M0_epoch
    double M0_epoch;

    double trend_epoch;

    /// store medians of each instrument's RVs
    vector<double> medians;


    // to deprecate a function (C++14), put
    // [[deprecated("Replaced by bar, which has an improved interface")]]
    // before the definition


    // /// Check if the data was read correctly and all vectors are the right size
    // bool check_data(bool check_for_indicator_errors = true) const;

    /// Get the total number of RV points
    int N() const { return static_cast<int>(t.size()); }

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
    int get_trend_magnitude(int degree, int i=-1) const;

    /// Get the array of activity indictators
    const vector<vector<double>>& get_actind() const { return actind; }

    /// Get the mininum value of Activity Indicator i
    double get_actind_min(size_t i) const { return *min_element(actind.at(i).begin(), actind.at(i).end()); }
    /// Get the maximum value of Activity Indicator i
    double get_actind_max(size_t i) const { return *max_element(actind.at(i).begin(), actind.at(i).end()); }
    /// Get the span of Activity Indicator i
    double get_actind_span(size_t i) const { return get_actind_max(i) - get_actind_min(i); }
    /// Get the mean of Activity Indicator i
    double get_actind_mean(size_t i) const;
    /// Get the variance of Activity Indicator i
    double get_actind_var(size_t i) const;
    /// Get the standard deviation of Activity Indicator i
    double get_actind_std(size_t i) const { return sqrt(get_actind_var(i)); }


    /// Normalize the activity indicators from 0 to 1
    void normalize_actind();

    /// Get the array of normalized activity indictators
    const vector<vector<double>>& get_normalized_actind() const { return normalized_actind; }

    /// Get the array of instrument identifiers
    const vector<int>& get_obsi() const { return obsi; }

    /// Get the vector of unique times
    const vector<double>& get_unique_t() {
        if (_unique_times.size() == 0) {
          _unique_times = t;
          std::sort(_unique_times.begin(), _unique_times.end());
          _unique_times.erase(std::unique(_unique_times.begin(), _unique_times.end()), _unique_times.end());
        }
        return _unique_times;
    }

    //
    const vector<int>& _inverse_time_indices() {
      _unique_times = get_unique_t();
      if (_inverse_indices.size() == 0) {
        _inverse_indices.reserve(t.size());
        for (const double& tt : t) {
          auto it = std::find(_unique_times.begin(), _unique_times.end(), tt);
          if (it != _unique_times.end()) {
            auto index = std::distance(_unique_times.begin(), it);
            _inverse_indices.push_back(index);
          }
        }
      }
      return _inverse_indices;
    }

    /// Get the number of instruments
    int Ninstruments() const
    {
        set<int> s(obsi.begin(), obsi.end());
        return static_cast<int>(s.size());
    }

  //  private:
  //   // Singleton
  //   static RVData instance;

  //  public:
  //   static RVData& get_instance() { return instance; }
};




class KIMA_API PHOTdata {

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

    PHOTdata() {};
    PHOTdata(const string& filename, int skip=0, const string& delimiter=" \t,")
    {
      load(filename, skip, delimiter);
    }

    friend ostream& operator<<(ostream& os, const PHOTdata& d);

    // to read data from one file, one instrument
    void load(const string filename, int skip=0, const string delimiter=" \t,");

    /// docs for M0_epoch
    double M0_epoch;

    /// Get the total number of RV points
    int N() const { return static_cast<int>(t.size()); }

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

class KIMA_API GAIAdata {

  friend class GAIAmodel;
  friend class RVGAIAmodel;

  private:
    vector<double> t, w, wsig, psi, pf;


  public:
    string _datafile;
    string _units;
    int _skip;

    GAIAdata();
    GAIAdata(const string& filename, const string& units="mas", int skip=0, int max_rows=0, 
            const string& delimiter=" \t,")
    {
      load(filename, units, skip, max_rows, delimiter);
    }

    friend ostream& operator<<(ostream& os, const GAIAdata& d);

    // to read data from one file, one instrument
    void load(const string filename, const string units, int skip=0, int max_rows=0,
              const string delimiter=" \t,");


    /// docs for M0_epoch
    double M0_epoch;

    // to deprecate a function (C++14), put
    // [[deprecated("Replaced by bar, which has an improved interface")]]
    // before the definition


    // /// Check if the data was read correctly and all vectors are the right size
    // bool check_data(bool check_for_indicator_errors = true) const;

    /// Get the total number of Gaia points
    int N() const { return static_cast<int>(t.size()); }

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


// hold data from the Hipparcos-Gaia catalog of accelerations (Brandt 2021)
struct hgca_data {
    bool found;
    size_t row;
    uint64_t gaia_id;
    double gaia_ra, gaia_dec;
    float radial_velocity, radial_velocity_error;
    float parallax_gaia, parallax_gaia_error;
    float pmra_gaia, pmdec_gaia;
    float pmra_gaia_error, pmdec_gaia_error, pmra_pmdec_gaia;
    float pmra_hg, pmdec_hg;
    float pmra_hg_error, pmdec_hg_error, pmra_pmdec_hg;
    float pmra_hip, pmdec_hip;
    float pmra_hip_error, pmdec_hip_error, pmra_pmdec_hip;
    double epoch_ra_gaia, epoch_dec_gaia, epoch_ra_hip, epoch_dec_hip;
    float crosscal_pmra_hip, crosscal_pmdec_hip;
    float crosscal_pmra_hg, crosscal_pmdec_hg;
    float nonlinear_dpmra, nonlinear_dpmdec;
    float chisq;
};

class KIMA_API HGPMdata {

  friend class RVHGPMmodel;

  public:
    HGPMdata();
    HGPMdata(uint64_t gaia_id) { load(gaia_id); };
    void load(uint64_t gaia_id);
    hgca_data get_data(uint64_t target_id);

    // store the path to a temporary directory, which will be accessible in the
    // Python bindings
    inline static std::string temp_path = fs::temp_directory_path().string();

    uint64_t gaia_id;
    double parallax_gaia, parallax_gaia_error;
    double epoch_ra_hip, epoch_dec_hip;   // epochs for Hipparcos proper motions
    double epoch_ra_gaia, epoch_dec_gaia; // epochs for Gaia proper motions
    // proper motion measurements
    double pm_ra_hip, pm_dec_hip, pm_ra_gaia, pm_dec_gaia, pm_ra_hg, pm_dec_hg;
    // Hipparcos measurement uncertainties and correlation
    double sig_hip_ra, sig_hip_dec, rho_hip;
    // Hipparcos - Gaia measurement uncertainties and correlation
    double sig_hg_ra, sig_hg_dec, rho_hg;
    // Gaia measurement uncertainties and correlation
    double sig_gaia_ra, sig_gaia_dec, rho_gaia;
    // chi square value
    double chisq;

    vector<double> get_epochs(size_t n_average=1) const;

};  

class ETVData {

  friend class ETVmodel;
  private:
    vector<double> epochs, et, etsig, y2, sig2;

  public:
  
    string _datafile;
    string _units;
    int _skip;
    
    ETVData();
    ETVData(const string& filename, const string& units="days", int skip=0, int max_rows=0, 
            const string& delimiter=" \t,")
    {
      load(filename, units, skip, max_rows, delimiter);
    }
    
    friend ostream& operator<<(ostream& os, const ETVData& d);
    
    // to read data from one file, one instrument
    void load(const string filename, const string units, int skip=0, int max_rows=0,
              const string delimiter=" \t,");

    /// docs for M0_epoch
    double M0_epoch;

    /// Get the total number of data points
    int N() const { return static_cast<int>(epochs.size()); }

    /// @brief Get the array of epoch @return const vector<double>&
    const vector<double>& get_epochs() const { return epochs; }

    /// @brief Get the array of RVs @return const vector<double>&
    const vector<double>& get_et() const { return et; }
    const vector<double>& get_y2() const { return y2; }
    /// Get the array of errors @return const vector<double>&
    const vector<double>& get_etsig() const { return etsig; }
    const vector<double>& get_sig2() const { return sig2; }

    /// @brief Get the mininum (starting) time @return double
    double get_et_min() const { return *min_element(et.begin(), et.end()); }
    /// @brief Get the maximum (ending) time @return double
    double get_et_max() const { return *max_element(et.begin(), et.end()); }
    /// @brief Get the timespan @return double
    double get_timespan() const { return get_et_max() - get_et_min(); }
    double get_et_span() const { return get_et_max() - get_et_min(); }
    /// @brief Get the middle time @return double
    double get_et_middle() const { return get_et_min() + 0.5 * get_timespan(); }

    /// @brief Get the mininum RV @return double
    double get_epoch_min() const { return *min_element(epochs.begin(), epochs.end()); }
    /// @brief Get the maximum RV @return double
    double get_epoch_max() const { return *max_element(epochs.begin(), epochs.end()); }
    /// @brief Get the RV span @return double

    /// @brief Get the mininum y2 @return double
    double get_y2_min() const { return *min_element(y2.begin(), y2.end()); }
    /// @brief Get the maximum y2 @return double
    double get_y2_max() const { return *max_element(y2.begin(), y2.end()); }
    /// @brief Get the y2 span @return double
    double get_y2_span() const { return get_y2_max() - get_y2_min(); }

};