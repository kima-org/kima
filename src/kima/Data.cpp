#include "Data.h"


double median(vector<double> v)
{
  if(v.empty()) {
    return 0.0;
  }
  auto n = v.size() / 2;
  nth_element(v.begin(), v.begin()+n, v.end());
  auto med = v[n];
  if(!(v.size() & 1)) { //If the set size is even
    auto max_it = max_element(v.begin(), v.begin()+n);
    med = (*max_it + med) / 2.0;
  }
  return med;    
}


/// @brief Load data from a single filename
RVData::RVData(const string& filename, const string& units, int skip, int max_rows, 
               const string& delimiter, const vector<string>& indicators)
{
    load(filename, units, skip, max_rows, delimiter, indicators);
}

/// @brief Load data from a list of filenames
RVData::RVData(const vector<string>& filenames, const string& units, int skip, int max_rows, 
               const string& delimiter, const vector<string>& indicators)
{
    load_multi(filenames, units, skip, max_rows, delimiter, indicators);
}

/// @brief Load data from vectors directly
RVData::RVData(const vector<double> _t, const vector<double> _y, const vector<double> _sig,
               const string& units, const string& instrument)
: t(_t), y(_y), sig(_sig)
{
    obsi = vector<int>(t.size(), 1);
    actind.clear();

    _datafile = "";
    _datafiles = {};
    _units = units;
    _skip = 0;
    _multi = false;
    _indicator_names = {};
    number_indicators = 0;
    number_instruments = 1;
    _instrument = instrument;
    _instruments = {};

    if (units == "kms")
    {
        for (size_t i = 0; i < N(); i++)
        {
            y[i] *= 1e3;
            sig[i] *= 1e3;
        }
    }

    // epoch for the mean anomaly, by default the time of the first observation
    M0_epoch = t[0];

    // How many points did we read?
    if (VERBOSE)
        printf("# Loaded %zu data points from arrays\n", t.size());

    // What are the units?
    if (units == "kms" && VERBOSE)
        printf("# Multiplied all RVs by 1000; units are now m/s.\n");

}

/// @brief Load data from vectors directly, for multiple instruments
RVData::RVData(const vector<vector<double>> _t, 
               const vector<vector<double>> _y, 
               const vector<vector<double>> _sig,
               const string& units, const vector<string>& instruments)
{
    t.clear();
    y.clear();
    sig.clear();

    y2.clear();
    sig2.clear();

    medians.clear();

    if (_t.size() != _y.size()) 
    {
        string msg = "RVData: data arrays must have the same size size(t) != size(y)";
        throw invalid_argument(msg);
    }
    if (_t.size() != _sig.size()) 
    {
        string msg = "RVData: data arrays must have the same size size(t) != size(sig)";
        throw invalid_argument(msg);
    }

    for (size_t i = 0; i < _t.size(); i++)
    {
        t.insert(t.end(), _t[i].begin(), _t[i].end());
        y.insert(y.end(), _y[i].begin(), _y[i].end());
        sig.insert(sig.end(), _sig[i].begin(), _sig[i].end());

        // store medians
        medians.push_back(median(_y[i]));

        for (size_t n = 0; n < _t[i].size(); n++)
            obsi.push_back(i + 1);
    }
    actind.clear();

    _datafile = "";
    _datafiles = {};
    _units = units;
    _skip = 0;
    _multi = true;
    _indicator_names = {};
    number_indicators = 0;
    number_instruments = _t.size();
    _instrument = "";
    _instruments = instruments;

    if (units == "kms")
    {
        for (size_t i = 0; i < N(); i++)
        {
            y[i] *= 1e3;
            sig[i] *= 1e3;
        }
    }

    // epoch for the mean anomaly, by default the time of the first observation
    M0_epoch = t[0];

    // How many points did we read?
    if (VERBOSE)
        printf("# Loaded %zu data points from arrays\n", t.size());

    // What are the units?
    if (units == "kms" && VERBOSE)
        printf("# Multiplied all RVs by 1000; units are now m/s.\n");

    // We need to sort t because it comes from different instruments
    if (number_instruments > 1) {
        size_t N = t.size();
        vector<double> tt(N), yy(N);
        vector<double> sigsig(N), obsiobsi(N);
        vector<int> order(N);

        // order = argsort(t)
        int x = 0;
        std::iota(order.begin(), order.end(), x++);
        sort(order.begin(), order.end(),
            [&](int i, int j) { return t[i] < t[j]; });

        for (size_t i = 0; i < N; i++) {
            tt[i] = t[order[i]];
            yy[i] = y[order[i]];
            sigsig[i] = sig[order[i]];
            obsiobsi[i] = obsi[order[i]];
        }

        for (size_t i = 0; i < N; i++) {
            t[i] = tt[i];
            y[i] = yy[i];
            sig[i] = sigsig[i];
            obsi[i] = obsiobsi[i];
        }
    }

}



/**
 * @brief Load RV data from a file.
 *
 * Read a tab/space separated file with columns
 * ```
 *   time  vrad  error  quant  error
 *   ...   ...   ...    ...    ...
 * ```
 *
 * @param filename   the name of the file
 * @param units      units of the RVs and errors, either "kms" or "ms"
 * @param skip       number of lines to skip in the beginning of the file (default = 2)
 * @param indicators
 */
void RVData::load(const string filename, const string units, int skip, int max_rows,
                    const string delimiter, const vector<string>& indicators)
{
    if (filename.empty()) {
        std::string msg = "kima: RVData: no filename provided";
        throw std::invalid_argument(msg);
        // exit(1);
    }

    if (filename.size() == 1) {
        std::string msg = "kima: RVData: filename with one character is probably an error";
        throw std::runtime_error(msg);
    }

    auto data = loadtxt(filename)
                    .skiprows(skip)
                    .max_rows(max_rows)
                    .delimiter(delimiter)();

    if (data.size() < 3) {
        std::string msg = "kima: RVData: file (" + filename + ") contains less than 3 columns! (is skip correct?)";
        throw std::runtime_error(msg);
    }
    
    _datafile = filename;
    _datafiles = {};
    _units = units;
    _skip = skip;
    _multi = false;
    number_instruments = 1;

    t = data[0];
    y = data[1];
    sig = data[2];

    obsi = vector<int>(t.size(), 1);
    
    if (sb2)
    {
        y2 = data[3];
        sig2 = data[4];
    }

    // check for indicator correlations and store stuff
    int nempty = count(indicators.begin(), indicators.end(), "");
    number_indicators = indicators.size() - nempty;
    indicator_correlations = number_indicators > 0;

    _indicator_names = indicators;
    // _indicator_names.erase(
    //     std::remove(_indicator_names.begin(), _indicator_names.end(), ""),
    //     _indicator_names.end()
    // );

    // empty and resize the indicator vectors
    actind.clear();
    actind.resize(number_indicators);
    for (int n = 0; n < number_indicators; n++)
        actind[n].clear();

    // set the indicator vectors to the right columns
    if (indicator_correlations)
    {
        int j = 0;
        for (size_t i = 0; i < number_indicators + nempty; i++)
        {
            if (indicators[i] == "")
                continue; // skip column
            else
            {
                actind[j] = data[3 + i];
                j++;
            }
        }
    }

    double factor = 1.;
    if (units == "kms") factor = 1E3;

    for (size_t n = 0; n < t.size(); n++) {
        y[n] = y[n] * factor;
        sig[n] = sig[n] * factor;
        if (sb2)
        {
            y2[n] = y2[n] * factor;
            sig2[n] = sig2[n] * factor;
        }
    }

    // epoch for the mean anomaly, by default the time of the first observation
    M0_epoch = t[0];

    // How many points did we read?
    if (VERBOSE)
        printf("# Loaded %zu data points from file %s\n", t.size(), filename.c_str());

    // What are the units?
    if (units == "kms" && VERBOSE)
        printf("# Multiplied all RVs by 1000; units are now m/s.\n");
}

/**
 * @brief Load RV data from a multi-instrument file.
 *
 * Read a tab/space separated file with columns
 * ```
 *   time  vrad  error  ...  obs
 *   ...   ...   ...    ...  ...
 * ```
 * The `obs` column should be an integer identifying the instrument.
 *
 * @param filename   the name of the file
 * @param units      units of the RVs and errors, either "kms" or "ms"
 * @param skip       number of lines to skip in the beginning of the file (default = 2)
 */
void RVData::load_multi(const string filename, const string units, int skip, int max_rows,
                        const string delimiter, const vector<string> &indicators)
{

    auto data = loadtxt(filename)
                    .skiprows(skip)
                    .max_rows(max_rows)
                    .delimiter(delimiter)();

    if (data.size() < 4) {
        printf("Data file (%s) contains less than 4 columns!\n", filename.c_str());
        exit(1);
    }

    auto Ncol = data.size();
    auto N = data[0].size();

    _datafile = filename;
    _datafiles = {};
    _units = units;
    _skip = skip;
    _multi = true;

    t = data[0];
    y = data[1];
    sig = data[2];
    
    if (sb2)
    {
        y2 = data[3];
        sig2 = data[4];
    }

    // check for indicator correlations and store stuff
    int nempty = count(indicators.begin(), indicators.end(), "");
    number_indicators = indicators.size() - nempty;
    indicator_correlations = number_indicators > 0;
    _indicator_names = indicators;
    // indicator_names.erase(
    //     std::remove(indicator_names.begin(), indicator_names.end(), ""),
    //     indicator_names.end());

    // empty and resize the indicator vectors
    actind.clear();
    actind.resize(number_indicators);
    for (int n = 0; n < number_indicators; n++)
        actind[n].clear();

    // set the indicator vectors to the right columns
    if (indicator_correlations)
    {
        int j = 0;
        for (size_t i = 0; i < number_indicators + nempty; i++)
        {
            if (indicators[i] == "")
                continue; // skip column
            else
            {
                actind[j] = data[3 + i];
                j++;
            }
        }
    }

    double factor = 1.;
    if (units == "kms") factor = 1E3;
    for (size_t n = 0; n < t.size(); n++) {
        y[n] = y[n] * factor;
        sig[n] = sig[n] * factor;
        if (sb2)
        {
            y2[n] = y2[n] * factor;
            sig2[n] = sig2[n] * factor;
        }
    }

    // the 4th column of the file identifies the instrument; it can have "0"s
    // this is to make sure the obsi vector always starts at 1, to avoid
    // segmentation faults later
    vector<int> inst_id;
    inst_id.push_back(data[Ncol - 1][0]);

    for (size_t n = 1; n < N; n++) {
        if (data[Ncol - 1][n] != inst_id.back()) {
            inst_id.push_back(data[Ncol - 1][n]);
        }
    }
    int id_offset = *min_element(inst_id.begin(), inst_id.end());

    obsi.clear();
    for (unsigned n = 0; n < N; n++) {
        obsi.push_back(data[Ncol - 1][n] - id_offset + 1);
    }

    // How many points did we read?
    if (VERBOSE)
        printf("# Loaded %zu data points from file %s\n", t.size(), filename.c_str());

    // Of how many instruments?
    set<int> s(obsi.begin(), obsi.end());
    number_instruments = s.size();
    if (VERBOSE)
        printf("# RVs come from %zu different instruments.\n", s.size());

    if (units == "kms" && VERBOSE)
        cout << "# Multiplied all RVs by 1000; units are now m/s." << endl;

    // epoch for the mean anomaly, by default the time of the first observation
    M0_epoch = t[0];
}

/**
 * @brief Load RV data from a multiple files.
 *
 * Read a tab/space separated files, each with columns
 * ```
 *   time  vrad  error
 *   ...   ...   ...
 * ```
 * All files should have the same structure and values in the same units.
 *
 * @param filenames  the names of the files
 * @param units      units of the RVs and errors, either "kms" or "ms"
 * @param skip       number of lines to skip in the beginning of the file (default = 2)
 * @param indicators
 */
void RVData::load_multi(vector<string> filenames, const string units, int skip, int max_rows,
                        const string delimiter, const vector<string>& indicators)
{
    if (filenames.empty()) {
        std::string msg = "kima: RVData: no filenames provided";
        throw std::invalid_argument(msg);
    }

    t.clear();
    y.clear();
    sig.clear();
    y2.clear();
    sig2.clear();
    obsi.clear();
    medians.clear();

    // check for indicator correlations and store stuff
    int nempty = count(indicators.begin(), indicators.end(), "");
    number_indicators = indicators.size() - nempty;
    indicator_correlations = number_indicators > 0;
    _indicator_names = indicators;
    // indicator_names.erase(
    //     std::remove(indicator_names.begin(), indicator_names.end(), ""),
    //     indicator_names.end());

    // empty and resize the indicator vectors
    actind.clear();
    actind.resize(number_indicators);
    for (int n = 0; n < number_indicators; n++)
        actind[n].clear();


    int filecount = 1;
    for (auto& filename : filenames) {
        auto data = loadtxt(filename).skiprows(skip)();

        if (data.size() < 3) {
            std::string msg = "kima: RVData: file (" + filename + ") contains less than 3 columns! (is skip correct?)";
            throw std::runtime_error(msg);
        }

        t.insert(t.end(), data[0].begin(), data[0].end());
        y.insert(y.end(), data[1].begin(), data[1].end());
        sig.insert(sig.end(), data[2].begin(), data[2].end());

        // store medians
        medians.push_back(median(data[1]));

        if (sb2)
        {
            y2.insert(y2.end(), data[3].begin(), data[3].end());
            sig2.insert(sig2.end(), data[4].begin(), data[4].end());
        }

        // set the indicator vectors to the right columns
        if (indicator_correlations)
        {
            int j = 0;
            for (size_t i = 0; i < number_indicators + nempty; i++)
            {
                if (indicators[i] == "")
                    continue; // skip column
                else
                {
                    actind[j].insert(actind[j].end(), 
                                        data[3 + i].begin(), 
                                        data[3 + i].end());
                    j++;
                }
            }
        }

        for (size_t n = 0; n < data[0].size(); n++)
            obsi.push_back(filecount);
        filecount++;
    }

    double factor = 1.;
    if (units == "kms") factor = 1E3;

    for (size_t n = 0; n < t.size(); n++) {
        y[n] = y[n] * factor;
        sig[n] = sig[n] * factor;
        if (sb2)
        {
            y2[n] = y2[n] * factor;
            sig2[n] = sig2[n] * factor;
        }
    }

    _datafile = "";
    _datafiles = filenames;
    _units = units;
    _skip = skip;
    _multi = true;

    // How many points did we read?
    if (VERBOSE) {
        printf("# Loaded %zu data points from files\n", t.size());
        cout << "#   ";
        for (auto f : filenames) {
            cout << f.c_str();
            (f != filenames.back()) ? cout << " | " : cout << " ";
        }
        cout << endl;
    }

    // Of how many instruments?
    set<int> s(obsi.begin(), obsi.end());
    // set<int>::iterator iter;
    // for(iter=s.begin(); iter!=s.end();++iter) {  cout << (*iter) << endl;}
    if (VERBOSE)
        printf("# RVs come from %zu different instruments.\n", s.size());

    number_instruments = s.size();

    if (VERBOSE && units == "kms")
        cout << "# Multiplied all RVs by 1000; units are now m/s." << endl;

    if (number_instruments > 1) {
        // We need to sort t because it comes from different instruments
        size_t N = t.size();
        vector<double> tt(N), yy(N);
        vector<double> sigsig(N), obsiobsi(N);
        vector<int> order(N);

        // order = argsort(t)
        int x = 0;
        std::iota(order.begin(), order.end(), x++);
        sort(order.begin(), order.end(),
            [&](int i, int j) { return t[i] < t[j]; });

        for (unsigned i = 0; i < N; i++) {
            tt[i] = t[order[i]];
            yy[i] = y[order[i]];
            sigsig[i] = sig[order[i]];
            obsiobsi[i] = obsi[order[i]];
        }

        for (unsigned i = 0; i < N; i++) {
            t[i] = tt[i];
            y[i] = yy[i];
            sig[i] = sigsig[i];
            obsi[i] = obsiobsi[i];
        }
    }

    // epoch for the mean anomaly, by default the time of the first observation
    M0_epoch = t[0];
}


double RVData::get_RV_mean() const
{
    double sum = accumulate(begin(y), end(y), 0.0);
    return sum / y.size();
}

double RVData::get_RV_var() const
{
    double sum = accumulate(begin(y), end(y), 0.0);
    double mean = sum / y.size();

    double accum = 0.0;
    for_each(begin(y), end(y),
            [&](const double d) { accum += (d - mean) * (d - mean); });
    return accum / (y.size() - 1);
}

/**
 * @brief Calculate the maximum slope "allowed" by the data
 *
 * This calculates peak-to-peak(RV) / peak-to-peak(time), which is a good upper
 * bound for the linear slope of a given dataset. When there are multiple
 * instruments, the function returns the maximum of this peak-to-peak ratio of
 * all individual instruments.
 */
double RVData::topslope() const
{
    if (_multi) {
        double slope = 0.0;
        for (size_t j = 0; j < number_instruments; j++) {
            vector<double> obsy, obst;
            for (size_t i = 0; i < y.size(); ++i) {
                if (obsi[i] == j + 1) {
                    obsy.push_back(y[i]);
                    obst.push_back(t[i]);
                }
            }
            const auto miny = min_element(obsy.begin(), obsy.end());
            const auto maxy = max_element(obsy.begin(), obsy.end());
            const auto mint = min_element(obst.begin(), obst.end());
            const auto maxt = max_element(obst.begin(), obst.end());
            double this_obs_topslope = (*maxy - *miny) / (*maxt - *mint);
            if (this_obs_topslope > slope) slope = this_obs_topslope;
        }
        return slope;
    }

    else {
        return get_RV_span() / get_timespan();
    }
}

/**
 * @brief Calculate the maximum span (peak to peak) of the radial velocities
 *
 * This is different from get_RV_span only in the case of multiple instruments:
 * it returns the maximum of the spans of each instrument's RVs.
 */
double RVData::get_max_RV_span() const
{
    // for multiple instruments, calculate individual RV spans and return
    // the largest one
    if (_multi) {
        double span = 0.0;
        for (size_t j = 0; j < number_instruments; j++) {
            vector<double> obsy;
            for (size_t i = 0; i < y.size(); ++i) {
                if (obsi[i] == j + 1) {
                    obsy.push_back(y[i]);
                }
            }
            const auto min = min_element(obsy.begin(), obsy.end());
            const auto max = max_element(obsy.begin(), obsy.end());
            double this_obs_span = *max - *min;
            if (this_obs_span > span) span = this_obs_span;
        }
        return span;
    }

    // for one instrument only, this is easy
    else {
        return get_RV_span();
    }
}

double RVData::get_adjusted_RV_var() const
{
    int ni;
    double sum, mean;
    vector<double> rva(t.size());

    for (size_t j = 0; j < number_instruments; j++) {
        ni = 0;
        sum = 0.;
        for (size_t i = 0; i < t.size(); i++)
            if (obsi[i] == j + 1) {
                sum += y[i];
                ni++;
            }
        mean = sum / ni;
        // cout << "sum: " << sum << endl;
        // cout << "mean: " << mean << endl;
        for (size_t i = 0; i < t.size(); i++)
            if (obsi[i] == j + 1) rva[i] = y[i] - mean;
    }

    mean = accumulate(rva.begin(), rva.end(), 0.0) / rva.size();
    double accum = 0.0;
    for_each(rva.begin(), rva.end(),
            [&](const double d) { accum += (d - mean) * (d - mean); });
    return accum / (y.size() - 1);
}

/**
 * @brief Order of magnitude of trend coefficient (of degree) given the data
 *
 * Returns the expected order of magnitude of the trend coefficient of degree
 * `degree` supported by the data. It calculates the order of magnitude of
 *    RVspan / timespan^degree
 */
int RVData::get_trend_magnitude(int degree) const
{
    return (int)round(log10(get_RV_span() / pow(get_timespan(), degree)));
}


ostream& operator<<(ostream& os, const RVData& d)
{
    os << "RV data from file " << d._datafile << " with " << d.N() << " points";
    return os;
}


/*****************************************************************************/

PHOTdata::PHOTdata() {};

    /**
     * @brief Load RV data from a file.
     *
     * Read a tab/space separated file with columns
     * ```
     *   time  vrad  error  quant  error
     *   ...   ...   ...    ...    ...
     * ```
     *
     * @param filename   the name of the file
     * @param units      units of the RVs and errors, either "kms" or "ms"
     * @param skip       number of lines to skip in the beginning of the file (default = 2)
     * @param indicators
     */
    void PHOTdata::load(const string filename, const string units, int skip,
                        const string delimiter)
    {
        if (filename.empty()) {
            std::string msg = "kima: PHOTdata: no filename provided";
            throw std::invalid_argument(msg);
            // exit(1);
        }

        auto data = loadtxt(filename)
                        .skiprows(skip)
                        .delimiter(delimiter)();

        if (data.size() < 3) {
            std::string msg = "kima: PHOTdata: file (" + filename + ") contains less than 3 columns! (is skip correct?)";
            throw std::runtime_error(msg);
        }
        

        _datafile = filename;
        _units = units;
        _skip = skip;

        t = data[0];
        y = data[1];
        sig = data[2];

        double factor = 1.;
        if (units == "kms") factor = 1E3;

        for (size_t n = 0; n < t.size(); n++) {
            y[n] = y[n] * factor;
            sig[n] = sig[n] * factor;
        }

        // epoch for the mean anomaly, by default the time of the first observation
        M0_epoch = t[0];

        // How many points did we read?
        if (VERBOSE)
            printf("# Loaded %zu data points from file %s\n", t.size(),
                filename.c_str());

        // What are the units?
        if (units == "kms" && VERBOSE)
            printf("# Multiplied all RVs by 1000; units are now m/s.\n");
    }


    double PHOTdata::get_flux_mean() const
    {
        double sum = accumulate(begin(y), end(y), 0.0);
        return sum / y.size();
    }

    double PHOTdata::get_flux_var() const
    {
        double sum = accumulate(begin(y), end(y), 0.0);
        double mean = sum / y.size();

        double accum = 0.0;
        for_each(begin(y), end(y),
                [&](const double d) { accum += (d - mean) * (d - mean); });
        return accum / (y.size() - 1);
    }

    /**
     * @brief Calculate the maximum slope "allowed" by the data
     *
     * This calculates peak-to-peak(RV) / peak-to-peak(time), which is a good upper
     * bound for the linear slope of a given dataset. When there are multiple
     * instruments, the function returns the maximum of this peak-to-peak ratio of
     * all individual instruments.
     */
    double PHOTdata::topslope() const
    {
        return get_flux_span() / get_timespan();
    }

    /**
     * @brief Order of magnitude of trend coefficient (of degree) given the data
     *
     * Returns the expected order of magnitude of the trend coefficient of degree
     * `degree` supported by the data. It calculates the order of magnitude of
     *    RVspan / timespan^degree
     */
    int PHOTdata::get_trend_magnitude(int degree) const
    {
        return (int)round(log10(get_flux_span() / pow(get_timespan(), degree)));
    }


    ostream& operator<<(ostream& os, const PHOTdata& d)
    {
        os << "PHOT data from file " << d._datafile << " with " << d.N() << " points";
        return os;
    }

/*****************************************************************************/

GAIAData::GAIAData() {};
    /**
      * @brief Load Gaia epoch astrometry data from a file.
      *
      * Read a tab/space separated file with columns
      * ```
      *   time  position  error  scan-angle  parallax-factor-along-scan
      *   ...   ...   ...    ...    ...
      * ```
      *
      * @param filename   the name of the file
      * @param units      units of the positions and errors, either "mas" or "muas"(?)
      * @param skip       number of lines to skip in the beginning of the file (default = 2)
      */

    void GAIAData::load(const string filename, const string units, int skip, int max_rows,
                        const string delimiter)
    {
        if (filename.empty()) {
            std::string msg = "kima: GAIAData: no filename provided";
            throw std::invalid_argument(msg);
            // exit(1);
        }

        if (filename.size() == 1) {
            std::string msg = "kima: GAIAData: filename with one character is probably an error";
            throw std::runtime_error(msg);
        }

        auto data = loadtxt(filename)
                        .skiprows(skip)
                        .max_rows(max_rows)
                        .delimiter(delimiter)();

        if (data.size() < 5) {
            std::string msg = "kima: GAIAData: file (" + filename + ") contains less than 5 columns! (is skip correct?)";
            throw std::runtime_error(msg);
        }
        

        _datafile = filename;
        _units = units;
        _skip = skip;

        t = data[0];
        w = data[1];
        wsig = data[2];
        psi = data[3];
        pf = data[4];


        // epoch for the mean anomaly, by default the gaia reference time
        M0_epoch = 57388.5;

        // How many points did we read?
        if (VERBOSE)
            printf("# Loaded %zu data points from file %s\n", t.size(),
                filename.c_str());

    }
                    

/*****************************************************************************/

// class RVData_publicist : public RVData
// {
//     public:
//         using RVData::_datafile;
//         using RVData::_datafiles;
//         using RVData::_units;
//         using RVData::_skip;
//         using RVData::_multi;
// };


// the types of objects in the RVData state (for pickling)
using _state_type = std::tuple<std::string, std::vector<std::string>, std::string, int, std::vector<std::string>, bool>;


NB_MODULE(Data, m) {
    // 
    nb::class_<loadtxt>(m, "loadtxt")
        .def(nb::init<std::string>())
        .def("skiprows", &loadtxt::skiprows)
        .def("comments", &loadtxt::comments)
        .def("delimiter", &loadtxt::delimiter)
        .def("usecols", &loadtxt::usecols)
        .def("max_rows", &loadtxt::max_rows)
        .def("__call__", &loadtxt::operator());

    // 
    nb::class_<RVData>(m, "RVData", "Load and store RV data")
        // constructors
        .def(nb::init<const vector<string>&, const string&, int, int, const string&, const vector<string>&>(),
             "filenames"_a, "units"_a="ms", "skip"_a=0, "max_rows"_a=0, "delimiter"_a=" ", "indicators"_a=vector<string>(),
             "Load RV data from a list of files")
        //
        .def(nb::init<const string&, const string&, int, int, const string&, const vector<string>&>(),
             "filename"_a,  "units"_a="ms", "skip"_a=0, "max_rows"_a=0, "delimiter"_a=" ", "indicators"_a=vector<string>(),
             "Load RV data from a file")
        //
        .def(nb::init<const vector<double>, const vector<double>, const vector<double>, const string&,  const string&>(),
             "t"_a, "y"_a, "sig"_a, "units"_a="ms", "instrument"_a="",
             "Load RV data from arrays")
        //
        .def(nb::init<const vector<vector<double>>, const vector<vector<double>>, const vector<vector<double>>, const string&,  const vector<string>&>(),
             "t"_a, "y"_a, "sig"_a, "units"_a="ms", "instruments"_a=vector<string>(),
             "Load RV data from arrays, for multiple instruments")


        // properties
        .def_prop_ro("t", [](RVData &d) { return d.get_t(); }, "The times of observations")
        .def_prop_ro("y", [](RVData &d) { return d.get_y(); }, "The observed radial velocities")
        .def_prop_ro("sig", [](RVData &d) { return d.get_sig(); }, "The observed RV uncertainties")
        .def_prop_ro("obsi", [](RVData &d) { return d.get_obsi(); }, "The instrument identifier")
        .def_prop_ro("N", [](RVData &d) { return d.N(); }, "Total number of observations")
        .def_prop_ro("actind", [](RVData &d) { return d.get_actind(); }, "Activity indicators")
        //
        .def_ro("units", &RVData::_units, "Units of the RVs and uncertainties")
        .def_ro("multi", &RVData::_multi, "Data comes from multiple instruments")
        .def_ro("skip", &RVData::_skip, "Lines skipped when reading data")
        .def_rw("instrument", &RVData::_instrument, "instrument name")
        //
        .def_rw("M0_epoch", &RVData::M0_epoch, "reference epoch for the mean anomaly")

        // to un/pickle RVData

        .def("__getstate__", [](const RVData &d)
            {
                return std::make_tuple(d._datafile, d._datafiles, d._units, d._skip, d._indicator_names, d._multi);
            })
        .def("__setstate__", [](RVData &d, const _state_type &state)
            {
                bool _multi = std::get<5>(state);
                if (_multi) {
                    new (&d) RVData(std::get<1>(state), std::get<2>(state), std::get<3>(state), 0, " ", std::get<4>(state));
                    //              filename,           units,              skip   
                } else {
                    new (&d) RVData(std::get<0>(state), std::get<2>(state), std::get<3>(state), 0, " ", std::get<4>(state));
                    //              filenames,          units,              skip   
                }
            })
        //
        .def("get_timespan", &RVData::get_timespan)
        .def("get_RV_span", &RVData::get_RV_span)
        .def("topslope", &RVData::topslope)
        // ...
        .def("load", &RVData::load, "filename"_a, "units"_a, "skip"_a, "max_rows"_a, "delimiter"_a, "indicators"_a,
            //  nb::raw_doc(
             R"D(
Load RV data from a tab/space separated file with columns
```
time  vrad  error  quant  error
...   ...   ...    ...    ...
```
Args:
    filename (str): the name of the file
    untis (str): units of the RVs and errors, either "kms" or "ms"
    skip (int): number of lines to skip in the beginning of the file (default = 2)
    indicators (list[str]): nodoc
)D");
// )

    // 

    nb::class_<PHOTdata>(m, "PHOTdata", "docs")
        // constructor
        .def(nb::init<const string&, const string&, int, const string&>(),
             "filename"_a, "units"_a="ms", "skip"_a=0, "delimiter"_a=" ",
             "Load photometric data from a file")
        // properties
        .def_prop_ro("t", [](PHOTdata &d) { return d.get_t(); }, "The times of observations")
        .def_prop_ro("y", [](PHOTdata &d) { return d.get_y(); }, "The observed flux")
        .def_prop_ro("sig", [](PHOTdata &d) { return d.get_sig(); }, "The observed flux uncertainties")
        .def_prop_ro("N", [](PHOTdata &d) { return d.N(); }, "Total number of observations");
        
    // 

    nb::class_<GAIAData>(m, "GAIAData", "docs")
        // constructor
        .def(nb::init<const string&, const string&, int, int, const string&>(),
              "filename"_a, "units"_a="ms", "skip"_a=0, "max_rows"_a=0, "delimiter"_a=" ",
              "Load astrometric data from a file")
        // properties
        .def_prop_ro("t", [](GAIAData &d) { return d.get_t(); }, "The times of observations")
        .def_prop_ro("w", [](GAIAData &d) { return d.get_w(); }, "The observed centroid positions")
        .def_prop_ro("wsig", [](GAIAData &d) { return d.get_wsig(); }, "The observed centroid position uncertainties")
        .def_prop_ro("psi", [](GAIAData &d) { return d.get_psi(); }, "The Gaia scan angles")
        .def_prop_ro("pf", [](GAIAData &d) { return d.get_pf(); }, "the parallax factors");
        //
        //.def("load", &GAIAData::load, "filename"_a, "units"_a, "skip"_a, "max_rows"_a, "delimiter"_a)
}
