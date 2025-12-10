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

template <typename T>
T read_value(std::ifstream &file) {
    T value;
    file.read(reinterpret_cast<char *>(&value), sizeof(value));
    value = swap_endian<T>(value);
    return value;
}

template <typename T>
T swap_endian(T u)
{
    static_assert (CHAR_BIT == 8, "CHAR_BIT != 8");

    union 
    {
        T u;
        unsigned char u8[sizeof(T)];
    } source, dest;

    source.u = u;

    for (size_t k = 0; k < sizeof(T); k++)
        dest.u8[k] = source.u8[sizeof(T) - k - 1];

    return dest.u;
}


/// @brief Load data from a single filename
RVData::RVData(const string& filename, const string& units, int skip, int max_rows, bool multi,
               const string& delimiter, const vector<string>& indicators, bool double_lined)
{
    if (double_lined)
        sb2=true;
    if (multi)
        load_multi(filename, units, skip, max_rows, delimiter, indicators);
    else
        load(filename, units, skip, max_rows, delimiter, indicators);
    
    normalize_actind();
}

/// @brief Load data from a list of filenames
RVData::RVData(const vector<string>& filenames, const string& units, int skip, int max_rows, 
               const string& delimiter, const vector<string>& indicators, bool double_lined)
{
    if (double_lined)
        sb2=true;
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

    // epoch for the mean anomaly, by default the mid time
    M0_epoch = get_t_middle();
    // epoch for the trend, by default the mid time
    trend_epoch = get_t_middle();

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

    if (_t.size() != instruments.size()) 
    {
        string msg = "RVData: data and instruments must have the same size size(t) != size(instruments)";
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
            obsi.push_back(static_cast<int>(i) + 1);
    }
    actind.clear();

    _datafile = "";
    _datafiles = {};
    _units = units;
    _skip = 0;
    _multi = true;
    _indicator_names = {};
    number_indicators = 0;
    number_instruments = static_cast<int>(_t.size());
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

    // epoch for the mean anomaly, by default the mid time
    M0_epoch = get_t_middle();

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
        vector<double> sigsig(N);
        vector<int> order(N), obsiobsi(N);

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
                    .max_errors(0)
                    .skiprows(skip)
                    .max_rows(max_rows)
                    .delimiters(delimiter)();

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
    _instrument = "RVs";//Just call the instrument RVs by default

    t = data[0];
    y = data[1];
    sig = data[2];
    obsi = vector<int>(t.size(), 1);
    
    if (sb2)
    {
        if (data.size() < 5) {
            std::string msg = "kima: RVData: sb2 is true but file (" + filename + ") contains less than 5 columns!";
            throw std::runtime_error(msg);
        }
        y2 = data[3];
        sig2 = data[4];
    }

    // check for indicator correlations and store stuff
    int nempty = static_cast<int>( count(indicators.begin(), indicators.end(), "") );
    number_indicators = static_cast<int>(indicators.size()) - nempty;
    indicator_correlations = number_indicators > 0;

    if (data.size() < 3 + number_indicators + nempty) {
        std::string msg = "kima: RVData: file (" + filename + ") contains too few columns!";
        throw std::runtime_error(msg);
    }

    _indicator_names = indicators;
    _indicator_names.erase(std::remove(_indicator_names.begin(), _indicator_names.end(), ""), _indicator_names.end());

    // empty and resize the indicator vectors
    actind.clear();
    actind.resize(number_indicators);
    normalized_actind.clear();
    normalized_actind.resize(number_indicators);
    for (int n = 0; n < number_indicators; n++) {
        actind[n].clear();
        normalized_actind[n].clear();
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
                actind[j] = data[3 + i];
                normalized_actind[j] = data[3 + i];
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

    // epoch for the mean anomaly, by default the mid time
    M0_epoch = get_t_middle();
    // epoch for the trend, by default the mid time
    trend_epoch = get_t_middle();

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
 * The `obs` column should be the last one and an integer identifying the instrument.
 *
 * @param filename   the name of the file
 * @param units      units of the RVs and errors, either "kms" or "ms"
 * @param skip       number of lines to skip in the beginning of the file (default = 2)
 */
void RVData::load_multi(const string filename, const string units, int skip, int max_rows,
                        const string delimiter, const vector<string> &indicators)
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
                    .max_errors(0)
                    .skiprows(skip)
                    .max_rows(max_rows)
                    .delimiters(delimiter)();

    if (data.size() < 4) {
        std::string msg = "kima: RVData: file (" + filename + ") contains less than 4 columns! (multi=true)";
        throw std::runtime_error(msg);
    }


    auto Ncol = data.size();
    auto N = data[0].size();

    _datafile = filename;
    _datafiles = {};
    _units = units;
    _skip = skip;
    _multi = true;
    _instrument = "";
    _instruments = {};

    t = data[0];
    y = data[1];
    sig = data[2];
    
    if (sb2)
    {
        y2 = data[3];
        sig2 = data[4];
    }

    // check for indicator correlations and store stuff
    int nempty = (int) count(indicators.begin(), indicators.end(), "");
    number_indicators = (int)(indicators.size()) - nempty;
    indicator_correlations = number_indicators > 0;

    if (data.size() < 3 + number_indicators + nempty) {
        std::string msg = "kima: RVData: file (" + filename + ") contains too few columns!";
        throw std::runtime_error(msg);
    }

    _indicator_names = indicators;
    _indicator_names.erase(std::remove(_indicator_names.begin(), _indicator_names.end(), ""), _indicator_names.end());

    // empty and resize the indicator vectors
    actind.clear();
    actind.resize(number_indicators);
    normalized_actind.clear();
    normalized_actind.resize(number_indicators);
    for (int n = 0; n < number_indicators; n++) {
        actind[n].clear();
        normalized_actind[n].clear();
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
                actind[j] = data[3 + i];
                normalized_actind[j] = data[3 + i];
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

    // the last column of the file identifies the instrument; it can have "0"s
    // this is to make sure the obsi vector always starts at 1, to avoid
    // segmentation faults later
    vector<int> inst_id;
    inst_id.push_back(static_cast<int>(data[Ncol - 1][0]));

    for (size_t n = 1; n < N; n++) {
        if (data[Ncol - 1][n] != inst_id.back()) {
            inst_id.push_back(static_cast<int>(data[Ncol - 1][n]));
        }
    }
    int id_offset = *min_element(inst_id.begin(), inst_id.end());

    obsi.clear();
    for (size_t n = 0; n < N; n++) {
        obsi.push_back(static_cast<int>(data[Ncol - 1][n]) - id_offset + 1);
    }

    // How many points did we read?
    if (VERBOSE)
        printf("# Loaded %zu data points from file %s\n", t.size(), filename.c_str());

    // Of how many instruments?
    set<int> s(inst_id.begin(), inst_id.end());

    for (auto& inst : s) {
        _instruments.push_back(std::to_string(inst));
    }

    number_instruments = (int) s.size();
    if (VERBOSE)
        printf("# RVs come from %zu different instruments.\n", s.size());

    if (units == "kms" && VERBOSE)
        cout << "# Multiplied all RVs by 1000; units are now m/s." << endl;

    // epoch for the mean anomaly, by default the mid time
    M0_epoch = get_t_middle();
    // epoch for the trend, by default the mid time
    trend_epoch = get_t_middle();
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
    int nempty = (int) count(indicators.begin(), indicators.end(), "");
    number_indicators = (int)(indicators.size()) - nempty;
    indicator_correlations = number_indicators > 0;
    _indicator_names = indicators;
    _indicator_names.erase(std::remove(_indicator_names.begin(), _indicator_names.end(), ""), _indicator_names.end());

    // empty and resize the indicator vectors
    actind.clear();
    actind.resize(number_indicators);
    normalized_actind.clear();
    normalized_actind.resize(number_indicators);
    for (int n = 0; n < number_indicators; n++) {
        actind[n].clear();
        normalized_actind[n].clear();
    }

    int filecount = 1;
    for (auto& filename : filenames) {
        auto data = loadtxt(filename)
                        .max_errors(0)
                        .skiprows(skip)
                        .max_rows(max_rows)
                        .delimiters(delimiter)();

        if (data.size() < 3) {
            std::string msg = "kima: RVData: file (" + filename + ") contains less than 3 columns! (is skip correct?)";
            throw std::runtime_error(msg);
        }

        if (data.size() < 3 + number_indicators + nempty) {
            std::string msg = "kima: RVData: file (" + filename + ") contains too few columns!";
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
                    actind[j].insert(actind[j].end(), data[3 + i].begin(), data[3 + i].end());
                    normalized_actind[j].insert(normalized_actind[j].end(), data[3 + i].begin(), data[3 + i].end());
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

    number_instruments = (int) s.size();

    if (VERBOSE && units == "kms")
        cout << "# Multiplied all RVs by 1000; units are now m/s." << endl;

    if (number_instruments > 1) {
        // We need to sort t because it comes from different instruments
        size_t N = t.size();
        vector<double> tt(N), yy(N), sigsig(N);
        vector<vector<double>> aiai(number_indicators, vector<double>(N));
        vector<vector<double>> nainai(number_indicators, vector<double>(N));
        vector<int> obsiobsi(N);
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
            for (size_t j = 0; j < number_indicators; j++)
            {
                aiai[j][i] = actind[j][order[i]];
                nainai[j][i] = normalized_actind[j][order[i]];
            }
        }

        for (size_t i = 0; i < N; i++) {
            t[i] = tt[i];
            y[i] = yy[i];
            sig[i] = sigsig[i];
            obsi[i] = obsiobsi[i];
            for (size_t j = 0; j < number_indicators; j++)
            {
                actind[j][i] = aiai[j][i];
                normalized_actind[j][i] = nainai[j][i];
            }
        }
    }

    // epoch for the mean anomaly, by default the mid time
    M0_epoch = get_t_middle();
    // epoch for the trend, by default the mid time
    trend_epoch = get_t_middle();

    // normalize the activity indicators
    normalize_actind();
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
    for_each(begin(y), end(y), [&](const double d) { accum += (d - mean) * (d - mean); });
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
int RVData::get_trend_magnitude(int degree, int i) const
{
    if (i == -1)
        return (int)round(log10(get_RV_span() / pow(get_timespan(), degree)));
    else
    {
        return (int)round(log10(get_actind_span( static_cast<size_t>(i) ) / pow(get_timespan(), degree)));
    }
}

double RVData::get_actind_mean(size_t i) const
{
    auto ind = actind[i];

    size_t n = std::count_if(begin(ind), end(ind), [](double d) { return !std::isnan(d); });

    double sum = std::accumulate(begin(ind), end(ind), 0.0,
                                 [](double acc, double other) { return std::isnan(other) ? acc : acc + other; });
    return sum / n;
}

double RVData::get_actind_var(size_t i) const
{
    double sum = accumulate(begin(actind[i]), end(actind[i]), 0.0);
    double mean = sum / actind[i].size();
    double accum = 0.0;
    for_each(begin(actind[i]), end(actind[i]), [&](const double d) { accum += (d - mean) * (d - mean); });
    return accum / (actind[i].size() - 1);
}

// normalize activity indicators from 0 to 1
// by subtracting the minimum value and dividing by the range
void RVData::normalize_actind()
{
    if (actind.size() == 0) return;

    for (size_t i=0; i<actind.size(); i++) {
        double min = *min_element(actind[i].begin(), actind[i].end());
        double max = *max_element(actind[i].begin(), actind[i].end());
        for (size_t j = 0; j < actind[i].size(); j++)
        {
            normalized_actind[i][j] = (actind[i][j] - min) / (max - min);
        }
    }
}

ostream& operator<<(ostream& os, const RVData& d)
{
    os << "RV data from file " << d._datafile << " with " << d.N() << " points";
    return os;
}


/*****************************************************************************/

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
void PHOTdata::load(const string filename, int skip, const string delimiter)
{
    if (filename.empty()) {
        std::string msg = "kima: PHOTdata: no filename provided";
        throw std::invalid_argument(msg);
        // exit(1);
    }

    if (filename.size() == 1) {
        std::string msg = "kima: RVData: filename with one character is probably an error";
        throw std::runtime_error(msg);
    }

    auto data = loadtxt(filename)
                    .max_errors(0)
                    .skiprows(skip)
                    .delimiters(delimiter)();

    if (data.size() < 3) {
        std::string msg = "kima: PHOTdata: file (" + filename + ") contains less than 3 columns! (is skip correct?)";
        throw std::runtime_error(msg);
    }

    _datafile = filename;
    _skip = skip;

    t = data[0];
    y = data[1];
    sig = data[2];

    // epoch for the mean anomaly, by default the mid time
    M0_epoch = get_t_middle();

    // How many points did we read?
    if (VERBOSE)
        printf("# Loaded %zu data points from file %s\n", t.size(),
            filename.c_str());
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

GAIAdata::GAIAdata() {};
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

    void GAIAdata::load(const string filename, const string units, int skip, int max_rows,
                        const string delimiter)
    {
        if (filename.empty()) {
            std::string msg = "kima: GAIAdata: no filename provided";
            throw std::invalid_argument(msg);
            // exit(1);
        }

        if (filename.size() == 1) {
            std::string msg = "kima: GAIAdata: filename with one character is probably an error";
            throw std::runtime_error(msg);
        }

        auto data = loadtxt(filename)
                        .max_errors(0)
                        .skiprows(skip)
                        .max_rows(max_rows)
                        .delimiters(delimiter)();

        if (data.size() < 5) {
            std::string msg = "kima: GAIAdata: file (" + filename + ") contains less than 5 columns! (is skip correct?)";
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
        M0_epoch = 57936.875;

        // How many points did we read?
        if (VERBOSE)
            printf("# Loaded %zu data points from file %s\n", t.size(),
                filename.c_str());

    }
                    

/*****************************************************************************/

HGPMdata::HGPMdata() {};

    void HGPMdata::load(uint64_t _gaia_id)
    {
        auto data = get_data(_gaia_id);

        if (!data.found) {
            std::string msg = "kima: HGPMdata: no data for gaia_id " + std::to_string(_gaia_id);
            throw std::runtime_error(msg);
        }

        gaia_id = _gaia_id;

        // // auto catalog_file = "C://Users/joaof/Work/HGCA_catalog.dat";
        // // auto gaia_ids = loadtxt<unsigned long long>(catalog_file).delimiters("|").usecols({2})();
        // // auto catalog = loadtxt(catalog_file).delimiters("|")();
        
        // size_t nrows = catalog.size();
        // std::cout << "read " << nrows << " rows from HGPM catalog" << std::endl;

        // auto index = find(gaia_ids.begin(), gaia_ids.end(), gaia_id);
        // size_t found = index - gaia_ids.begin();
        // std::cout << "found gaia_id in row " << found << std::endl;

        // store parallax
        parallax_gaia = data.parallax_gaia;
        parallax_gaia_error = data.parallax_gaia_error;
        // parallax_gaia = catalog[found][5];
        // parallax_gaia_error = catalog[found][6];


        // Convert measurement epochs to MJD, assuming dates are Julian years
        double J2000_mjd = 51544.5; // year J2000 in MJD
        double epoch_ra_hip_mjd = (data.epoch_ra_hip - 2000)*365.25 + J2000_mjd;
        double epoch_dec_hip_mjd = (data.epoch_dec_hip - 2000)*365.25 + J2000_mjd;
        double epoch_ra_gaia_mjd = (data.epoch_ra_gaia - 2000)*365.25 + J2000_mjd;
        double epoch_dec_gaia_mjd = (data.epoch_dec_gaia - 2000)*365.25 + J2000_mjd;
        
        // Rough timespans for Hipparcos and Gaia
        double dt_gaia = 1038; // EDR3: days between 2017-05-28 and 2014-07-25
        double dt_hip = 4 * 365.25; // 4 years

        // Hipparcos
        epoch_ra_hip = epoch_ra_hip_mjd + 0.0;
        epoch_dec_hip = epoch_dec_hip_mjd + 0.0;
        // Gaia
        epoch_ra_gaia = epoch_ra_gaia_mjd + 0.0;
        epoch_dec_gaia = epoch_dec_gaia_mjd + 0.0;

        // proper motions
        pm_ra_hip = data.pmra_hip;
        pm_dec_hip = data.pmdec_hip;
        pm_ra_hg = data.pmra_hg;
        pm_dec_hg = data.pmdec_hg;
        pm_ra_gaia = data.pmra_gaia;
        pm_dec_gaia = data.pmdec_gaia;

        // Hipparcos epoch
        rho_hip = data.pmra_pmdec_hip; // * data.pmra_hip_error * data.pmdec_hip_error;
        sig_hip_ra = data.pmra_hip_error;
        sig_hip_dec = data.pmdec_hip_error;
        // Hipparcos - GAIA epoch
        rho_hg = data.pmra_pmdec_hg; // * data.pmra_hg_error * data.pmdec_hg_error;
        sig_hg_ra = data.pmra_hg_error;
        sig_hg_dec = data.pmdec_hg_error;
        // GAIA epoch
        rho_gaia = data.pmra_pmdec_gaia; // * data.pmra_gaia_error * data.pmdec_gaia_error;
        sig_gaia_ra = data.pmra_gaia_error;
        sig_gaia_dec = data.pmdec_gaia_error;

        // chi square
        chisq = data.chisq;
    };

    vector<double> HGPMdata::get_epochs(size_t n_average) const
    {
        if (n_average == 1) {
            return {epoch_ra_hip, epoch_dec_hip, epoch_ra_gaia, epoch_dec_gaia};
        }
        else {
            throw std::runtime_error("n_average must be 1");
        }
    }

    hgca_data HGPMdata::get_data(uint64_t target_id)
    {
        const size_t BLOCK_SIZE = 2880;
        const size_t ROW_SIZE = 172;
        const size_t NROWS = 115346;
        auto data = hgca_data();
        data.found = false;

        // this is a bit of a hack...
        // fs::temp_directory_path provides a platform independent path to a
        // temporary directory, which is guaranteed to exist. This path is
        // stored in the `temp_path` property of HGPMdata. On the Python side,
        // we use pooch to download (only once) the HGCA fits file and place it
        // in this temporary directory, so that it exists and can be opened
        // below. We go through all this trouble just to not add the fits file
        // to the kima package/wheels/repository/etc.
        // I am almost sure this code will come back to haunt me one day...

        fs::path HGCA_file = fs::temp_directory_path() / "HGCA_vEDR3.fits";

        std::ifstream file(HGCA_file.string(), std::ios::binary);
        if (file) {
            // skip primary HDU header and first extension HDU header
            file.seekg(4 * BLOCK_SIZE);

            for (size_t i = 0; i < NROWS; i++)
            {
                uint32_t id = read_value<uint32_t>(file);
                uint64_t gaia_id = read_value<uint64_t>(file);
                // std::cout << id << " ";
                // std::cout << gaia_id << std::endl;
        
                if (gaia_id == target_id) {
                    data.found = true;
                    data.row = i;
                    data.gaia_id = gaia_id;
                    data.gaia_ra = read_value<double>(file);
                    data.gaia_dec = read_value<double>(file);
                    data.radial_velocity = read_value<float>(file);
                    data.radial_velocity_error = read_value<float>(file);
        
                    // ignore radial_velocity_source
                    file.seekg(8, std::ios::cur);
        
                    data.parallax_gaia = read_value<float>(file);
                    data.parallax_gaia_error = read_value<float>(file);
                    data.pmra_gaia = read_value<float>(file);
                    data.pmdec_gaia = read_value<float>(file);
                    data.pmra_gaia_error = read_value<float>(file);
                    data.pmdec_gaia_error = read_value<float>(file);
                    data.pmra_pmdec_gaia = read_value<float>(file);
                    data.pmra_hg = read_value<float>(file);
                    data.pmdec_hg = read_value<float>(file);
                    data.pmra_hg_error = read_value<float>(file);
                    data.pmdec_hg_error = read_value<float>(file);
                    data.pmra_pmdec_hg = read_value<float>(file);
                    data.pmra_hip = read_value<float>(file);
                    data.pmdec_hip = read_value<float>(file);
                    data.pmra_hip_error = read_value<float>(file);
                    data.pmdec_hip_error = read_value<float>(file);
                    data.pmra_pmdec_hip = read_value<float>(file);
                    data.epoch_ra_gaia = read_value<double>(file);
                    data.epoch_dec_gaia = read_value<double>(file);
                    data.epoch_ra_hip = read_value<double>(file);
                    data.epoch_dec_hip = read_value<double>(file);
                    data.crosscal_pmra_hip = read_value<float>(file);
                    data.crosscal_pmdec_hip = read_value<float>(file);
                    data.crosscal_pmra_hg = read_value<float>(file);
                    data.crosscal_pmdec_hg = read_value<float>(file);
                    data.nonlinear_dpmra = read_value<float>(file);
                    data.nonlinear_dpmdec = read_value<float>(file);
                    data.chisq = read_value<float>(file);
                    break;
                }
                file.seekg(ROW_SIZE - sizeof(id) - sizeof(gaia_id), std::ios::cur);
            }
            file.close();
        }
        else {
            std::cout << "Unable to open file (" << HGCA_file << ")" << std::endl;
        }
        return data;
    }



/*****************************************************************************/

ETVData::ETVData() {};
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

    void ETVData::load(const string filename, const string units, int skip, int max_rows,
                        const string delimiter)
    {
        if (filename.empty()) {
            std::string msg = "kima: ETVData: no filename provided";
            throw std::invalid_argument(msg);
            // exit(1);
        }

        if (filename.size() == 1) {
            std::string msg = "kima: ETVData: filename with one character is probably an error";
            throw std::runtime_error(msg);
        }

        auto data = loadtxt(filename)
                        .max_errors(0)
                        .skiprows(skip)
                        .max_rows(max_rows)
                        .delimiters(delimiter)();

        if (data.size() < 3) {
            std::string msg = "kima: ETVData: file (" + filename + ") contains less than 3 columns! (is skip correct?)";
            throw std::runtime_error(msg);
        }
        

        _datafile = filename;
        _units = units;
        _skip = skip;

        epochs = data[0];
        et = data[1];
        etsig = data[2];

        // epoch for the mean anomaly, by default the first epoch
        M0_epoch = et[0];

        // How many points did we read?
        if (VERBOSE)
            printf("# Loaded %zu data points from file %s\n", epochs.size(),
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
    nb::class_<loadtxt<double>>(m, "loadtxt")
        .def(nb::init<std::string>())
        .def("skiprows", &loadtxt<double>::skiprows)
        .def("comments", &loadtxt<double>::comments)
        .def("delimiters", &loadtxt<double>::delimiters)
        .def("usecols", &loadtxt<double>::usecols)
        .def("max_rows", &loadtxt<double>::max_rows)
        .def("max_errors", &loadtxt<double>::max_errors)
        .def("__call__", &loadtxt<double>::operator());

    // 
    nb::class_<RVData>(m, "RVData", "Load and store RV data")
        // constructors
        .def(nb::init<const vector<string>&, const string&, int, int, const string&, const vector<string>&, bool>(),
             "filenames"_a, "units"_a="ms", "skip"_a=0, "max_rows"_a=0, "delimiter"_a=" \t,", "indicators"_a=vector<string>(), "double_lined"_a=false,
             "Load RV data from a list of files")
        //
        .def(nb::init<const string&, const string&, int, int, bool, const string&, const vector<string>&, bool>(),
             "filename"_a,  "units"_a="ms", "skip"_a=0, "max_rows"_a=0, "multi"_a=false, "delimiter"_a=" \t,", "indicators"_a=vector<string>(), "double_lined"_a=false,
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
        .def_ro("datafile", &RVData::_datafile, "The file name")
        .def_ro("datafiles", &RVData::_datafiles, "The list of file names")
        // 
        .def_prop_ro("t", [](RVData &d) { return d.get_t(); }, "The times of observations")
        .def_prop_ro("y", [](RVData &d) { return d.get_y(); }, "The observed radial velocities")
        .def_prop_ro("sig", [](RVData &d) { return d.get_sig(); }, "The observed RV uncertainties")
        .def_prop_ro("y2", [](RVData &d) { return d.get_y2(); }, "The observed secondary radial velocities (double-lined binary)")
        .def_prop_ro("sig2", [](RVData &d) { return d.get_sig2(); }, "The observed secondary RV uncertainties (double-lined binary)")
        .def_prop_ro("obsi", [](RVData &d) { return d.get_obsi(); }, "The instrument identifier")
        .def_prop_ro("N", [](RVData &d) { return d.N(); }, "Total number of observations")
        .def_prop_ro("actind", [](RVData &d) { return d.get_actind(); }, "Activity indicators")
        .def_prop_ro("normalized_actind", [](RVData &d) { return d.get_normalized_actind(); }, 
                     "Activity indicators normalized to [0,1]")
        //
        .def_ro("units", &RVData::_units, "Units of the RVs and uncertainties")
        .def_ro("multi", &RVData::_multi, "Data comes from multiple instruments")
        .def_ro("skip", &RVData::_skip, "Lines skipped when reading data")
        .def_rw("instrument", &RVData::_instrument, "instrument name")
        .def_rw("instruments", &RVData::_instruments, "instrument names")
        .def_rw("indicator_names", &RVData::_indicator_names, "names of activity indicators")
        //
        .def_rw("M0_epoch", &RVData::M0_epoch, "reference epoch for the mean anomaly")
        .def_rw("trend_epoch", &RVData::trend_epoch, "reference epoch for the trend")
        .def_rw("double_lined", &RVData::sb2, "if the data is for a double-lined binary")

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
                    //              filenames,          units,              skip   
                } else {
                    new (&d) RVData(std::get<0>(state), std::get<2>(state), std::get<3>(state), 0, _multi, " ", std::get<4>(state));
                    //              filename,           units,              skip   
                }
            })
        //

        .def("get_t_min", &RVData::get_t_min, "Get the minimum time")
        .def("get_t_max", &RVData::get_t_max, "Get the maximum time")
        .def("get_timespan", &RVData::get_timespan, "Get the timespan")
        .def("get_t_middle", &RVData::get_t_middle, "Get the middle time")
        // 
        .def("get_RV_min", &RVData::get_RV_min, "Get the minimum RV")
        .def("get_RV_max", &RVData::get_RV_max, "Get the maximum RV")
        .def("get_RV_span", &RVData::get_RV_span, "Get the RV span")
        .def("get_max_RV_span", &RVData::get_max_RV_span, "Get the maximum RV span of individual instruments")
        .def("get_RV_mean", &RVData::get_RV_mean, "Get the mean RV")
        .def("get_RV_var", &RVData::get_RV_var, "Get the variance of RVs")
        .def("get_RV_std", &RVData::get_RV_std, "Get the standard deviation of RVs")
        .def("topslope", &RVData::topslope, "Get the maximum slope allowed by the data")
        .def("get_trend_magnitude", &RVData::get_trend_magnitude, "degree"_a, "i"_a = -1, "Order of magnitude of trend coefficient (of degree) given the data")
        .def("get_unique_t", &RVData::get_unique_t, "Get the unique times")
        .def("_inverse_time_indices", &RVData::_inverse_time_indices, "")
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
        .def(nb::init<const string&, int, const string&>(),
             "filename"_a, "skip"_a=0, "delimiter"_a=" \t,",
             "Load photometric data from a file")
        // properties
        .def_prop_ro("t", [](PHOTdata &d) { return d.get_t(); }, "The times of observations")
        .def_prop_ro("y", [](PHOTdata &d) { return d.get_y(); }, "The observed flux")
        .def_prop_ro("sig", [](PHOTdata &d) { return d.get_sig(); }, "The observed flux uncertainties")
        .def_prop_ro("N", [](PHOTdata &d) { return d.N(); }, "Total number of observations");
        
    // 

    nb::class_<GAIAdata>(m, "GAIAdata", "Load and store Gaia astrometric data")
        // constructor
        .def(nb::init<const string&, const string&, int, int, const string&>(),
              "filename"_a, "units"_a="ms", "skip"_a=0, "max_rows"_a=0, "delimiter"_a=" \t,",
              "Load astrometric data from a file")
        // properties
        .def_ro("datafile", &GAIAdata::_datafile, "The file name")
        //
        .def_prop_ro("t", [](GAIAdata &d) { return d.get_t(); }, "The times of observations")
        .def_prop_ro("w", [](GAIAdata &d) { return d.get_w(); }, "The observed centroid positions")
        .def_prop_ro("wsig", [](GAIAdata &d) { return d.get_wsig(); }, "The observed centroid position uncertainties")
        .def_prop_ro("psi", [](GAIAdata &d) { return d.get_psi(); }, "The Gaia scan angles")
        .def_prop_ro("pf", [](GAIAdata &d) { return d.get_pf(); }, "The parallax factors")
        .def_prop_ro("N", [](GAIAdata &d) { return d.N(); }, "Total number of observations")
        //
        .def_rw("M0_epoch", &GAIAdata::M0_epoch, "reference epoch for the mean anomaly")
        //.def("load", &GAIAdata::load, "filename"_a, "units"_a, "skip"_a, "max_rows"_a, "delimiter"_a)
        .def_ro("units", &GAIAdata::_units, "Units of the Gaia data and uncertainties");
    
    // 

    nb::class_<HGPMdata>(m, "HGPMdata", "Load and store Hipparcos-Gaia proper motion data")
        // constructor
        .def(nb::init<unsigned long long>(), "gaia_id"_a, "Load the Hipparcos-Gaia Catalog of Accelerations")
        // 
        .def_ro_static("_temp_path", &HGPMdata::temp_path)
        // 
        .def_ro("gaia_id", &HGPMdata::gaia_id, "Gaia DR3 ID")
        // 
        .def_ro("parallax_gaia", &HGPMdata::parallax_gaia, "Gaia DR3 parallax")
        .def_ro("parallax_gaia_error", &HGPMdata::parallax_gaia_error, "Gaia DR3 parallax error")
        // 
        .def_ro("epoch_ra_hip", &HGPMdata::epoch_ra_hip, "Central epoch of Hipparcos RA measurement")
        .def_ro("epoch_dec_hip", &HGPMdata::epoch_dec_hip, "Central epoch of Hipparcos DEC measurement")
        .def_ro("epoch_ra_gaia", &HGPMdata::epoch_ra_gaia, "Central epoch of Gaia RA measurement")
        .def_ro("epoch_dec_gaia", &HGPMdata::epoch_dec_gaia, "Central epoch of Gaia DEC measurement")
        // proper motion measurements
        .def_ro("pm_ra_hip", &HGPMdata::pm_ra_hip, "Calibrated proper motion in RA from the composite Hipparcos catalog")
        .def_ro("pm_dec_hip", &HGPMdata::pm_dec_hip, "Calibrated proper motion in DEC from the composite Hipparcos catalog")
        .def_ro("pm_ra_gaia", &HGPMdata::pm_ra_gaia, "Gaia EDR3 proper motion in RA")
        .def_ro("pm_dec_gaia", &HGPMdata::pm_dec_gaia, "Gaia EDR3 proper motion in DEC")
        .def_ro("pm_ra_hg", &HGPMdata::pm_ra_hg, "Calibrated proper motion in RA from the Hipparcos-Gaia positional difference")
        .def_ro("pm_dec_hg", &HGPMdata::pm_dec_hg, "Calibrated proper motion in DEC from the Hipparcos-Gaia positional difference")
        // uncertainties and correlations
        .def_ro("sig_hip_ra", &HGPMdata::sig_hip_ra, "").def_ro("sig_hip_dec", &HGPMdata::sig_hip_dec, "").def_ro("rho_hip", &HGPMdata::rho_hip, "")
        .def_ro("sig_gaia_ra", &HGPMdata::sig_gaia_ra, "").def_ro("sig_gaia_dec", &HGPMdata::sig_gaia_dec, "").def_ro("rho_gaia", &HGPMdata::rho_gaia, "")
        .def_ro("sig_hg_ra", &HGPMdata::sig_hg_ra, "").def_ro("sig_hg_dec", &HGPMdata::sig_hg_dec, "").def_ro("rho_hg", &HGPMdata::rho_hg, "")
        // chi square value
        .def_ro("chisq", &HGPMdata::chisq, "")
        // for pickling
        .def("__getstate__",
             [](const HGPMdata &d) { 
                return std::make_tuple(d.gaia_id); 
        })
        .def("__setstate__",
             [](HGPMdata &d, const std::tuple<uint64_t> &t) { 
                d.load(std::get<0>(t));
        });



    nb::class_<ETVData>(m, "ETVData", "docs")
        // constructor
        .def(nb::init<const string&, const string& , int, int, const string&>(),
              "filename"_a, "units"_a="days", "skip"_a=0, "max_rows"_a=0, "delimiter"_a=" ",
              "Load Eclipse timing data from a file")
        // properties
        .def_ro("datafile", &ETVData::_datafile, "The file name")
        //
        .def_prop_ro("epochs", [](ETVData &d) { return d.get_epochs(); }, "The epoch (Nth eclipse since number 0)")
        .def_prop_ro("et", [](ETVData &d) { return d.get_et(); }, "The observed mid-eclipse times")
        .def_prop_ro("etsig", [](ETVData &d) { return d.get_etsig(); }, "The uncertainties in the eclipse times")
        .def_prop_ro("N", [](ETVData &d) { return d.N(); }, "Total number of observations")

        .def_rw("M0_epoch", &ETVData::M0_epoch, "reference epoch for the mean anomaly");

        //
        //.def("load", &GAIAData::load, "filename"_a, "units"_a, "skip"_a, "max_rows"_a, "delimiter"_a)
}
