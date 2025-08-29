#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <numeric>
#include <optional>
#include <vector>
#include <map>
#include <exception>
#include <stdexcept>

using namespace std;

template <typename T=double>
struct loadtxt {
    // Mandatory arguments
    loadtxt(string fname) : _fname(fname) {}

    // Optional arguments
    // ('this' is returned for chaining)
    loadtxt& comments(string comments) { _comments = comments; return *this; }
    loadtxt& delimiters(string delimiters) { _delimiters = delimiters; return *this; }
    loadtxt& skiprows(size_t skiprows) { _skiprows = skiprows; return *this; }
    loadtxt& usecols(vector<int> usecols) { _usecols = usecols; return *this; }
    loadtxt& max_rows(int max_rows) { _max_rows = max_rows; return *this; }
    loadtxt& max_errors(int max_errors) { _max_errors = max_errors; return *this; }

    T convert(const string& s) {
        if constexpr (std::is_same_v<T, int>)
            return stoi(s);
        else if constexpr (std::is_same_v<T, long>)
            return stol(s);
        else if constexpr (std::is_same_v<T, long long>)
            return stoll(s);
        else if constexpr (std::is_same_v<T, unsigned long long>)
            return stoull(s);
        
        return stod(s);
    }

    vector<vector<T>> operator()()
    {
        if (_fname.empty()) {
            string msg = "loadtxt: filename is empty";
            throw invalid_argument(msg);
        }

        ifstream infile( _fname );

        if (infile.fail()) {
            string msg = "loadtxt: could not read file (" + _fname + ")!";
            throw runtime_error(msg);
        }
        
        // ignore first `skiprows` lines
        static const int max_line = 65536;
        for (size_t i = 0; i < _skiprows; i++)
            infile.ignore(max_line, '\n');

        vector<T> record;
        size_t rows_read = 0;
        size_t min_cols = numeric_limits<size_t>::max();
        size_t _errors = 0;

        // read each line
        while (rows_read < _max_rows || _max_rows == 0)
        {
            // clear the record before reading
            record.clear();

            // read the entire line into a string (a record is terminated by a newline)
            string line;
            getline(infile, line);

            if (infile.eof() && line.empty())
                break;
            
            if (line.find(_comments, 0) == 0)
                continue;

            // // use a stringstream to separate the fields out of the line
            // stringstream ss(line);

            size_t found = line.find_first_of(_delimiters);

            while (found != string::npos) {
                string cell = line.substr(0, found);

                if (!cell.empty()) {
                    // convert the field to a double
                    try
                    {
                        T val = convert(cell);
                        record.push_back(val);
                    }
                    catch(const std::invalid_argument& e)
                    {
                        if (_errors < _max_errors) {
                            std::cerr << "loadtxt: while reading file \"" << _fname << "\", ";
                            std::cerr << "could not convert field '" << cell << "' to double ";
                            std::cerr << "at row " << rows_read + 1 << " column " << record.size() + 1 << "\n";
                        }
                        else if (_errors == _max_errors) {
                            std::cerr << "loadtxt: while reading file \"" << _fname << "\", ";
                            if (_errors == 0) 
                                std::cerr << "errors omitted\n";
                            else
                                std::cerr << "other errors omitted\n";
                        }
                        _errors++;
                        record.push_back(std::nan(""));
                        // throw;
                    }
                    
                }
                line.erase(0, found + 1);
                found = line.find_first_of(_delimiters);

                if (found == string::npos) {
                    if (line.empty())
                        continue;

                    try
                    {
                        T val = convert(line);
                        record.push_back(val);
                    }
                    catch(const std::invalid_argument& e)
                    {
                        if (_errors < _max_errors) {
                            std::cerr << "loadtxt: while reading file \"" << _fname << "\", ";
                            std::cerr << "could not convert field '" << line << "' to double ";
                            std::cerr << "at row " << rows_read + 1 << " column " << record.size() + 1 << "\n";
                        }
                        else if (_errors == _max_errors) {
                            std::cerr << "loadtxt: while reading file \"" << _fname << "\", ";
                            if (_errors == 0) 
                                std::cerr << "errors omitted\n";
                            else
                                std::cerr << "other errors omitted\n";
                        }
                        _errors++;
                        // throw;
                    }
                }
            }

            min_cols = min(min_cols, record.size());

            // // convert each field to a double
            // // and add the newly-converted field to the end of the record
            // string cell;
            // double f;

            // while (getline(ss, cell, _delimiters[0])) {
            //     if (cell.empty()) continue;
            //     f = stod(cell);   
            //     record.push_back(f);
            // }

            // while (ss >> f) {
            //         record.push_back(f);
            // }

            _filedata.push_back(record);

            rows_read++;
        }
        
        // complain if something went wrong
        if (rows_read != _max_rows && !infile.eof())
        {
            cout << "loadtxt: could not read file (" << _fname << ")!\n";
            exit(1);
        }

        infile.close();

        size_t nlines = _filedata.size();
        
        // only read _max_rows of content
        if (_max_rows != 0)
        nlines = min(nlines, _max_rows);
        
        size_t ncols = min_cols;

        vector<int> cols;
        if (_usecols.size() == 0)
        {
            data.resize(ncols);
            cols.resize(ncols);
            std::iota(cols.begin(), cols.end(), 0);
        }
        else
        {
            data.resize(_usecols.size());
            cols = _usecols;
            //! usecols starts from 1!
            for (auto& j : cols)
                j--;
        }

        for (size_t i = 0; i < data.size(); i++)
            data[i].reserve(nlines);
        
        for (size_t i = 0; i < nlines; i++)
        {
            int k = 0;
            for (auto j : cols)
            {
                data[k].push_back(_filedata[i][j]);
                k++;
            }
        }
        
        return data;
    }

    ~loadtxt(){};

    public:
        vector<vector<T>> data;

    private:
        string _fname;
        string _comments = "#";
        string _delimiters = " \t,";
        size_t _skiprows = 0;
        vector<int> _usecols;
        size_t _max_rows = 0;
        vector<vector<T>> _filedata;

        size_t _max_errors = -1;
};


struct loadrdb : loadtxt<double> {
    loadrdb(string fname) : loadtxt<double>(fname), _fname(fname) { skiprows(2); }

    private:
        string _fname;
};




    /*
        Load data from a text file.

        Each row in the text file must have the same number of values.

        Parameters
        ----------
        fname : file, str, or pathlib.Path
            File, filename, or generator to read.  If the filename extension is
            ``.gz`` or ``.bz2``, the file is first decompressed. Note that
            generators should return byte strings.
        dtype : data-type, optional
            Data-type of the resulting array; default: float.  If this is a
            structured data-type, the resulting array will be 1-dimensional, and
            each row will be interpreted as an element of the array.  In this
            case, the number of columns used must match the number of fields in
            the data-type.
        comments : str or sequence of str, optional
            The characters or list of characters used to indicate the start of a
            comment. None implies no comments. For backwards compatibility, byte
            strings will be decoded as 'latin1'. The default is '#'.
        delimiter : str, optional
            The string used to separate values. For backwards compatibility, byte
            strings will be decoded as 'latin1'. The default is whitespace.
        converters : dict, optional
            A dictionary mapping column number to a function that will parse the
            column string into the desired value.  E.g., if column 0 is a date
            string: ``converters = {0: datestr2num}``.  Converters can also be
            used to provide a default value for missing data (but see also
            `genfromtxt`): ``converters = {3: lambda s: float(s.strip() or 0)}``.
            Default: None.
        skiprows : int, optional
            Skip the first `skiprows` lines, including comments; default: 0.
        usecols : int or sequence, optional
            Which columns to read, with 0 being the first. For example,
            ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
            The default, None, results in all columns being read.

            .. versionchanged:: 1.11.0
                When a single column has to be read it is possible to use
                an integer instead of a tuple. E.g ``usecols = 3`` reads the
                fourth column the same way as ``usecols = (3,)`` would.
        unpack : bool, optional
            If True, the returned array is transposed, so that arguments may be
            unpacked using ``x, y, z = loadtxt(...)``.  When used with a structured
            data-type, arrays are returned for each field.  Default is False.
        ndmin : int, optional
            The returned array will have at least `ndmin` dimensions.
            Otherwise mono-dimensional axes will be squeezed.
            Legal values: 0 (default), 1 or 2.

            .. versionadded:: 1.6.0
        encoding : str, optional
            Encoding used to decode the inputfile. Does not apply to input streams.
            The special value 'bytes' enables backward compatibility workarounds
            that ensures you receive byte arrays as results if possible and passes
            'latin1' encoded strings to converters. Override this value to receive
            unicode arrays and pass strings as input to converters.  If set to None
            the system default is used. The default value is 'bytes'.

            .. versionadded:: 1.14.0
        max_rows : int, optional
            Read `max_rows` lines of content after `skiprows` lines. The default
            is to read all the lines.

            .. versionadded:: 1.16.0
    */