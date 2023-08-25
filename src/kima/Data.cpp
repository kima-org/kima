#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;
using namespace nb::literals;

#include "Data.h"

/**
 * @brief Find pathnames matching a pattern
 *
 * from https://stackoverflow.com/a/8615450
 */
std::vector<std::string> glob(const std::string& pattern)
{
    // glob struct resides on the stack
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    // do the glob operation
    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if (return_value != 0) {
        globfree(&glob_result);
        std::stringstream ss;
        ss << "glob() failed with return_value " << return_value << std::endl;
        throw std::runtime_error(ss.str());
    }

    // collect all the filenames into a vector<string>
    std::vector<std::string> filenames;
    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
        filenames.push_back(std::string(glob_result.gl_pathv[i]));
    }

    // cleanup
    globfree(&glob_result);

    return filenames;
}


RVData::RVData() {};


void RVData::load(const std::string filename, const std::string units, int skip,
                  const std::string delimiter, const std::vector<std::string>& indicators)
{
    auto data = loadtxt(filename)
                    .skiprows(skip)
                    .delimiter(delimiter)();
}


NB_MODULE(Data, m) {
    m.def("add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);
    m.def("glob", [](const std::string& pattern) { return glob(pattern); });
    // 
    nb::class_<RVData>(m, "RVData")
        // .def(nb::init<>())
        .def("load", &RVData::load, "filename"_a, "units"_a, "skip"_a=2, "delimiter"_a=" ", "indicators"_a,
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
)D")
// )
        .def("N", &RVData::N, "Total number of points");
}
