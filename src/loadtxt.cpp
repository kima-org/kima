#include <assert.h>
#include "loadtxt.hpp"

using namespace std;

int main() {
    { // test 1
        std::string fname = "data.txt";
        std::ifstream f(fname);
        auto data = loadtxt(fname).comments("#")();
        std::cout << "Read " << data.size() << " columns, each with " << data[0].size() << " rows from file '" << fname << "'\n";

        assert(data.size() == 4 && "should have read 4 columns");
        assert(data[0].size() == 100 && "should have read 100 rows");
        assert(data[1].size() == 100 && "should have read 100 rows");
        assert(data[2].size() == 100 && "should have read 100 rows");
        assert(data[3].size() == 100 && "should have read 100 rows");
    }
    { // test 2
        auto fname = "data.rdb";
        auto data = loadtxt(fname).skiprows(2)();
        std::cout << "Read " << data.size() << " columns, each with " << data[0].size() << " rows from file '" << fname << "'\n";

        assert(data.size() == 3 && "should have read 3 columns");
        assert(data[0].size() == 5 && "should have read 5 rows");
        assert(data[1].size() == 5 && "should have read 5 rows");
        assert(data[2].size() == 5 && "should have read 5 rows");
    }
    { // test 3
        auto fname = "data.txt";
        auto data = loadtxt(fname).skiprows(10).comments("#")();
        std::cout << "Read " << data.size() << " columns, each with " << data[0].size() << " rows from file '" << fname << "'\n";

        assert(data.size() == 4 && "should have read 4 columns");
        assert(data[0].size() == 91 && "should have read 91 rows");
        assert(data[1].size() == 91 && "should have read 91 rows");
        assert(data[2].size() == 91 && "should have read 91 rows");
        assert(data[3].size() == 91 && "should have read 91 rows");
    }
    { // test 4
        auto fname = "data.txt";
        auto data = loadtxt(fname).skiprows(10).max_rows(8).comments("#")();
        std::cout << "Read " << data.size() << " columns, each with " << data[0].size() << " rows from file '" << fname << "'\n";

        assert(data.size() == 4 && "should have read 4 columns");
        assert(data[0].size() == 8 && "should have read 8 rows");
        assert(data[1].size() == 8 && "should have read 8 rows");
        assert(data[2].size() == 8 && "should have read 8 rows");
        assert(data[3].size() == 8 && "should have read 8 rows");
    }
    { // test 5
        auto fname = "data.csv";
        auto data = loadtxt(fname)();
        std::cout << "Read " << data.size() << " columns, each with " << data[0].size() << " rows from file '" << fname << "'\n";

        assert(data.size() == 4 && "should have read 4 columns");
        assert(data[0].size() == 8 && "should have read 8 rows");
        assert(data[1].size() == 8 && "should have read 8 rows");
        assert(data[2].size() == 8 && "should have read 8 rows");
        assert(data[3].size() == 8 && "should have read 8 rows");
    }
    { // test 6
        auto fname = "data.tsv";
        auto data = loadtxt(fname)();
        std::cout << "Read " << data.size() << " columns, each with " << data[0].size() << " rows from file '" << fname << "'\n";

        assert(data.size() == 3 && "should have read 3 columns");
        assert(data[0].size() == 20 && "should have read 20 rows");
        assert(data[1].size() == 20 && "should have read 20 rows");
        assert(data[2].size() == 20 && "should have read 20 rows");
    }
    { // test 7
        auto fname = "complicated_data.rdb";
        auto data = loadtxt(fname).max_errors(1)();
        std::cout << "Read " << data.size() << " columns, each with " << data[0].size() << " rows from file '" << fname << "'\n";

        assert(data.size() == 74 && "should have read 74 columns");
        assert(data[0].size() == 3 && "should have read 3 rows");
        assert(data[1].size() == 3 && "should have read 3 rows");
        assert(data[2].size() == 3 && "should have read 3 rows");
        assert(std::isnan(data[7][0]) && "should have found NaN in column 8");
    }

    std::cout << "Done reading!\n";
    return 0;
}