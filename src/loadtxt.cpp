#include <assert.h>
#include "loadtxt.hpp"

using namespace std;

int main() {
    { // test 1
        auto fname = "data.txt";
        auto data = loadtxt(fname).comments("#")();
        printf("Read %ld columns, each with %ld rows from file '%s'\n", 
               data.size(), data[0].size(), fname);

        assert(data.size() == 4 && "should have read 4 columns");
        assert(data[0].size() == 100 && "should have read 100 rows");
        assert(data[1].size() == 100 && "should have read 100 rows");
        assert(data[2].size() == 100 && "should have read 100 rows");
        assert(data[3].size() == 100 && "should have read 100 rows");
    }
    { // test 2
        auto fname = "data.rdb";
        auto data = loadrdb(fname)();
        printf("Read %ld columns, each with %ld rows from file '%s'\n", 
               data.size(), data[0].size(), fname);

        assert(data.size() == 3 && "should have read 3 columns");
        assert(data[0].size() == 5 && "should have read 5 rows");
        assert(data[1].size() == 5 && "should have read 5 rows");
        assert(data[2].size() == 5 && "should have read 5 rows");
    }
    { // test 3
        auto fname = "data.txt";
        auto data = loadrdb(fname).skiprows(10).comments("#")();
        printf("Read %ld columns, each with %ld rows from file '%s'\n", 
               data.size(), data[0].size(), fname);

        assert(data.size() == 4 && "should have read 4 columns");
        assert(data[0].size() == 91 && "should have read 91 rows");
        assert(data[1].size() == 91 && "should have read 91 rows");
        assert(data[2].size() == 91 && "should have read 91 rows");
    }
    { // test 4
        auto fname = "data.txt";
        auto data = loadrdb(fname).skiprows(10).max_rows(8).comments("#")();
        printf("Read %ld columns, each with %ld rows from file '%s'\n", 
               data.size(), data[0].size(), fname);

        assert(data.size() == 4 && "should have read 4 columns");
        assert(data[0].size() == 8 && "should have read 8 rows");
        assert(data[1].size() == 8 && "should have read 8 rows");
        assert(data[2].size() == 8 && "should have read 8 rows");
    }
    printf("Done reading!\n");
    return 0;
}