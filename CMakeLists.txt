cmake_minimum_required(VERSION 3.15...3.26)

project(kima LANGUAGES CXX)

# import Python components needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# import nanobind
find_package(nanobind CONFIG REQUIRED)

# compile the extension modules
nanobind_add_module(Data STABLE_ABI NB_STATIC src/kima/Data.cpp)

set(INCLUDES src/vendor/cpp-loadtxt/src)
target_include_directories(Data PRIVATE ${INCLUDES})

# install directive for scikit-build-core
install(TARGETS Data LIBRARY DESTINATION kima)
