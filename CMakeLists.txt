cmake_minimum_required(VERSION 3.15...3.26)

project(kima LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)


set(DNEST4_PATH src/vendor/DNest4/code)
FILE(GLOB DNEST4_SRC 
     ${DNEST4_PATH}/*.cpp
     ${DNEST4_PATH}/Distributions/*.cpp
     ${DNEST4_PATH}/RJObject/ConditionalPriors/*.cpp
)

add_library(dnest4 STATIC ${DNEST4_SRC})
target_compile_features(dnest4 PRIVATE cxx_std_17)


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
