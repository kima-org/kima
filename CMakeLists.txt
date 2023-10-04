cmake_minimum_required(VERSION 3.19...3.26)

project(kima LANGUAGES CXX C)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# add_compile_options(-Wall -Wextra -Wpedantic)

# if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
#   set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
#   set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
# endif()

# DNest4 library
set(DNEST4_PATH src/vendor/DNest4/code)
FILE(GLOB DNEST4_SRC 
     ${DNEST4_PATH}/*.cpp
     ${DNEST4_PATH}/Distributions/*.cpp
     ${DNEST4_PATH}/RJObject/ConditionalPriors/*.cpp
)
add_library(dnest4 STATIC ${DNEST4_SRC})
target_compile_features(dnest4 PRIVATE cxx_std_17)

# Eigen
set(EIGEN_PATH src/vendor/eigen)

# spleaf library
set(SPLEAF_PATH src/vendor/spleaf/spleaf)
add_library(libspleaf STATIC ${SPLEAF_PATH}/libspleaf.c)


# import Python components needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# import nanobind
find_package(nanobind CONFIG REQUIRED)

# compile the extension modules

nanobind_add_module(distributions STABLE_ABI NB_STATIC 
                    src/kima/distributions.cpp)


nanobind_add_module(Data STABLE_ABI NB_STATIC src/kima/Data.cpp)

nanobind_add_module(ConditionalPrior STABLE_ABI NB_STATIC src/kima/ConditionalPrior.cpp)

nanobind_add_module(RVmodel STABLE_ABI NB_STATIC
                    src/kima/RVmodel.cpp src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/kepler.cpp src/kima/AMDstability.cpp)

nanobind_add_module(GPmodel STABLE_ABI NB_STATIC
                    src/kima/GPmodel.cpp src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/kepler.cpp src/kima/AMDstability.cpp)

nanobind_add_module(RVFWHMmodel STABLE_ABI NB_STATIC
                    src/kima/RVFWHMmodel.cpp src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/kepler.cpp src/kima/AMDstability.cpp)

nanobind_add_module(Sampler STABLE_ABI NB_STATIC
                    src/kima/run.cpp
                    src/kima/RVmodel.cpp src/kima/GPmodel.cpp src/kima/RVFWHMmodel.cpp
                    src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/kepler.cpp src/kima/AMDstability.cpp)


nanobind_add_module(kepler STABLE_ABI NB_STATIC 
                    src/kima/kepler.cpp)

target_compile_features(kepler PRIVATE cxx_std_17)


set(INCLUDES src/vendor/cpp-loadtxt/src ${DNEST4_PATH})

foreach(targ distributions Data ConditionalPrior RVmodel)
  target_compile_features(${targ} PRIVATE cxx_std_17)
  target_include_directories(${targ} PRIVATE ${INCLUDES})
  target_link_libraries(${targ} PRIVATE dnest4)
endforeach()

set(INCLUDES_EIGEN ${INCLUDES} src/vendor/eigen)

foreach(targ GPmodel RVFWHMmodel Sampler)
  target_compile_features(${targ} PRIVATE cxx_std_17)
  target_include_directories(${targ} PRIVATE ${INCLUDES_EIGEN})
  target_link_libraries(${targ} PRIVATE dnest4)
endforeach()


# spleaf bindings and C++ wrapper
set(SPLEAF_PATH src/vendor/spleaf/spleaf)
add_library(libspleaf STATIC ${SPLEAF_PATH}/libspleaf.c)

nanobind_add_module(spleaf STABLE_ABI NB_STATIC
                    src/kima/spleaf.cpp)

target_compile_features(spleaf PRIVATE cxx_std_17)
target_include_directories(spleaf PRIVATE ${SPLEAF_PATH})
target_include_directories(spleaf PRIVATE ${INCLUDES_EIGEN})
target_link_libraries(spleaf PRIVATE libspleaf)


# install directive for scikit-build-core
install(TARGETS distributions LIBRARY DESTINATION kima)
# 
install(TARGETS Data LIBRARY DESTINATION kima)
install(TARGETS ConditionalPrior LIBRARY DESTINATION kima)
install(TARGETS RVmodel LIBRARY DESTINATION kima)
install(TARGETS GPmodel LIBRARY DESTINATION kima)
install(TARGETS RVFWHMmodel LIBRARY DESTINATION kima)
install(TARGETS Sampler LIBRARY DESTINATION kima)
# 
install(TARGETS kepler LIBRARY DESTINATION kima)
install(TARGETS spleaf LIBRARY DESTINATION kima)
