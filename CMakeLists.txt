cmake_minimum_required(VERSION 3.19...3.26)

project(kima LANGUAGES CXX C)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# add_compile_options(-Wall -Wextra -Wpedantic)

# DNest4 library
set(DNEST4_PATH src/vendor/DNest4/code)
FILE(GLOB DNEST4_SRC 
     ${DNEST4_PATH}/*.cpp
     ${DNEST4_PATH}/Distributions/*.cpp
     ${DNEST4_PATH}/RJObject/ConditionalPriors/*.cpp
)
add_library(dnest4 STATIC ${DNEST4_SRC})
target_compile_features(dnest4 PRIVATE cxx_std_17)

# path to include Eigen
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
# this will make them available as kima.module

nanobind_add_module(distributions STABLE_ABI NB_STATIC 
                    src/kima/distributions.cpp)

nanobind_add_module(Data STABLE_ABI NB_STATIC
                    src/kima/Data.cpp)

nanobind_add_module(ConditionalPrior STABLE_ABI NB_STATIC 
                    src/kima/ConditionalPrior.cpp)

# different models, with their corresponding dependencies
nanobind_add_module(RVmodel STABLE_ABI NB_STATIC
                    src/kima/RVmodel.cpp src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/kepler.cpp src/kima/AMDstability.cpp)

nanobind_add_module(OutlierRVmodel STABLE_ABI NB_STATIC
                    src/kima/OutlierRVmodel.cpp src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/kepler.cpp src/kima/AMDstability.cpp)

nanobind_add_module(GPmodel STABLE_ABI NB_STATIC
                    src/kima/GPmodel.cpp src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/kepler.cpp src/kima/AMDstability.cpp)

nanobind_add_module(RVFWHMmodel STABLE_ABI NB_STATIC
                    src/kima/RVFWHMmodel.cpp src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/kepler.cpp src/kima/AMDstability.cpp)

nanobind_add_module(SPLEAFmodel STABLE_ABI NB_STATIC
                    src/kima/SPLEAFmodel.cpp src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/spleaf.cpp src/kima/kepler.cpp src/kima/AMDstability.cpp)


# compile the Sampler module to create the kima.run function
nanobind_add_module(Sampler STABLE_ABI NB_STATIC
                    src/kima/run.cpp
                    src/kima/RVmodel.cpp
                    src/kima/OutlierRVmodel.cpp
                    src/kima/GPmodel.cpp
                    src/kima/RVFWHMmodel.cpp
                    src/kima/SPLEAFmodel.cpp
                    src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/spleaf.cpp src/kima/kepler.cpp src/kima/AMDstability.cpp)


nanobind_add_module(kepler STABLE_ABI NB_STATIC 
                    src/kima/kepler.cpp)

target_compile_features(kepler PRIVATE cxx_std_17)



# add loadtxt path to includes
set(INCLUDES src/vendor/cpp-loadtxt/src ${DNEST4_PATH})

# compilation settings for each module
foreach(targ distributions Data ConditionalPrior RVmodel OutlierRVmodel)
  target_compile_features(${targ} PRIVATE cxx_std_17)
  target_include_directories(${targ} PRIVATE ${INCLUDES})
  target_link_libraries(${targ} PRIVATE dnest4)
endforeach()

# add Eigen path to includes
set(INCLUDES_EIGEN ${INCLUDES} src/vendor/eigen)

# compilation settings for each module that requires Eigen
foreach(targ GPmodel RVFWHMmodel)
  target_compile_features(${targ} PRIVATE cxx_std_17)
  target_include_directories(${targ} PRIVATE ${INCLUDES_EIGEN})
  target_link_libraries(${targ} PRIVATE dnest4)
endforeach()


# spleaf bindings and C++ wrapper
nanobind_add_module(spleaf STABLE_ABI NB_STATIC
                    src/kima/spleaf.cpp)

target_compile_features(spleaf PRIVATE cxx_std_17)
target_include_directories(spleaf PRIVATE ${SPLEAF_PATH})
target_include_directories(spleaf PRIVATE ${INCLUDES_EIGEN})
target_link_libraries(spleaf PRIVATE libspleaf)
target_link_libraries(spleaf PRIVATE dnest4)

# SPLEAFmodel
set(INCLUDES_SPLEAF ${INCLUDES_EIGEN} ${SPLEAF_PATH})

target_compile_features(SPLEAFmodel PRIVATE cxx_std_17)
target_include_directories(SPLEAFmodel PRIVATE ${INCLUDES_SPLEAF})
target_link_libraries(SPLEAFmodel PRIVATE dnest4)
target_link_libraries(SPLEAFmodel PRIVATE libspleaf)


# compilation settings for Sampler
target_compile_features(Sampler PRIVATE cxx_std_17)
target_include_directories(Sampler PRIVATE ${INCLUDES_SPLEAF})
target_link_libraries(Sampler PRIVATE dnest4)
target_link_libraries(Sampler PRIVATE libspleaf)


# install directives for scikit-build-core
foreach(module distributions kepler transits spleaf)
  install(TARGETS ${module} LIBRARY DESTINATION kima)  
endforeach()

foreach(module Data ConditionalPrior)
  install(TARGETS ${module} LIBRARY DESTINATION kima)  
endforeach()

foreach(module RVmodel OutlierRVmodel GPmodel RVFWHMmodel SPLEAFmodel)
  install(TARGETS ${module} LIBRARY DESTINATION kima)  
endforeach()

install(TARGETS Sampler LIBRARY DESTINATION kima)
