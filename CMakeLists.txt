cmake_minimum_required(VERSION 3.17)

project(kima LANGUAGES CXX C)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(WIN32)
  add_compile_definitions(_USE_MATH_DEFINES)
endif()

# if (MSVC)
#     # warning level 3
#     add_compile_options(/W3)
# else()
#     # additional warnings
#     add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

# add_compile_options(-Wall -Wextra -Wpedantic)

# paths for #include 
set(LOADTXT_PATH src/vendor/cpp-loadtxt/src)
set(GCEM_PATH src/vendor/gcem/include)
set(EIGEN_PATH src/vendor/eigen)
set(SPLEAF_PATH src/vendor/spleaf/spleaf)




# find_package(HDF5 REQUIRED COMPONENTS CXX
#              PATHS C:/Users/joaof/Downloads/hdf5-1.14.4-2-win-vs2022_intel/hdf5/HDF5-1.14.4-win64/HDF5-1.14.4-win64)
# add_subdirectory(src/vendor/HighFive)


# DNest4 library
set(DNEST4_PATH src/vendor/DNest4/code)
FILE(GLOB DNEST4_SRC 
${DNEST4_PATH}/*.cpp
${DNEST4_PATH}/Distributions/*.cpp
${DNEST4_PATH}/RJObject/ConditionalPriors/*.cpp
)
add_library(dnest4 STATIC ${DNEST4_SRC})
target_compile_features(dnest4 PRIVATE cxx_std_17)


add_library(libspleaf STATIC ${SPLEAF_PATH}/libspleaf.c)


# import Python components needed by nanobind
find_package(Python 3.8
REQUIRED COMPONENTS Interpreter Development.Module
OPTIONAL_COMPONENTS Development.SABIModule)

# import nanobind
find_package(nanobind CONFIG REQUIRED)


# add loadtxt path to includes
set(INCLUDES ${LOADTXT_PATH} ${DNEST4_PATH} ${GCEM_PATH})
# add Eigen path to includes
set(INCLUDES_EIGEN ${INCLUDES} ${EIGEN_PATH})
# add spleaf path to includes
set(INCLUDES_SPLEAF ${INCLUDES_EIGEN} ${SPLEAF_PATH})


set(COMMON_UNITS src/kima/Data.cpp src/kima/ConditionalPrior.cpp 
                 src/kima/default_priors.cpp
                 src/kima/kepler.cpp src/kima/postkepler.cpp src/kima/AMDstability.cpp src/kima/kmath.cpp)

set(GP_UNITS src/kima/GP.cpp src/kima/bessel-library.cpp)

set(DISTRIBUTION_UNITS 
    src/kima/distributions/InverseGamma.cpp
    src/kima/distributions/InverseMoment.cpp
    src/kima/distributions/ExponentialRayleighMixture.cpp
    src/kima/distributions/GaussianMixture.cpp
    src/kima/distributions/BivariateGaussian.cpp
    src/kima/distributions/PowerLaw.cpp
    src/kima/distributions/Sine.cpp)

set(COMMON_UNITS ${COMMON_UNITS} ${DISTRIBUTION_UNITS})

# compile the extension modules
# this will make them available as kima.module

nanobind_add_module(kmath STABLE_ABI NB_STATIC NB_DOMAIN kima 
                    src/kima/kmath.cpp)

nanobind_add_module(distributions STABLE_ABI NB_STATIC NB_DOMAIN kima 
                    src/kima/distributions.cpp ${DISTRIBUTION_UNITS})

nanobind_add_module(default_priors STABLE_ABI NB_STATIC NB_DOMAIN kima 
                    src/kima/default_priors.cpp src/kima/Data.cpp ${DISTRIBUTION_UNITS})

nanobind_add_module(Data STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/Data.cpp)

nanobind_add_module(ConditionalPrior STABLE_ABI NB_STATIC NB_DOMAIN kima 
                    ${COMMON_UNITS})

# different models, with their corresponding dependencies
nanobind_add_module(RVmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/RVmodel.cpp ${COMMON_UNITS})

nanobind_add_module(OutlierRVmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/OutlierRVmodel.cpp ${COMMON_UNITS})

nanobind_add_module(TRANSITmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/TRANSITmodel.cpp 
                    ${COMMON_UNITS} src/kima/transits.cpp)

nanobind_add_module(GPmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/GPmodel.cpp ${COMMON_UNITS} ${GP_UNITS})

nanobind_add_module(SPLEAFmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/SPLEAFmodel.cpp ${COMMON_UNITS} ${GP_UNITS})

nanobind_add_module(RVFWHMmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/RVFWHMmodel.cpp ${COMMON_UNITS} ${GP_UNITS})

nanobind_add_module(RVFWHMRHKmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/RVFWHMRHKmodel.cpp ${COMMON_UNITS} ${GP_UNITS})

nanobind_add_module(BINARIESmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/BINARIESmodel.cpp 
                    ${COMMON_UNITS} src/kima/postkepler.cpp)
                    
nanobind_add_module(GAIAmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/GAIAmodel.cpp ${COMMON_UNITS})
                    
nanobind_add_module(RVGAIAmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/RVGAIAmodel.cpp
                    ${COMMON_UNITS} src/kima/postkepler.cpp)
                    
nanobind_add_module(ETVmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/ETVmodel.cpp ${COMMON_UNITS})

nanobind_add_module(ApodizedRVmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/ApodizedRVmodel.cpp ${COMMON_UNITS})

nanobind_add_module(RVHGPMmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/RVHGPMmodel.cpp ${COMMON_UNITS})


# compile the Sampler module to create the kima.run function
nanobind_add_module(Sampler STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/run.cpp
                    src/kima/RVmodel.cpp
                    src/kima/OutlierRVmodel.cpp
                    src/kima/TRANSITmodel.cpp
                    src/kima/GPmodel.cpp
                    src/kima/RVFWHMmodel.cpp
                    src/kima/BINARIESmodel.cpp
                    src/kima/GAIAmodel.cpp
                    src/kima/RVGAIAmodel.cpp
                    src/kima/RVFWHMRHKmodel.cpp
                    src/kima/ETVmodel.cpp
                    src/kima/SPLEAFmodel.cpp
                    src/kima/ApodizedRVmodel.cpp
                    src/kima/RVHGPMmodel.cpp
                    ${COMMON_UNITS}
                    src/kima/postkepler.cpp 
                    src/kima/transits.cpp
                    ${GP_UNITS}
                    ${DISTRIBUTION_UNITS}
                    )


nanobind_add_module(kepler STABLE_ABI NB_STATIC NB_DOMAIN kima 
                    src/kima/kepler.cpp)
target_compile_features(kepler PRIVATE cxx_std_17)

nanobind_add_module(postkepler STABLE_ABI NB_STATIC NB_DOMAIN kima 
                    src/kima/postkepler.cpp src/kima/kepler.cpp)
target_compile_features(postkepler PRIVATE cxx_std_17)

nanobind_add_module(AMDstability STABLE_ABI NB_STATIC NB_DOMAIN kima 
                    src/kima/AMDstability.cpp)
target_compile_features(AMDstability PRIVATE cxx_std_17)

nanobind_add_module(GP STABLE_ABI NB_STATIC NB_DOMAIN kima 
                    ${GP_UNITS})
target_compile_features(GP PRIVATE cxx_std_17)

nanobind_add_module(transits STABLE_ABI NB_STATIC NB_DOMAIN kima 
                    src/kima/transits.cpp src/kima/kepler.cpp)
target_compile_features(transits PRIVATE cxx_std_17)


# compilation settings for each module
foreach(targ distributions default_priors Data ConditionalPrior 
        RVmodel OutlierRVmodel TRANSITmodel BINARIESmodel GAIAmodel RVGAIAmodel ETVmodel
        ApodizedRVmodel RVHGPMmodel)
  target_compile_features(${targ} PRIVATE cxx_std_17)
  target_include_directories(${targ} PRIVATE ${INCLUDES})
  target_link_libraries(${targ} PRIVATE dnest4)
endforeach()


# compilation settings for each module that requires Eigen
foreach(targ GPmodel RVFWHMmodel RVFWHMRHKmodel SPLEAFmodel)
  target_compile_features(${targ} PRIVATE cxx_std_17)
  target_include_directories(${targ} PRIVATE ${INCLUDES_SPLEAF})
  target_link_libraries(${targ} PRIVATE dnest4)
  target_link_libraries(${targ} PRIVATE libspleaf)
endforeach()
# and for GP module too
target_include_directories(GP PRIVATE ${INCLUDES_SPLEAF})
target_link_libraries(GP PRIVATE libspleaf)



# compilation settings for Sampler
target_compile_features(Sampler PRIVATE cxx_std_17)
target_include_directories(Sampler PRIVATE ${INCLUDES_SPLEAF})
target_link_libraries(Sampler PRIVATE dnest4)
target_link_libraries(Sampler PRIVATE libspleaf)


# install directives for scikit-build-core
foreach(module kmath distributions default_priors kepler postkepler GP transits AMDstability)
  install(TARGETS ${module} LIBRARY DESTINATION kima)  
endforeach()

foreach(module Data ConditionalPrior)
  install(TARGETS ${module} LIBRARY DESTINATION kima)  
endforeach()

foreach(module RVmodel OutlierRVmodel ApodizedRVmodel RVHGPMmodel TRANSITmodel 
        GPmodel RVFWHMmodel RVFWHMRHKmodel SPLEAFmodel
        BINARIESmodel GAIAmodel RVGAIAmodel ETVmodel)
  install(TARGETS ${module} LIBRARY DESTINATION kima)  
endforeach()

install(TARGETS Sampler LIBRARY DESTINATION kima)
