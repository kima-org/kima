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
set(SPLEAF_PATH src/vendor/spleaf/src/spleaf)




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
find_package(Python 3.9
  REQUIRED COMPONENTS Interpreter Development.Module)

# import nanobind
find_package(nanobind CONFIG REQUIRED)


# add loadtxt path to includes
set(INCLUDES ${LOADTXT_PATH} ${DNEST4_PATH} ${GCEM_PATH})
# add Eigen path to includes
set(INCLUDES_EIGEN ${INCLUDES} ${EIGEN_PATH})
# add spleaf path to includes
set(INCLUDES_SPLEAF ${INCLUDES_EIGEN} ${SPLEAF_PATH})


set(COMMON_UNITS src/kima/ConditionalPrior.cpp)

# add_library(common_objects OBJECT ${COMMON_UNITS})
# target_compile_features(common_objects PRIVATE cxx_std_17)
# target_include_directories(common_objects PRIVATE ${INCLUDES})
# target_include_directories(common_objects PRIVATE ${NB_DIR}/include)
# target_include_directories(common_objects PRIVATE ${Python_INCLUDE_DIRS})

set(DISTRIBUTION_UNITS 
    src/kima/distributions/InverseGamma.cpp
    src/kima/distributions/InverseMoment.cpp
    src/kima/distributions/ExponentialRayleighMixture.cpp
    src/kima/distributions/GaussianMixture.cpp
    src/kima/distributions/BivariateGaussian.cpp
    src/kima/distributions/PowerLaw.cpp
    src/kima/distributions/Sine.cpp)

# set(COMMON_UNITS ${COMMON_UNITS} ${DISTRIBUTION_UNITS})

# compile the extension modules
# this will make them available as kima.module

nanobind_add_module(kmath NB_STATIC NB_DOMAIN kima 
                    src/kima/kmath.cpp)

add_library(distribution_objects OBJECT ${DISTRIBUTION_UNITS})
target_compile_features(distribution_objects PRIVATE cxx_std_17)
target_include_directories(distribution_objects PRIVATE ${INCLUDES})

add_library(common_objects OBJECT 
            src/kima/Data.cpp src/kima/kepler.cpp src/kima/postkepler.cpp
            src/kima/AMDstability.cpp src/kima/kmath.cpp
            src/kima/default_priors.cpp)
target_compile_features(common_objects PRIVATE cxx_std_17)
target_include_directories(common_objects PRIVATE ${INCLUDES})
target_include_directories(common_objects PRIVATE ${NB_DIR}/include)
target_include_directories(common_objects PRIVATE ${Python_INCLUDE_DIRS})

add_library(conditional OBJECT src/kima/ConditionalPrior.cpp)
target_compile_features(conditional PRIVATE cxx_std_17)
target_include_directories(conditional PRIVATE ${INCLUDES})
target_include_directories(conditional PRIVATE ${NB_DIR}/include)
target_include_directories(conditional PRIVATE ${Python_INCLUDE_DIRS})


add_library(gp_objects OBJECT src/kima/GP.cpp src/kima/bessel-library.cpp)
target_compile_features(gp_objects PRIVATE cxx_std_17)
target_include_directories(gp_objects PRIVATE ${INCLUDES_SPLEAF})
target_include_directories(gp_objects PRIVATE ${NB_DIR}/include)
target_include_directories(gp_objects PRIVATE ${Python_INCLUDE_DIRS})

nanobind_add_module(distributions NB_STATIC NB_DOMAIN kima 
                    src/kima/distributions.cpp
                    $<TARGET_OBJECTS:distribution_objects>)

nanobind_add_module(default_priors NB_STATIC NB_DOMAIN kima 
                    src/kima/default_priors.cpp src/kima/Data.cpp
                    $<TARGET_OBJECTS:distribution_objects>)

nanobind_add_module(Data NB_STATIC NB_DOMAIN kima
                    src/kima/Data.cpp)

# target_link_directories(common_objects PRIVATE ${Python_SABI_LIBRARY_DIRS})
# target_link_libraries(common_objects PRIVATE dnest4)

nanobind_add_module(ConditionalPrior NB_STATIC NB_DOMAIN kima 
                    src/kima/ConditionalPrior.cpp 
                    $<TARGET_OBJECTS:common_objects> $<TARGET_OBJECTS:distribution_objects>)

# different models, with their corresponding dependencies
nanobind_add_module(RVmodel NB_STATIC NB_DOMAIN kima
                    src/kima/RVmodel.cpp 
                    $<TARGET_OBJECTS:common_objects> $<TARGET_OBJECTS:conditional> $<TARGET_OBJECTS:distribution_objects>)
# nanobind_add_stub(RVmodel_stub MODULE kima.RVmodel OUTPUT RVmodel.pyi DEPENDS RVmodel)

nanobind_add_module(OutlierRVmodel NB_STATIC NB_DOMAIN kima
                    src/kima/OutlierRVmodel.cpp 
                    $<TARGET_OBJECTS:conditional> $<TARGET_OBJECTS:common_objects> $<TARGET_OBJECTS:distribution_objects>)

nanobind_add_module(TRANSITmodel NB_STATIC NB_DOMAIN kima
                    src/kima/TRANSITmodel.cpp src/kima/transits.cpp
                    $<TARGET_OBJECTS:conditional> $<TARGET_OBJECTS:common_objects> $<TARGET_OBJECTS:distribution_objects>)

nanobind_add_module(GPmodel NB_STATIC NB_DOMAIN kima
                    src/kima/GPmodel.cpp $<TARGET_OBJECTS:conditional> 
                    $<TARGET_OBJECTS:gp_objects> $<TARGET_OBJECTS:common_objects> $<TARGET_OBJECTS:distribution_objects>)

nanobind_add_module(SPLEAFmodel NB_STATIC NB_DOMAIN kima
                    src/kima/SPLEAFmodel.cpp $<TARGET_OBJECTS:conditional> 
                    $<TARGET_OBJECTS:gp_objects> $<TARGET_OBJECTS:common_objects> $<TARGET_OBJECTS:distribution_objects>)

nanobind_add_module(RVFWHMmodel NB_STATIC NB_DOMAIN kima
                    src/kima/RVFWHMmodel.cpp $<TARGET_OBJECTS:conditional> 
                    $<TARGET_OBJECTS:gp_objects> $<TARGET_OBJECTS:common_objects> $<TARGET_OBJECTS:distribution_objects>)

nanobind_add_module(RVFWHMRHKmodel NB_STATIC NB_DOMAIN kima
                    src/kima/RVFWHMRHKmodel.cpp 
                    $<TARGET_OBJECTS:conditional> $<TARGET_OBJECTS:common_objects> $<TARGET_OBJECTS:distribution_objects>
                    $<TARGET_OBJECTS:gp_objects>)

nanobind_add_module(BINARIESmodel NB_STATIC NB_DOMAIN kima
                    src/kima/BINARIESmodel.cpp
                    $<TARGET_OBJECTS:conditional> $<TARGET_OBJECTS:common_objects> $<TARGET_OBJECTS:distribution_objects>)
                    
nanobind_add_module(GAIAmodel NB_STATIC NB_DOMAIN kima
                    src/kima/GAIAmodel.cpp 
                    $<TARGET_OBJECTS:conditional> $<TARGET_OBJECTS:common_objects> $<TARGET_OBJECTS:distribution_objects>)
                    
nanobind_add_module(RVGAIAmodel NB_STATIC NB_DOMAIN kima
                    src/kima/RVGAIAmodel.cpp
                    $<TARGET_OBJECTS:conditional> $<TARGET_OBJECTS:common_objects> $<TARGET_OBJECTS:distribution_objects>)
                    
nanobind_add_module(ETVmodel NB_STATIC NB_DOMAIN kima
                    src/kima/ETVmodel.cpp 
                    $<TARGET_OBJECTS:conditional> $<TARGET_OBJECTS:common_objects> $<TARGET_OBJECTS:distribution_objects>)

nanobind_add_module(ApodizedRVmodel NB_STATIC NB_DOMAIN kima
                    src/kima/ApodizedRVmodel.cpp 
                    $<TARGET_OBJECTS:conditional> $<TARGET_OBJECTS:common_objects> $<TARGET_OBJECTS:distribution_objects>)

nanobind_add_module(RVHGPMmodel NB_STATIC NB_DOMAIN kima
                    src/kima/RVHGPMmodel.cpp 
                    $<TARGET_OBJECTS:conditional> $<TARGET_OBJECTS:common_objects> $<TARGET_OBJECTS:distribution_objects>)


# compile the Sampler module to create the kima.run function
nanobind_add_module(Sampler NB_STATIC NB_DOMAIN kima
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
                    src/kima/transits.cpp
                    $<TARGET_OBJECTS:common_objects> 
                    $<TARGET_OBJECTS:gp_objects> 
                    $<TARGET_OBJECTS:distribution_objects>
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
target_include_directories(AMDstability PRIVATE ${INCLUDES})

nanobind_add_module(GP STABLE_ABI NB_STATIC NB_DOMAIN kima 
                    src/kima/GP.cpp src/kima/bessel-library.cpp)
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
install(TARGETS distributions    LIBRARY DESTINATION kima)
install(TARGETS default_priors   LIBRARY DESTINATION kima)
install(TARGETS kmath            LIBRARY DESTINATION kima)  
install(TARGETS kepler           LIBRARY DESTINATION kima)
install(TARGETS postkepler       LIBRARY DESTINATION kima)
install(TARGETS GP               LIBRARY DESTINATION kima)
install(TARGETS transits         LIBRARY DESTINATION kima)
install(TARGETS AMDstability     LIBRARY DESTINATION kima)
install(TARGETS Data             LIBRARY DESTINATION kima)
install(TARGETS ConditionalPrior LIBRARY DESTINATION kima)
install(TARGETS RVmodel          LIBRARY DESTINATION kima)
install(TARGETS GPmodel          LIBRARY DESTINATION kima)
install(TARGETS RVFWHMmodel      LIBRARY DESTINATION kima)
install(TARGETS TRANSITmodel     LIBRARY DESTINATION kima)
install(TARGETS RVHGPMmodel      LIBRARY DESTINATION kima)
install(TARGETS RVFWHMRHKmodel   LIBRARY DESTINATION kima)
install(TARGETS SPLEAFmodel      LIBRARY DESTINATION kima)
install(TARGETS BINARIESmodel    LIBRARY DESTINATION kima)
install(TARGETS GAIAmodel        LIBRARY DESTINATION kima)
install(TARGETS RVGAIAmodel      LIBRARY DESTINATION kima)
install(TARGETS ETVmodel         LIBRARY DESTINATION kima)
install(TARGETS OutlierRVmodel   LIBRARY DESTINATION kima)
install(TARGETS ApodizedRVmodel  LIBRARY DESTINATION kima)
install(TARGETS Sampler          LIBRARY DESTINATION kima)

# foreach(module distributions kepler GP Data RVmodel OutlierRVmodel ApodizedRVmodel RVHGPMmodel TRANSITmodel)
foreach(module Sampler kepler)
  nanobind_add_stub(${module}_stub MODULE kima.${module} OUTPUT ${module}.pyi DEPENDS ${module})
endforeach()