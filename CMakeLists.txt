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

# DNest4 library
set(DNEST4_PATH src/vendor/DNest4/code)
FILE(GLOB DNEST4_SRC 
     ${DNEST4_PATH}/*.cpp
     ${DNEST4_PATH}/Distributions/*.cpp
     ${DNEST4_PATH}/RJObject/ConditionalPriors/*.cpp
)
add_library(dnest4 STATIC ${DNEST4_SRC})
target_compile_features(dnest4 PRIVATE cxx_std_17)

# paths for #include 
set(LOADTXT_PATH src/vendor/cpp-loadtxt/src)
set(GCEM_PATH src/vendor/gcem/include)
set(EIGEN_PATH src/vendor/eigen)


# import Python components needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# import nanobind
find_package(nanobind CONFIG REQUIRED)


# compile the extension modules
# this will make them available as kima.module

nanobind_add_module(math STABLE_ABI NB_STATIC NB_DOMAIN kima 
                    src/kima/math.cpp)

nanobind_add_module(distributions STABLE_ABI NB_STATIC NB_DOMAIN kima 
                    src/kima/distributions.cpp 
                    src/kima/InverseGamma.cpp src/kima/InverseMoment.cpp
                    src/kima/ExponentialRayleighMixture.cpp)

nanobind_add_module(Data STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/Data.cpp)

nanobind_add_module(ConditionalPrior STABLE_ABI NB_STATIC NB_DOMAIN kima 
                    src/kima/ConditionalPrior.cpp src/kima/Data.cpp)

# different models, with their corresponding dependencies
nanobind_add_module(RVmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/RVmodel.cpp src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/kepler.cpp src/kima/AMDstability.cpp)

nanobind_add_module(OutlierRVmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/OutlierRVmodel.cpp src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/kepler.cpp src/kima/AMDstability.cpp)

nanobind_add_module(TRANSITmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/TRANSITmodel.cpp src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/kepler.cpp src/kima/transits.cpp src/kima/AMDstability.cpp)

nanobind_add_module(GPmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/GPmodel.cpp src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/kepler.cpp src/kima/AMDstability.cpp src/kima/GP.cpp)


nanobind_add_module(RVFWHMmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/RVFWHMmodel.cpp src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/kepler.cpp src/kima/AMDstability.cpp)

nanobind_add_module(RVFWHMRHKmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/RVFWHMRHKmodel.cpp src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/kepler.cpp src/kima/AMDstability.cpp)

nanobind_add_module(BINARIESmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/BINARIESmodel.cpp src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/postkepler.cpp src/kima/kepler.cpp src/kima/AMDstability.cpp)
                    
nanobind_add_module(GAIAmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/GAIAmodel.cpp src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/kepler.cpp src/kima/AMDstability.cpp)
                    
nanobind_add_module(RVGAIAmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/RVGAIAmodel.cpp src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/postkepler.cpp src/kima/kepler.cpp src/kima/AMDstability.cpp)
                    
nanobind_add_module(ETVmodel STABLE_ABI NB_STATIC NB_DOMAIN kima
                    src/kima/ETVmodel.cpp src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/kepler.cpp)


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
                    src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/kepler.cpp src/kima/postkepler.cpp 
                    src/kima/transits.cpp src/kima/AMDstability.cpp
                    src/kima/GP.cpp)


nanobind_add_module(kepler STABLE_ABI NB_STATIC NB_DOMAIN kima 
                    src/kima/kepler.cpp)
target_compile_features(kepler PRIVATE cxx_std_17)

nanobind_add_module(postkepler STABLE_ABI NB_STATIC NB_DOMAIN kima 
                    src/kima/postkepler.cpp src/kima/kepler.cpp)
target_compile_features(postkepler PRIVATE cxx_std_17)

nanobind_add_module(GP STABLE_ABI NB_STATIC NB_DOMAIN kima 
                    src/kima/GP.cpp)
target_compile_features(GP PRIVATE cxx_std_17)

nanobind_add_module(transits STABLE_ABI NB_STATIC NB_DOMAIN kima 
                    src/kima/transits.cpp src/kima/kepler.cpp)
target_compile_features(transits PRIVATE cxx_std_17)


# add loadtxt path to includes
set(INCLUDES ${LOADTXT_PATH} ${DNEST4_PATH} ${GCEM_PATH})

# compilation settings for each module
foreach(targ distributions Data ConditionalPrior RVmodel OutlierRVmodel TRANSITmodel BINARIESmodel GAIAmodel RVGAIAmodel ETVmodel)
  target_compile_features(${targ} PRIVATE cxx_std_17)
  target_include_directories(${targ} PRIVATE ${INCLUDES})
  target_link_libraries(${targ} PRIVATE dnest4)
endforeach()

# add Eigen path to includes
set(INCLUDES_EIGEN ${INCLUDES} ${EIGEN_PATH})

# compilation settings for each module that requires Eigen
foreach(targ GPmodel RVFWHMmodel RVFWHMRHKmodel)
  target_compile_features(${targ} PRIVATE cxx_std_17)
  target_include_directories(${targ} PRIVATE ${INCLUDES_EIGEN})
  target_link_libraries(${targ} PRIVATE dnest4)
endforeach()
# and for GP module too
target_include_directories(GP PRIVATE ${INCLUDES_EIGEN})



# compilation settings for Sampler
target_compile_features(Sampler PRIVATE cxx_std_17)
target_include_directories(Sampler PRIVATE ${INCLUDES_EIGEN})
target_link_libraries(Sampler PRIVATE dnest4)


# install directives for scikit-build-core
foreach(module math distributions kepler GP transits)
  install(TARGETS ${module} LIBRARY DESTINATION kima)  
endforeach()

foreach(module Data ConditionalPrior)
  install(TARGETS ${module} LIBRARY DESTINATION kima)  
endforeach()

foreach(module RVmodel OutlierRVmodel TRANSITmodel 
        GPmodel RVFWHMmodel RVFWHMRHKmodel BINARIESmodel GAIAmodel RVGAIAmodel ETVmodel)
  install(TARGETS ${module} LIBRARY DESTINATION kima)  
endforeach()

install(TARGETS Sampler LIBRARY DESTINATION kima)
