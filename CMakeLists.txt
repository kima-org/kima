cmake_minimum_required(VERSION 3.19...3.26)

project(kima LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Deployment target/architectures for macOS
if(APPLE)
  # on macOS "uname -m" returns the architecture (x86_64 or arm64)
  execute_process(
      COMMAND uname -m
      RESULT_VARIABLE result
      OUTPUT_VARIABLE OSX_NATIVE_ARCHITECTURE
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  # determine if we do universal build or native build
  if((CMAKE_GENERATOR STREQUAL "Xcode")                # works only with Xcode
      AND (OSX_NATIVE_ARCHITECTURE STREQUAL "arm64"))  # and only when running on arm64
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "")
    set(ARCHIVE_ARCHITECTURE "macOS_universal")
    message(STATUS "macOS universal (x86_64 / arm64) build")
  else()
    set(ARCHIVE_ARCHITECTURE "macOS_${JAMBA_OSX_NATIVE_ARCHITECTURE}")
    message(STATUS "macOS native ${JAMBA_OSX_NATIVE_ARCHITECTURE} build")
  endif()
endif()

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

nanobind_add_module(Sampler STABLE_ABI NB_STATIC
                    src/kima/run.cpp
                    src/kima/RVmodel.cpp src/kima/GPmodel.cpp 
                    src/kima/Data.cpp src/kima/ConditionalPrior.cpp
                    src/kima/kepler.cpp src/kima/AMDstability.cpp)


nanobind_add_module(kepler STABLE_ABI NB_STATIC 
                    src/kima/kepler.cpp)

target_compile_features(kepler PRIVATE cxx_std_17)



set(INCLUDES src/vendor/cpp-loadtxt/src src/vendor/DNest4/code)

foreach(targ distributions Data ConditionalPrior RVmodel)
  target_compile_features(${targ} PRIVATE cxx_std_17)
  target_include_directories(${targ} PRIVATE ${INCLUDES})
  target_link_libraries(${targ} PRIVATE dnest4)
endforeach()

set(INCLUDES_EIGEN src/vendor/cpp-loadtxt/src src/vendor/DNest4/code src/vendor/eigen)

foreach(targ GPmodel Sampler)
  target_compile_features(${targ} PRIVATE cxx_std_17)
  target_include_directories(${targ} PRIVATE ${INCLUDES_EIGEN})
  target_link_libraries(${targ} PRIVATE dnest4)
endforeach()


# install directive for scikit-build-core
install(TARGETS distributions LIBRARY DESTINATION kima)
# 
install(TARGETS Data LIBRARY DESTINATION kima)
install(TARGETS ConditionalPrior LIBRARY DESTINATION kima)
install(TARGETS RVmodel LIBRARY DESTINATION kima)
install(TARGETS GPmodel LIBRARY DESTINATION kima)
install(TARGETS Sampler LIBRARY DESTINATION kima)
# 
install(TARGETS kepler LIBRARY DESTINATION kima)
