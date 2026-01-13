cmake_minimum_required(VERSION 3.31.6)

# Enable experimental import std support
if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.0.3)
  set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
elseif(CMAKE_VERSION VERSION_GREATER_EQUAL 4.0.0)
  set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "a9e1cf81-9932-4810-974b-6eccaf14e457")
elseif(CMAKE_VERSION VERSION_GREATER_EQUAL 3.31.6)
  set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "0e5b6991-d74f-4b3d-a41c-cf096e0b2508")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_MODULE_STD ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

project(lebesgue_quadrature VERSION 0.1.260113 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

find_package(lam_linearalgebra REQUIRED)

option(LAM_USE_TBB "Use Intel TBB for parallel execution" ON)
if(LAM_USE_TBB)
  find_package(TBB QUIET)
  if(TBB_FOUND)
    message(STATUS "Found TBB: enabling parallel execution backend")
  else()
    message(STATUS "TBB not found: parallel std::execution will fall back to sequential")
  endif()
endif()

add_library(lebesgue_quadrature)
add_library(lam::lebesgue ALIAS lebesgue_quadrature)

target_sources(lebesgue_quadrature
  PUBLIC
    FILE_SET CXX_MODULES
      TYPE CXX_MODULES
      BASE_DIRS ${PROJECT_SOURCE_DIR}/src
      FILES
        src/lebesgue.cppm
        src/lebesgue-detail.cppm
        src/lebesgue-basis.cppm
        src/lebesgue-gram.cppm
        src/lebesgue-quadrature.cppm
)
target_link_libraries(lebesgue_quadrature PUBLIC lam::linearalgebra)
if(TBB_FOUND)
  target_link_libraries(lebesgue_quadrature PUBLIC TBB::tbb)
  target_compile_definitions(lebesgue_quadrature PUBLIC LEB_HAS_TBB=1)
endif()
target_compile_features(lebesgue_quadrature PUBLIC cxx_std_23)
set_target_properties(lebesgue_quadrature PROPERTIES OUTPUT_NAME "lam_lebesgue")

# Installation Helpers
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Tests
enable_testing()
add_subdirectory(tests)

# Installation Rules
install(TARGETS lebesgue_quadrature
  EXPORT lebesgue_quadratureTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_LIBDIR}/c++/v1/lam/lebesgue
)

install(EXPORT lebesgue_quadratureTargets
  FILE lam_lebesgueTargets.cmake
  NAMESPACE lam::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lam_lebesgue
)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/lam_lebesgueConfig.cmake" [[
include(CMakeFindDependencyMacro)
find_dependency(lam_linearalgebra)
include("${CMAKE_CURRENT_LIST_DIR}/lam_lebesgueTargets.cmake")
]])

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/lam_lebesgueConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/lam_lebesgueConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/lam_lebesgueConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lam_lebesgue
)
