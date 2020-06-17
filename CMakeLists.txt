cmake_minimum_required(VERSION 3.13)

add_subdirectory(contrib)

# Options

set(CHIVE_GLOBAL_SIZE_T "size_t"
    CACHE STRING "Type of the size of global quantities, e.g., vectors")
option(CHIVE_BOUND_CHECKS "Perform bound checks on array accesses" OFF)

# Find Python
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  # Default to Python 3
  # Later calls to the deprecated module FindPythonInterp will use
  # this Python executable
  set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE} CACHE PATH
      "Path to the Python executable.")
endif()

# Configure GoogleTest

set(GTEST_SRC_DIR "${CMAKE_BINARY_DIR}/contrib/googletest"
    CACHE PATH "Source dir of googletest.")
add_subdirectory(${GTEST_SRC_DIR} "${CMAKE_BINARY_DIR}/contrib/googletest")
include_directories("${GTEST_SRC_DIR}/googletest/include")
set(GTEST_LIBRARIES "gtest")

# Configure MPI

find_package(MPI REQUIRED)

list(APPEND COMPILE_OPTIONS ${MPI_CXX_COMPILE_OPTIONS})
include_directories("${MPI_CXX_INCLUDE_DIRS}")
list(APPEND LINK_OPTIONS ${MPI_CXX_LINK_FLAGS})
list(APPEND LIBRARIES ${MPI_CXX_LIBRARIES})

# PkgConfig
find_package(PkgConfig)

# Find PETSc
if(PKG_CONFIG_FOUND)
  pkg_check_modules(PETSC PETSc)

  list(APPEND COMPILE_OPTIONS ${PETSC_CFLAGS})
  include_directories(${PETSC_INCLUDE_DIRS})
  list(APPEND LINK_OPTIONS ${PETSC_LDFLAGS})
  list(APPEND LIBRARIES ${PETSC_LINK_LIBRARIES})
  set(CHIVE_USE_PETSC ON)
endif()

# Eigen3
include_directories(${CMAKE_BINARY_DIR}/contrib/eigen)

# pybind11
add_subdirectory(${CMAKE_BINARY_DIR}/contrib/pybind11)


# General Config

# The CHIVE_LIBRARIES variable will be populated in the subdirectories.
set(CHIVE_LIBRARIES "")

add_compile_options(${COMPILE_OPTIONS})
include_directories("${CMAKE_SOURCE_DIR}")
add_link_options(${LINK_OPTIONS})

add_subdirectory(chive)
add_subdirectory(python)
add_subdirectory(demo)
add_subdirectory(tests)

