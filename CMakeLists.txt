#  Copyright 2020 Hannah Rittich
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

cmake_minimum_required(VERSION 3.13)
project(allium
  VERSION 0.1.0
  DESCRIPTION "Parallel Computation Library"
  LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

add_subdirectory(contrib)

# Options
set(ALLIUM_GLOBAL_SIZE_T "size_t"
    CACHE STRING "Type of the size of global quantities, e.g., vectors")
option(ALLIUM_BOUND_CHECKS "Perform bound checks on array accesses" OFF)

# Find Python
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  # Default to Python 3
  # Later calls to the deprecated module FindPythonInterp will use
  # this Python executable
  set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE} CACHE PATH
      "Path to the Python executable.")

  set(ALLIUM_USE_PYTHON ON CACHE BOOL "Build the Python extension")
endif()
find_package(PythonInterp)

# Find mpi4py
if(ALLIUM_USE_PYTHON AND (NOT MPI4PY_INCLUDE_DIRS))
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import mpi4py; print(mpi4py.get_include())"
    RESULTS_VARIABLE MPI4PY_RESULT
    OUTPUT_VARIABLE MPI4PY_INCLUDE_DIRS)

  if(MPI4PY_RESULT EQUAL 0)
    string(STRIP ${MPI4PY_INCLUDE_DIRS} MPI4PY_INCLUDE_DIRS)
    message(STATUS "Found mpi4py include dir: ${MPI4PY_INCLUDE_DIRS}")
  else()
    set(MPI4PY_INCLUDE_DIRS NOTFOUND)
    message(WARNING "Could not find mpi4py include dir.")
  endif()
  set(MPI4PY_INCLUDE_DIRS ${MPI4PY_INCLUDE_DIRS} CACHE PATH "mpi4py include directory" FORCE)
endif()

if(MPI4PY_INCLUDE_DIRS)
  set(ALLIUM_USE_MPI4PY ON CACHE BOOL "Provide MPI4PY conversion macros.")
  include_directories(${MPI4PY_INCLUDE_DIRS})
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

  if(PETSC_FOUND)
    list(APPEND COMPILE_OPTIONS ${PETSC_CFLAGS})
    include_directories(${PETSC_INCLUDE_DIRS})
    list(APPEND LINK_OPTIONS ${PETSC_LDFLAGS})
    list(APPEND LIBRARIES ${PETSC_LINK_LIBRARIES})

    set(ALLIUM_USE_PETSC ON CACHE BOOL "Use PETSc")
  endif()
endif()

# Eigen3
include_directories(${CMAKE_BINARY_DIR}/contrib/eigen)

# pybind11
if(ALLIUM_USE_PYTHON)
  add_subdirectory(${CMAKE_BINARY_DIR}/contrib/pybind11
                   ${CMAKE_BINARY_DIR}/contrib/pybind11)
endif()

# Default linear algebra backend
set(ALLIUM_DEFAULT_BACKEND ""
    CACHE STRING "Default linear algebra backend = \"EIGEN\"|\"PETSC\"")
if(NOT ALLIUM_DEFAULT_BACKEND)
  set(ALLIUM_DEFAULT_BACKEND "EIGEN")
  if(ALLIUM_USE_PETSC)
    set(ALLIUM_DEFAULT_BACKEND "PETSC")
  endif()
endif()

if(ALLIUM_DEFAULT_BACKEND STREQUAL "EIGEN")
  set(ALLIUM_DEFAULT_BACKEND_EIGEN TRUE)
elseif(ALLIUM_DEFAULT_BACKEND STREQUAL "PETSC")
  set(ALLIUM_DEFAULT_BACKEND_PETSC TRUE)
else()
  message(SEND_ERROR "Invalid default backend.")
endif()

# General Config
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include_directories(${CMAKE_BINARY_DIR}) # Include for automatically generated files

# The ALLIUM_LIBRARIES variable will be populated in the subdirectories.
set(ALLIUM_LIBRARIES "")

add_compile_options(${COMPILE_OPTIONS})
include_directories("${CMAKE_SOURCE_DIR}")
add_link_options(${LINK_OPTIONS})

add_subdirectory(allium)
add_subdirectory(python)
add_subdirectory(tests)
add_subdirectory(demo)

# Print Configuration Summary

function(show_status DESCRIPTION VARIABLE)
  set(VALUE ${${VARIABLE}})
  if("${VALUE}" STREQUAL "")
    set(VALUE OFF)
  endif()
  message(STATUS "${DESCRIPTION} : ${VALUE}")
endfunction()

message(STATUS "--------------------------------------------")
message(STATUS "                Summary")
message(STATUS "--------------------------------------------")
show_status("  Default Backend    " ALLIUM_DEFAULT_BACKEND)
show_status("  Global Index Type  " ALLIUM_GLOBAL_SIZE_T)
show_status("  Bound Checks       " ALLIUM_BOUND_CHECKS)
show_status("  Use PETSc          " ALLIUM_USE_PETSC)
show_status("  Python Module      " ALLIUM_USE_PYTHON)
if(ALLIUM_USE_PYTHON)
  show_status("    Python Version   " Python3_VERSION)
  show_status("    Use mpi4py       " ALLIUM_USE_MPI4PY)
endif()
message(STATUS "--------------------------------------------")

