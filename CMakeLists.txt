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
find_package(PythonInterp)

# Find mpi4py
if(NOT MPI4PY_INCLUDE_DIRS)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import mpi4py; print(mpi4py.get_include())"
    RESULTS_VARIABLE MPI4PY_RESULT
    OUTPUT_VARIABLE MPI4PY_INCLUDE_DIRS)

  if(MPI4PY_RESULT EQUAL 0)
    string(STRIP ${MPI4PY_INCLUDE_DIRS} MPI4PY_INCLUDE_DIRS)
    message(STATUS "Found mpi4py include dir: ${MPI4PY_INCLUDE_DIRS}")
  else()
    set(MPI4PY_INCLUDE_DIRS NOTFOUND)
    message(SEND_ERROR "Could not find mpi4py include dir.")
  endif()
  set(MPI4PY_INCLUDE_DIRS ${MPI4PY_INCLUDE_DIRS} CACHE PATH "mpi4py include directory" FORCE)
endif()
include_directories(${MPI4PY_INCLUDE_DIRS})

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

  set(PROPOSED_DEFAULT_BACKEND "PETSC")
endif()

# Eigen3
include_directories(${CMAKE_BINARY_DIR}/contrib/eigen)

# pybind11
add_subdirectory(${CMAKE_BINARY_DIR}/contrib/pybind11
                 ${CMAKE_BINARY_DIR}/contrib/pybind11)


# Default linear algebra backend
if(NOT PROPOSED_DEFAULT_BACKEND)
  set(PROPOSED_DEFAULT_BACKEND "EIGEN")
endif()
set(CHIVE_DEFAULT_BACKEND ${PROPOSED_DEFAULT_BACKEND}
    CACHE STRING "Default linear algebra backend = \"EIGEN\"|\"PETSC\"")
if(CHIVE_DEFAULT_BACKEND STREQUAL "EIGEN")
  set(CHIVE_DEFAULT_BACKEND_EIGEN TRUE)
elseif(CHIVE_DEFAULT_BACKEND STREQUAL "PETSC")
  set(CHIVE_DEFAULT_BACKEND_PETSC TRUE)
else()
  message(SEND_ERROR "Invalid default backend.")
endif()

# General Config
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The CHIVE_LIBRARIES variable will be populated in the subdirectories.
set(CHIVE_LIBRARIES "")

add_compile_options(${COMPILE_OPTIONS})
include_directories("${CMAKE_SOURCE_DIR}")
add_link_options(${LINK_OPTIONS})

add_subdirectory(chive)
add_subdirectory(python)
add_subdirectory(demo)
add_subdirectory(tests)

