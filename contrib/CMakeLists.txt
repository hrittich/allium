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

set(GOOGLETEST_VERSION 1.10.0)
set(EIGEN_VERSION 3.3.7)
set(PYBIND11_VERSION 2.5.0)

function(UNPACK_SOURCE ARCHIVE DIRECTORY TARGET)

  if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${ARCHIVE})
      message(FATAL_ERROR
              "File ${ARCHIVE} is missing in ${CMAKE_CURRENT_SOURCE_DIR}")

    endif()

    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xfz
        ${CMAKE_CURRENT_SOURCE_DIR}/${ARCHIVE}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/${DIRECTORY}
                ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
  endif()

endfunction()

UNPACK_SOURCE(googletest-${GOOGLETEST_VERSION}.tar.gz
              googletest-release-${GOOGLETEST_VERSION}
              googletest)
UNPACK_SOURCE(eigen-${EIGEN_VERSION}.tar.gz
              eigen-${EIGEN_VERSION}
              eigen)
UNPACK_SOURCE(pybind11-${PYBIND11_VERSION}.tar.gz
              pybind11-${PYBIND11_VERSION}
              pybind11)


