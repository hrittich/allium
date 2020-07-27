// Copyright 2020 Hannah Rittich
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ALLIUM_LA_PETSC_UTIL_HPP
#define ALLIUM_LA_PETSC_UTIL_HPP

#include <allium/config.hpp>

#ifdef ALLIUM_USE_PETSC
#include <petscsys.h>
#endif

namespace allium {
  namespace petsc {

    #ifdef ALLIUM_USE_PETSC
    inline void chkerr(PetscErrorCode ierr) {
      if (PetscUnlikely(ierr)) {
        throw std::runtime_error("PETSc error");
      }
    }
    #endif

  }
}

#endif