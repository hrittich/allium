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

#include "init.hpp"

#include <chive/config.hpp>
#include <chive/la/petsc_util.hpp>

namespace chive {
  #ifdef CHIVE_USE_PETSC
  using petsc::chkerr;
  #endif

  Init::Init(int& argc, char** &argv)
    : mpi(argc, argv)
  {
    #ifdef CHIVE_USE_PETSC
      PetscErrorCode ierr;
      ierr = PetscInitialize(&argc, &argv, nullptr, nullptr); chkerr(ierr);
    #endif
  }

  Init::~Init() {
    #ifdef CHIVE_USE_PETSC
      PetscErrorCode ierr;
      ierr = PetscFinalize(); chkerr(ierr);
    #endif
  }
}

