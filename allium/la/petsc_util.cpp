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

#include <allium/config.hpp>
#include "petsc_util.hpp"

namespace allium { namespace petsc {

#ifdef ALLIUM_USE_PETSC
PetscInt vec_local_size(PetscObjectPtr<Vec> vec) {
  PetscErrorCode ierr;

  PetscInt size;
  ierr = VecGetLocalSize(vec, &size); chkerr(ierr);
  return size;
}

PetscInt vec_global_size(PetscObjectPtr<Vec> vec) {
  PetscErrorCode ierr;

  PetscInt size;
  ierr = VecGetLocalSize(vec, &size); chkerr(ierr);
  return size;
}

Comm object_comm_(PetscObject o) {
  PetscErrorCode ierr;

  MPI_Comm comm;
  ierr = PetscObjectGetComm(o, &comm); chkerr(ierr);

  return Comm(comm);
}
#endif

}}

