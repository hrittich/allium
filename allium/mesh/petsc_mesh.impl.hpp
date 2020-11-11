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

#include "petsc_mesh.hpp"

#ifdef ALLIUM_USE_PETSC

namespace allium {

  PetscMesh<2>::PetscMesh(std::shared_ptr<PetscMeshSpec<2>> spec)
    : PetscAbstractVectorStorage<PetscScalar>(spec->vector_spec()),
      m_spec(spec)
  {
    using namespace petsc;
    PetscErrorCode ierr;

    ierr = DMCreateGlobalVector(spec->dm(), m_ptr.writable_ptr());
    chkerr(ierr);
  }

  PetscMesh<2>::PetscMesh(std::shared_ptr<PetscMeshSpec<2>> spec,
              PetscObjectPtr<Vec> ptr)
      : PetscAbstractVectorStorage<PetscScalar>(spec->vector_spec()),
        m_spec(spec)
    {
      m_ptr = ptr;
    }

  PetscMesh<2>* PetscMesh<2>::allocate_like() const& {
    return new PetscMesh(m_spec);
  }

  PetscMesh<2>* PetscMesh<2>::clone() const& {
    std::unique_ptr<PetscMesh> cloned(allocate_like());
    cloned->assign(*this);
    return cloned.release();
  }
}

#endif
