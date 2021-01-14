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

  template <typename N, int D>
  PetscMesh<N, D>::PetscMesh(std::shared_ptr<PetscMeshSpec<D>> spec)
    : PetscAbstractVectorStorage<N>(spec->vector_spec()),
      m_spec(spec)
  {
    using namespace petsc;
    PetscErrorCode ierr;

    PetscObjectPtr<Vec> ptr;
    ierr = DMCreateGlobalVector(spec->dm(), ptr.writable_ptr());
    this->native(ptr);

    chkerr(ierr);
  }

  template <typename N, int D>
  PetscMesh<N, D>::PetscMesh(std::shared_ptr<PetscMeshSpec<D>> spec,
                             PetscObjectPtr<Vec> ptr)
      : PetscAbstractVectorStorage<N>(spec->vector_spec()),
        m_spec(spec)
    {
      this->native(ptr);
    }

  template <typename N, int D>
  PetscMesh<N, D>* PetscMesh<N, D>::allocate_like() const& {
    return new PetscMesh(m_spec);
  }

  template <typename N, int D>
  PetscMesh<N, D>* PetscMesh<N, D>::clone() const& {
    std::unique_ptr<PetscMesh> cloned(allocate_like());
    cloned->assign(*this);
    return cloned.release();
  }
}

#endif
