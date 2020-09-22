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

#ifndef ALLIUM_LA_PETSC_VECTOR_HPP
#define ALLIUM_LA_PETSC_VECTOR_HPP

#include <allium/config.hpp>
namespace allium {}
#ifdef ALLIUM_USE_PETSC
#include "petsc_abstract_vector.hpp"

namespace allium {

  template <typename N>
  class PetscVectorStorage
    : public PetscAbstractVectorStorage<N>
  {
    public:
      using PetscAbstractVectorStorage<N>::PetscAbstractVectorStorage;
      using PetscAbstractVectorStorage<N>::m_ptr;

      PetscVectorStorage(VectorSpec spec)
        : PetscAbstractVectorStorage<N>(spec)
      {
        PetscErrorCode ierr;

        ierr = VecCreateMPI(spec.comm().handle(),
                            spec.local_size(),
                            spec.global_size(),
                            m_ptr.writable_ptr()); petsc::chkerr(ierr);
      }

      PetscVectorStorage* allocate_like() const& override {
        return new PetscVectorStorage(this->petsc_allocate_like());
      }

      PetscVectorStorage* clone() const& override {
        return new PetscVectorStorage(this->petsc_clone());
      }
  };

  template <typename N>
  using PetscVector = VectorBase<PetscVectorStorage<N>>;
}

#endif
#endif
