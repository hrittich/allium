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

#ifdef ALLIUM_USE_PETSC

#include "petsc_object_ptr.hpp"
#include <petscvec.h>
#include "vector.hpp"

namespace allium {
  class PetscVectorStorage final
      : public VectorStorageBase<PetscVectorStorage, PetscScalar>
  {
    public:
      PetscVectorStorage(VectorSpec spec);

      void add(const VectorStorage& rhs) override;
      void scale(const Number& factor) override;
      Number dot(const VectorStorage<Number>& rhs) override;
      Real l2_norm() const override;

      PetscObjectPtr<Vec> native() const { return ptr; }
    protected:
      Number* aquire_data_ptr() override;
      void release_data_ptr(Number* data) override;
    private:
      PetscObjectPtr<Vec> ptr;
  };

  using PetscVector = VectorBase<PetscVectorStorage>;
}

#endif
#endif
