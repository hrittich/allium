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

#ifndef CHIVE_LA_PETSC_SPARSE_MATRIX_HPP
#define CHIVE_LA_PETSC_SPARSE_MATRIX_HPP

#include <chive/config.hpp>

#ifdef CHIVE_USE_PETSC

#include "sparse_matrix.hpp"
#include "petsc_util.hpp"
#include "petsc_object_ptr.hpp"
#include "petsc_vector.hpp"

namespace chive {
  class PetscSparseMatrixStorage final
    : public SparseMatrixStorage<PetscScalar> {
    public:
      using NativeVector = PetscVector;

      PetscSparseMatrixStorage(VectorSpec rows, VectorSpec cols);

      void set_entries(LocalCooMatrix<Number> mat);
      LocalCooMatrix<Number> get_entries();

      Vector<PetscScalar> vec_mult(const Vector<Number>& v) override;

    private:
      PetscObjectPtr<Mat> ptr;
  };

  using PetscSparseMatrix = SparseMatrixBase<PetscSparseMatrixStorage>;
}

#endif
#endif
