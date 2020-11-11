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

#ifndef ALLIUM_LA_PETSC_SPARSE_MATRIX_HPP
#define ALLIUM_LA_PETSC_SPARSE_MATRIX_HPP

#include <allium/config.hpp>

#ifdef ALLIUM_USE_PETSC

#include "sparse_matrix.hpp"
#include "petsc_util.hpp"
#include "petsc_object_ptr.hpp"
#include "petsc_vector.hpp"
#include "linear_operator.hpp"

namespace allium {
  class PetscSparseMatrixStorage final
      : public SparseMatrixStorage<PetscScalar>,
        public LinearOperator<PetscVectorStorage<PetscScalar>>
  {
    public:
      using NativeVector = PetscVector<PetscScalar>;
      using typename SparseMatrixStorage<PetscScalar>::Number;
      using typename SparseMatrixStorage<PetscScalar>::Real;

      PetscSparseMatrixStorage(VectorSpec rows, VectorSpec cols);

      void set_entries(LocalCooMatrix<Number> mat) override;
      LocalCooMatrix<Number> get_entries() override;

      allium::Vector<PetscScalar> vec_mult(const allium::Vector<Number>& v) override;

      void apply(PetscVectorStorage<PetscScalar>& result,
                 const PetscVectorStorage<PetscScalar>& arg) override;
    private:
      PetscObjectPtr<Mat> ptr;
  };

  using PetscSparseMatrix = SparseMatrixBase<PetscSparseMatrixStorage>;
}

#endif
#endif
