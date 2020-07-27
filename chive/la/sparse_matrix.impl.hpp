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

#include "sparse_matrix.hpp"

#include "petsc_sparse_matrix.hpp"
#include "eigen_sparse_matrix.hpp"

namespace chive {

  #if defined(CHIVE_DEFAULT_BACKEND_EIGEN)
    template <typename N>
      SparseMatrix<N> make_sparse_matrix(VectorSpec row_spec, VectorSpec col_spec) {
        return EigenSparseMatrix<N>(row_spec, col_spec);
      }
  #elif defined(CHIVE_DEFAULT_BACKEND_PETSC)
    template <typename N>
      SparseMatrix<N> make_sparse_matrix_(VectorSpec row_spec, VectorSpec col_spec) {
        throw std::logic_error("Not implemented");
      }

    template <>
      SparseMatrix<PetscScalar> make_sparse_matrix_(VectorSpec row_spec, VectorSpec col_spec) {
        return PetscSparseMatrix(row_spec, col_spec);
      }

    template <typename N>
      SparseMatrix<N> make_sparse_matrix(VectorSpec row_spec, VectorSpec col_spec) {
        return make_sparse_matrix_<N>(row_spec, col_spec);
      }
  #else
    #error "No default linear algebra backend"
  #endif

}

