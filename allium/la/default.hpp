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

#ifndef ALLIUM_LA_DEFAULT_HPP
#define ALLIUM_LA_DEFAULT_HPP

#include <allium/config.hpp>

#if defined(ALLIUM_DEFAULT_BACKEND_EIGEN)

  #include "eigen_vector.hpp"
  #include "eigen_sparse_matrix.hpp"

  namespace allium {
    template <typename N>
    using DefaultVector = EigenVectorStorage<N>;

    template <typename N>
    using DefaultSparseMatrix = EigenSparseMatrixStorage<N>;
  }

#elif defined(ALLIUM_DEFAULT_BACKEND_PETSC)

  #include "petsc_vector.hpp"
  #include "petsc_sparse_matrix.hpp"

  namespace allium {
    template <typename N>
    using DefaultVector = PetscVectorStorage<N>;

    template <typename N>
    using DefaultSparseMatrix =
      typename std::enable_if<std::is_same<N, PetscScalar>::value,
                              PetscSparseMatrixStorage>::type;
  }

#else
  #error "No default linear algebra backend"
#endif

#endif
