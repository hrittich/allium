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

#include <allium/util/numeric.hpp>
#include "sparse_matrix.hpp"
#include "petsc_util.hpp"
#include "petsc_object_ptr.hpp"
#include "petsc_vector.hpp"
#include "linear_operator.hpp"

namespace allium {
  template <typename N>
  class PetscSparseMatrixStorage;

  template <>
  class PetscSparseMatrixStorage<PetscScalar>
      : public SparseMatrixStorage<PetscAbstractVectorStorage<PetscScalar>>
  {
    public:
      using Vector = PetscAbstractVectorStorage<PetscScalar>;
      using DefaultVector = PetscVectorStorage<PetscScalar>;
      using typename SparseMatrixStorage<Vector>::Number;
      using typename SparseMatrixStorage<Vector>::Real;

      PetscSparseMatrixStorage(VectorSpec rows, VectorSpec cols);

      void set_entries(LocalCooMatrix<Number> mat) override;
      LocalCooMatrix<Number> get_entries() override;

      void apply(PetscAbstractVectorStorage<PetscScalar>& result,
                 const PetscAbstractVectorStorage<PetscScalar>& arg) override;
    private:
      PetscObjectPtr<Mat> ptr;
  };

  template <typename N>
  class PetscSparseMatrixStorage final
    : public SparseMatrixStorage<PetscAbstractVectorStorage<N>>
  {
    public:
      using Number = N;
      using Real = real_part_t<Number>;
      using Vector = PetscAbstractVectorStorage<N>;
      using DefaultVector = PetscVectorStorage<N>;

      PetscSparseMatrixStorage(VectorSpec rows, VectorSpec cols)
        : SparseMatrixStorage<PetscAbstractVectorStorage<N>>(rows, cols),
          m_native(rows, cols)
      {}

      void set_entries(LocalCooMatrix<Number> mat) override {
        std::vector<MatrixEntry<N>> entries = std::move(mat).entries();

        std::vector<MatrixEntry<PetscScalar>> converted(entries.size());
        std::transform(entries.begin(), entries.end(),
                       converted.begin(),
                       [](MatrixEntry<N> e) -> MatrixEntry<PetscScalar> {
                         return MatrixEntry<PetscScalar>(
                                  e.row(),
                                  e.col(),
                                  e.value());
                       });

        m_native.set_entries(LocalCooMatrix<PetscScalar>(converted));
      }

      LocalCooMatrix<Number> get_entries() override {
        std::vector<MatrixEntry<PetscScalar>> entries = m_native.get_entries().entries();

        std::vector<MatrixEntry<N>> converted(entries.size());
        std::transform(entries.begin(), entries.end(),
                       converted.begin(),
                       [](MatrixEntry<PetscScalar> e) {
                         return MatrixEntry<N>(
                                  e.row(),
                                  e.col(),
                                  narrow_number<N, PetscScalar>()(e.value()));
                       });

        return LocalCooMatrix<N>(converted);
      }

      void apply(PetscAbstractVectorStorage<N>& result,
                 const PetscAbstractVectorStorage<N>& arg) override
      {
        m_native.apply(result.native_scalar(), arg.native_scalar());
      }

    private:
      PetscSparseMatrixStorage<PetscScalar> m_native;
  };
}

#endif
#endif
