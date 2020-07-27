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

#ifndef CHIVE_LA_SPARSE_MATRIX_HPP
#define CHIVE_LA_SPARSE_MATRIX_HPP

#include <chive/config.hpp>

#include "local_coo_matrix.hpp"
#include "vector.hpp"
#include <memory>

namespace chive {
  template <typename N>
  class SparseMatrixStorage {
    public:
      using Number = N;
      using Real = real_part_t<N>;

      SparseMatrixStorage(VectorSpec rows, VectorSpec cols)
        : m_row_spec(rows), m_col_spec(cols) {}

      virtual void set_entries(LocalCooMatrix<N> mat) = 0;
      virtual LocalCooMatrix<N> get_entries() = 0;
      virtual Vector<N> vec_mult(const Vector<N>& v) = 0;

      VectorSpec row_spec() { return m_row_spec; }
      VectorSpec col_spec() { return m_col_spec; }
    private:
      VectorSpec m_row_spec;
      VectorSpec m_col_spec;
  };

  template <typename StorageT>
  class SparseMatrixBase {
    public:
      using Storage = StorageT;
      using Number = typename StorageT::Number;
      using Real = typename StorageT::Real;

      template <typename S2>
      SparseMatrixBase(SparseMatrixBase<S2> other) : ptr(other.storage()) {}

      SparseMatrixBase(VectorSpec rows, VectorSpec cols)
        : ptr(std::make_shared<StorageT>(rows, cols)) {}
      SparseMatrixBase(const std::shared_ptr<StorageT>& ptr) : ptr(ptr) {}

      void set_entries(LocalCooMatrix<Number> mat) {
        ptr->set_entries(std::move(mat));
      }

      LocalCooMatrix<Number> get_entries() {
        return ptr->get_entries();
      }

      Vector<Number> operator* (const Vector<Number>& rhs) {
        return ptr->vec_mult(rhs);
      }

      std::shared_ptr<Storage> storage() { return ptr; }
      std::shared_ptr<const Storage> storage() const { return ptr; }
    private:
      std::shared_ptr<StorageT> ptr;
  };

  template <typename N>
    using SparseMatrix = SparseMatrixBase<SparseMatrixStorage<N>>;

  template <typename N>
    SparseMatrix<N> make_sparse_matrix(VectorSpec row_spec, VectorSpec col_spec);

  #define CHIVE_LA_SPARSE_MATRIX_DECL(T, N) \
    T SparseMatrix<N> make_sparse_matrix(VectorSpec row_spec, VectorSpec col_spec);
  CHIVE_EXTERN(CHIVE_LA_SPARSE_MATRIX_DECL)
}

#endif
