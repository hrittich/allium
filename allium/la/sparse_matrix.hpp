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

#ifndef ALLIUM_LA_SPARSE_MATRIX_HPP
#define ALLIUM_LA_SPARSE_MATRIX_HPP

#include <allium/config.hpp>

#include "local_coo_matrix.hpp"
#include "linear_operator.hpp"
#include "vector_storage.hpp"
#include <memory>

namespace allium {

  /**
    Abstarct base type for all sparse-matrix classes.
  */
  template <typename V>
  class SparseMatrixStorage : public LinearOperator<V> {
    public:
      using Number = typename V::Number;
      using Real = real_part_t<Number>;

      SparseMatrixStorage(VectorSpec rows, VectorSpec cols)
        : m_row_spec(rows), m_col_spec(cols) {}

      virtual void set_entries(LocalCooMatrix<Number> mat) = 0;
      virtual LocalCooMatrix<Number> get_entries() = 0;

      VectorSpec row_spec() { return m_row_spec; }
      VectorSpec col_spec() { return m_col_spec; }
    private:
      VectorSpec m_row_spec;
      VectorSpec m_col_spec;
  };
}

#endif
