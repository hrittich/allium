// Copyright 2021 Hannah Rittich
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

#include <cstdlib>
#include <Eigen/Core>
#include "local_vector.hpp"
#include <iostream>

namespace allium {

  template <typename N>
  class LocalMatrix {
    public:
      /**
        Create a matrix with storage for the specified number of rows and
        columns.
      */
      LocalMatrix(size_t rows, size_t cols);

      /**
        Create a matrix from a nested list of initializers.
      */
      explicit LocalMatrix(std::initializer_list<std::initializer_list<N>> entries);

      /**
        Access matrix elements.
      */
      N& operator() (size_t i_row, size_t i_col) {
        return m_storage(i_row, i_col);
      }

      size_t rows() { return m_storage.rows(); }
      size_t cols() { return m_storage.cols(); }

      /**
        Apply the matrix to a vector, i.e., multiply the matrix and the
        vector.
      */
      LocalVector<N> apply(const LocalVector<N>& x);

      void set_row(size_t i_row, const LocalVector<N>& v);
      void set_col(size_t i_col, const LocalVector<N>& v);

      LocalVector<N> get_row(size_t i_row) const;
      LocalVector<N> get_col(size_t i_col) const;
    private:
      Eigen::Matrix<N, Eigen::Dynamic, Eigen::Dynamic> m_storage;
  };

  template <typename N>
  LocalMatrix<N>::LocalMatrix(size_t rows, size_t cols)
    : m_storage(rows, cols)
  {}

  template <typename N>
  LocalMatrix<N>::LocalMatrix(std::initializer_list<std::initializer_list<N>> entries)
  {
    const size_t new_rows = entries.size();
    const size_t new_cols = (new_rows > 0) ? (*entries.begin()).size() : 0;

    m_storage.resize(new_rows, new_cols);

    int i = 0;
    for (auto row : entries) {
      if (row.size() != new_cols)
        throw std::runtime_error("Rows are not of equal length.");

      int j = 0;
      for (auto entry : row) {
        m_storage(i, j) = entry;
        ++j;
      }

      ++i;
    }
  }

  template <typename N>
  LocalVector<N> LocalMatrix<N>::apply(const LocalVector<N>& x)
  {
    return LocalVector<N>(m_storage * x.native());
  }

  template <typename N>
  void LocalMatrix<N>::set_row(size_t i_row, const LocalVector<N>& v) {
    m_storage.row(i_row) = v.native();
  }

  template <typename N>
  void LocalMatrix<N>::set_col(size_t i_col, const LocalVector<N>& v) {
    m_storage.col(i_col) = v.native();
  }

  template <typename N>
  LocalVector<N> LocalMatrix<N>::get_row(size_t i_row) const {
    return LocalVector<N>(m_storage.row(i_row));
  }

  template <typename N>
  LocalVector<N> LocalMatrix<N>::get_col(size_t i_col) const {
    return LocalVector<N>(m_storage.col(i_col));
  }
}

