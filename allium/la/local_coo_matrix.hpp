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

#ifndef ALLIUM_LA_LOCAL_COO_MATRIX_HPP
#define ALLIUM_LA_LOCAL_COO_MATRIX_HPP

#include <allium/util/types.hpp>
#include <vector>

namespace allium {
  /**
   @brief Entry of a matrix given by row, coloumn and value.
   */
  template <typename N>
  class MatrixEntry {
    public:
      MatrixEntry() {}

      MatrixEntry(global_size_t row, global_size_t col, N value)
        : m_row(row), m_col(col), m_value(value) {}

      bool operator==(const MatrixEntry& rhs) const {
        return (m_row == rhs.m_row && m_col == rhs.m_col && m_value == rhs.m_value);
      }

      global_size_t row() { return m_row; }
      global_size_t col() { return m_col; }
      N value() { return m_value; }
    private:
      global_size_t m_row, m_col;
      N m_value;
  };

  /**
   @brief Local (non-distributed) matrix which stores the entries in coordinate format.
   */
  template <typename N>
  class LocalCooMatrix {
    public:
      LocalCooMatrix() {}
      template <typename T>
      explicit LocalCooMatrix(T&& entries)
        : m_entries(std::forward<T>(entries))
      {}

      void add(global_size_t row, global_size_t col, N value) {
        m_entries.push_back(MatrixEntry<N>(row, col, value));
      }

      size_t entry_count() { return m_entries.size(); }

      const std::vector<MatrixEntry<N>> entries() const& { return m_entries; }
      const std::vector<MatrixEntry<N>> entries() && { return std::move(m_entries); }
    private:
      std::vector<MatrixEntry<N>> m_entries;
  };


}

#endif
