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
  template <typename NumberT>
  class MatrixEntry {
    public:
      MatrixEntry(global_size_t row, global_size_t col, NumberT value)
        : row(row), col(col), value(value) {}

      bool operator==(const MatrixEntry& rhs) const {
        return (row == rhs.row && col == rhs.col && value == rhs.value);
      }

      global_size_t get_row() { return row; }
      global_size_t get_col() { return col; }
      NumberT get_value() { return value; }
    private:
      global_size_t row, col;
      NumberT value;
  };

  template <typename NumberT>
  class LocalCooMatrix {
    public:
      void add(global_size_t row, global_size_t col, NumberT value) {
        entries.push_back(MatrixEntry<NumberT>(row, col, value));
      }

      const std::vector<MatrixEntry<NumberT>> get_entries() const& { return entries; }
      const std::vector<MatrixEntry<NumberT>> get_entries() && { return std::move(entries); }
    private:
      std::vector<MatrixEntry<NumberT>> entries;
  };

}

#endif
