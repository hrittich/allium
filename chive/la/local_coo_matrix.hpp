#ifndef CHIVE_LA_LOCAL_COO_MATRIX_HPP
#define CHIVE_LA_LOCAL_COO_MATRIX_HPP

#include <chive/util/types.hpp>
#include <vector>

namespace chive {
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
