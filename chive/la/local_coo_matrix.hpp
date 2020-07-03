#ifndef CHIVE_LA_LOCAL_COO_MATRIX_HPP
#define CHIVE_LA_LOCAL_COO_MATRIX_HPP

#include <chive/util/types.hpp>
#include <vector>

namespace chive {
  template <typename NumberT>
  class LocalCooMatrix {
    public:
      struct Entry {
        global_size_t row;
        global_size_t col;
        NumberT value;
      };

      void add(global_size_t row, global_size_t col, NumberT value) {
        entries.push_back(Entry{ row, col, value });
      }

      const std::vector<Entry> get_entries() const& { return entries; }
      const std::vector<Entry> get_entries() && { return std::move(entries); }
    private:
      std::vector<Entry> entries;
  };

}

#endif
