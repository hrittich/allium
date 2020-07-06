#ifndef CHIVE_LA_SPARSE_MATRIX_HPP
#define CHIVE_LA_SPARSE_MATRIX_HPP

#include "local_coo_matrix.hpp"
#include "vector.hpp"
#include <memory>

namespace chive {
  template <typename N>
  class SparseMatrixStorage {
    public:
      SparseMatrixStorage(VectorSpec rows, VectorSpec cols)
        : row_spec(rows), col_spec(cols) {}

      virtual void set_entries(LocalCooMatrix<N> mat) = 0;
      virtual LocalCooMatrix<N> get_entries() = 0;

      VectorSpec get_row_spec() { return row_spec; }
      VectorSpec get_col_spec() { return col_spec; }
    private:
      VectorSpec row_spec;
      VectorSpec col_spec;
  };

  template <typename N,
            typename StorageT = SparseMatrixStorage<N>>
  class SparseMatrix {
    public:
      SparseMatrix(VectorSpec rows, VectorSpec cols)
        : ptr(std::make_shared<StorageT>(rows, cols)) {}
      SparseMatrix(const std::shared_ptr<StorageT>& ptr) : ptr(ptr) {}

      void set_entries(LocalCooMatrix<N> mat) {
        ptr->set_entries(std::move(mat));
      }

      LocalCooMatrix<N> get_entries() {
        return ptr->get_entries();
      }

    private:
      std::shared_ptr<StorageT> ptr;
  };

}

#endif
