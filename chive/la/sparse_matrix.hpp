#ifndef CHIVE_LA_SPARSE_MATRIX_HPP
#define CHIVE_LA_SPARSE_MATRIX_HPP

#include "local_coo_matrix.hpp"
#include <memory>

namespace chive {
  template <typename NumberT>
  class SparseMatrixStorage {
    public:
      virtual void add(LocalCooMatrix<NumberT> mat) = 0;
  };

  template <typename NumberT,
            typename StorageT = SparseMatrixStorage<NumberT>>
  class SparseMatrix {
    public:
      SparseMatrix(const std::shared_ptr<StorageT>& ptr) : ptr(ptr) {}

      void add(LocalCooMatrix<NumberT> mat) {
        ptr->add(std::move(mat));
      }

    private:
      std::shared_ptr<StorageT> ptr;
  };

}

#endif
