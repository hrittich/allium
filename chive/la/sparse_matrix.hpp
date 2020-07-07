#ifndef CHIVE_LA_SPARSE_MATRIX_HPP
#define CHIVE_LA_SPARSE_MATRIX_HPP

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
        : row_spec(rows), col_spec(cols) {}

      virtual void set_entries(LocalCooMatrix<N> mat) = 0;
      virtual LocalCooMatrix<N> get_entries() = 0;
      virtual Vector<N> vec_mult(const Vector<N>& v) = 0;

      VectorSpec get_row_spec() { return row_spec; }
      VectorSpec get_col_spec() { return col_spec; }
    private:
      VectorSpec row_spec;
      VectorSpec col_spec;
  };

  template <typename StorageT>
  class SparseMatrixBase {
    public:
      using Storage = StorageT;
      using Number = typename StorageT::Number;
      using Real = typename StorageT::Real;

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

    private:
      std::shared_ptr<StorageT> ptr;
  };

  template <typename N>
  using SparseMatrix = SparseMatrixBase<SparseMatrixStorage<N>>;
}

#endif
