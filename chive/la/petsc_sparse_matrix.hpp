#ifndef CHIVE_LA_PETSC_SPARSE_MATRIX_HPP
#define CHIVE_LA_PETSC_SPARSE_MATRIX_HPP

#include "sparse_matrix.hpp"
#include "petsc_util.hpp"
#include "petsc_object_ptr.hpp"
#include "petsc_vector.hpp"

namespace chive {
  class PetscSparseMatrixStorage final
    : public SparseMatrixStorage<PetscScalar> {
    public:
      using NativeVector = PetscVector;

      PetscSparseMatrixStorage(VectorSpec rows, VectorSpec cols);

      void set_entries(LocalCooMatrix<Number> mat);
      LocalCooMatrix<Number> get_entries();

      Vector<PetscScalar> vec_mult(const Vector<Number>& v) override {
        throw std::logic_error("Not implemented");
      }

    private:
      PetscObjectPtr<Mat> ptr;
  };

  using PetscSparseMatrix = SparseMatrixBase<PetscSparseMatrixStorage>;
}

#endif
