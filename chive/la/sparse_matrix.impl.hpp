#include "sparse_matrix.hpp"

#include "petsc_sparse_matrix.hpp"
#include "eigen_sparse_matrix.hpp"

namespace chive {

  #if defined(CHIVE_DEFAULT_BACKEND_EIGEN)
    template <typename N>
      SparseMatrix<N> make_sparse_matrix(VectorSpec row_spec, VectorSpec col_spec) {
        return EigenSparseMatrix<N>(row_spec, col_spec);
      }
  #elif defined(CHIVE_DEFAULT_BACKEND_PETSC)
    template <typename N>
      SparseMatrix<N> make_sparse_matrix_(VectorSpec row_spec, VectorSpec col_spec) {
        throw std::logic_error("Not implemented");
      }

    template <>
      SparseMatrix<PetscScalar> make_sparse_matrix_(VectorSpec row_spec, VectorSpec col_spec) {
        return PetscSparseMatrix(row_spec, col_spec);
      }

    template <typename N>
      SparseMatrix<N> make_sparse_matrix(VectorSpec row_spec, VectorSpec col_spec) {
        return make_sparse_matrix_<N>(row_spec, col_spec);
      }
  #else
    #error "No default linear algebra backend"
  #endif

}

