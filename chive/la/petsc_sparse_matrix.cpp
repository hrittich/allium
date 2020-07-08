#include "petsc_sparse_matrix.hpp"

namespace chive {
  using namespace petsc;

  class PetscMatRow final {
    public:
      PetscMatRow(PetscObjectPtr<Mat> mat, PetscInt row) : m_mat(mat), m_row(row)
      {
        PetscErrorCode ierr;
        ierr = MatGetRow(m_mat, m_row, &m_ncols, &m_cols, &m_vals); chkerr(ierr);
      }

      ~PetscMatRow() {
        PetscErrorCode ierr;
        MatRestoreRow(m_mat, m_row, &m_ncols, &m_cols, &m_vals);
      }

      global_size_t ncols() {
        return m_ncols;
      }

      global_size_t col(global_size_t i) {
        #ifdef CHIVE_BOUND_CHECKS
          if (i >= m_ncols)
            throw std::runtime_error("Out of bound access.");
        #endif
        return m_cols[i];
      }

      PetscScalar value(global_size_t i) {
        #ifdef CHIVE_BOUND_CHECKS
          if (i >= m_ncols)
            throw std::runtime_error("Out of bound access");
        #endif
        return m_vals[i];
      }

    private:
      PetscObjectPtr<Mat> m_mat;
      PetscInt m_row;
      PetscInt m_ncols;
      const PetscInt* m_cols;
      const PetscScalar* m_vals;
  };

  PetscSparseMatrixStorage::PetscSparseMatrixStorage(VectorSpec rows, VectorSpec cols)
    : SparseMatrixStorage<PetscScalar>(rows, cols)
  {
  }

  void PetscSparseMatrixStorage::set_entries(LocalCooMatrix<PetscScalar> mat)
  {
    PetscErrorCode ierr;

    if (row_spec().get_comm() != col_spec().get_comm()) {
      throw std::runtime_error(
        "PETSc requires the row and column communicator to be the same");
    }

    // ToDo: Create interface to d_nz and o_nz
    ierr = MatCreateAIJ(row_spec().get_comm().get_handle(), // comm,
                        row_spec().get_local_size(), // local rows
                        col_spec().get_local_size(), // local cols
                        row_spec().get_global_size(), // global rows
                        col_spec().get_global_size(),  // global cols
                        10, // d_nz
                        NULL, // d_nnz
                        10, // o_nz,
                        NULL, // o_nnz,
                        ptr.writable_ptr()); chkerr(ierr);

    auto entries = std::move(mat).get_entries();

    // ToDo: This is probably slow and can be optimized by using MatSetValues
    for (auto& e : entries) {
      ierr = MatSetValue(ptr, e.get_row(), e.get_col(), e.get_value(), ADD_VALUES);
      chkerr(ierr);
    }

    ierr = MatAssemblyBegin(ptr, MAT_FINAL_ASSEMBLY); chkerr(ierr);
    ierr = MatAssemblyEnd(ptr, MAT_FINAL_ASSEMBLY); chkerr(ierr);
  }

  LocalCooMatrix<PetscScalar> PetscSparseMatrixStorage::get_entries()
  {
    PetscErrorCode ierr;
    PetscInt start, end;

    ierr = MatGetOwnershipRange(ptr, &start, &end); chkerr(ierr);

    LocalCooMatrix<PetscScalar> lmat;

    for (global_size_t i_row=start; i_row < end; ++i_row) {
      PetscMatRow row(ptr, i_row);
      for (global_size_t i_col_entry=0; i_col_entry < row.ncols(); ++i_col_entry) {
        lmat.add(i_row, row.col(i_col_entry), row.value(i_col_entry));
      }
    }

    return lmat;
  }

  Vector<PetscScalar>
    PetscSparseMatrixStorage::vec_mult(const Vector<Number>& v)
  {
    PetscErrorCode ierr;

    auto v_store = std::dynamic_pointer_cast<const PetscVectorStorage>(v.storage());
    if (!v_store)
      throw std::runtime_error("Not implemented");

    NativeVector w(row_spec());

    ierr = MatMult(ptr, v_store->native(), w.storage()->native()); chkerr(ierr);

    return w;
  }


}

