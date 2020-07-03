#include <gtest/gtest.h>
#include <chive/la/eigen_sparse_matrix.hpp>

using namespace chive;

TEST(SparseMatrix, Create) {
  SparseMatrix<double> mat(std::make_shared<EigenSparseMatrixStorage<double>>());

  LocalCooMatrix<double> lmat;
  lmat.add(0,0, 2);
  lmat.add(0,1, -1);

  mat.add(lmat);
}

