#include <gtest/gtest.h>
#include <chive/la/eigen_sparse_matrix.hpp>

using namespace chive;

TEST(SparseMatrix, Create) {
  VectorSpec spec(MpiComm::world(), 1, 1);
  EigenSparseMatrix<double> mat(spec, spec);

  LocalCooMatrix<double> lmat;
  lmat.add(0, 0, 2);

  mat.set_entries(lmat);

  ASSERT_EQ(lmat.get_entries(), mat.get_entries().get_entries());
}

