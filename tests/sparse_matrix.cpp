#include <gtest/gtest.h>
#include <chive/la/eigen_sparse_matrix.hpp>
#include <chive/la/petsc_sparse_matrix.hpp>

using namespace chive;

typedef
  testing::Types<
    EigenSparseMatrix<double>,
    EigenSparseMatrix<std::complex<double>>,
    PetscSparseMatrix>
  MatrixStorageTypes;

template <typename S>
class SparseMatrixTest : public testing::Test {
};

TYPED_TEST_CASE(SparseMatrixTest, MatrixStorageTypes);

TYPED_TEST(SparseMatrixTest, Create) {
  VectorSpec spec(MpiComm::world(), 1, 1);
  TypeParam mat(spec, spec);
}

TYPED_TEST(SparseMatrixTest, SetAndReadEntries)
{
  using Number = typename TypeParam::Number;

  VectorSpec spec(MpiComm::world(), 1, 1);
  TypeParam mat(spec, spec);

  LocalCooMatrix<Number> lmat;
  lmat.add(0, 0, 2);

  mat.set_entries(lmat);

  ASSERT_EQ(lmat.get_entries(), mat.get_entries().get_entries());
}

TYPED_TEST(SparseMatrixTest, MatVecMult)
{
  using Number = typename TypeParam::Number;
  using NativeVector = typename TypeParam::Storage::NativeVector;

  VectorSpec spec(MpiComm::world(), 1, 1);
  TypeParam mat(spec, spec);

  LocalCooMatrix<Number> lmat;
  lmat.add(0, 0, 2);
  mat.set_entries(lmat);

  NativeVector v(spec);
  { auto loc = local_slice(v);
    loc[0] = 15;
  }

  auto w = mat * v;
}

