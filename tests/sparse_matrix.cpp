#include <gtest/gtest.h>
#include <chive/config.hpp>
#include <chive/la/eigen_sparse_matrix.hpp>
#include <chive/la/petsc_sparse_matrix.hpp>

using namespace chive;

typedef
  testing::Types<
    EigenSparseMatrix<double>
    , EigenSparseMatrix<std::complex<double>>
    #ifdef CHIVE_USE_PETSC
    , PetscSparseMatrix
    #endif
    > MatrixStorageTypes;

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

  { auto loc = local_slice(w);
    ASSERT_EQ(loc[0], 30.0);
  }
}

TYPED_TEST(SparseMatrixTest, MatVecMult2)
{
  using Number = typename TypeParam::Number;
  using NativeVector = typename TypeParam::Storage::NativeVector;

  VectorSpec spec(MpiComm::world(), 2, 2);
  TypeParam mat(spec, spec);

  LocalCooMatrix<Number> lmat;
  lmat.add(0, 0, 1);
  lmat.add(0, 1, 5);
  lmat.add(1, 0, 2);
  lmat.add(1, 1, 3);
  mat.set_entries(lmat);

  NativeVector v(spec);
  { auto loc = local_slice(v);
    loc[0] = 3;
    loc[1] = -1;
  }

  auto w = mat * v;

  { auto loc = local_slice(w);
    ASSERT_EQ(loc[0], -2.0);
    ASSERT_EQ(loc[1], 3.0);
  }
}

