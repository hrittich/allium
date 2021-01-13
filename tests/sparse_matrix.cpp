// Copyright 2020 Hannah Rittich
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <allium/config.hpp>
#include <allium/la/eigen_sparse_matrix.hpp>
#include <allium/la/petsc_sparse_matrix.hpp>

using namespace allium;

typedef
  testing::Types<
    EigenSparseMatrixStorage<double>
    , EigenSparseMatrixStorage<std::complex<double>>
    #ifdef ALLIUM_USE_PETSC
      , PetscSparseMatrixStorage<double>
      #ifdef ALLIUM_PETSC_HAS_COMPLEX
        , PetscSparseMatrixStorage<std::complex<double>>
      #endif
    #endif
    > MatrixStorageTypes;

template <typename S>
class SparseMatrixTest : public testing::Test {
};

TYPED_TEST_CASE(SparseMatrixTest, MatrixStorageTypes);

TYPED_TEST(SparseMatrixTest, Create) {
  VectorSpec spec(Comm::world(), 1, 1);
  TypeParam mat(spec, spec);
}

TYPED_TEST(SparseMatrixTest, SetAndReadEntries)
{
  using Number = typename TypeParam::Number;

  VectorSpec spec(Comm::world(), 1, 1);
  TypeParam mat(spec, spec);

  LocalCooMatrix<Number> lmat;
  lmat.add(0, 0, 2);

  mat.set_entries(lmat);

  ASSERT_EQ(lmat.entries(), mat.get_entries().entries());
}

TYPED_TEST(SparseMatrixTest, MatVecMult)
{
  using Number = typename TypeParam::Number;
  using Vector = typename TypeParam::DefaultVector;

  VectorSpec spec(Comm::world(), 1, 1);
  TypeParam mat(spec, spec);

  LocalCooMatrix<Number> lmat;
  lmat.add(0, 0, 2);
  mat.set_entries(lmat);

  Vector v(spec);
  local_slice(v) = { 15 };

  Vector w(spec);
  mat.apply(w, v);

  { auto loc = local_slice(w);
    ASSERT_EQ(loc[0], 30.0);
  }
}

TYPED_TEST(SparseMatrixTest, MatVecMult2)
{
  using Number = typename TypeParam::Number;
  using Vector = typename TypeParam::DefaultVector;

  VectorSpec spec(Comm::world(), 2, 2);
  TypeParam mat(spec, spec);

  LocalCooMatrix<Number> lmat;
  lmat.add(0, 0, 1);
  lmat.add(0, 1, 5);
  lmat.add(1, 0, 2);
  lmat.add(1, 1, 3);
  mat.set_entries(lmat);

  Vector v(spec);
  local_slice(v) = { 3, -1 };

  Vector w(spec);
  mat.apply(w, v);

  { auto loc = local_slice(w);
    ASSERT_EQ(loc[0], -2.0);
    ASSERT_EQ(loc[1], 3.0);
  }
}

