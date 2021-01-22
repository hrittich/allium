// Copyright 2021 Hannah Rittich
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

#include <allium/la/local_matrix.hpp>

#include <complex>
#include <gtest/gtest.h>

using TestTypes = ::testing::Types<float,
                                   double,
                                   std::complex<float>,
                                   std::complex<double>>;
template <typename T>
struct LocalMatrixTest : public testing::Test {};
TYPED_TEST_SUITE(LocalMatrixTest, TestTypes);

using namespace allium;

TYPED_TEST(LocalMatrixTest, ReadWrite1x1)
{
  using Number = TypeParam;
  LocalMatrix<Number> m(1, 1);

  m(0,0) = Number(42);
  EXPECT_EQ(m(0,0), Number(42));
}

TYPED_TEST(LocalMatrixTest, InitializeEmpty)
{
  using Number = TypeParam;
  LocalMatrix<Number> m({});

  EXPECT_EQ(m.rows(), 0);
  EXPECT_EQ(m.cols(), 0);
}

TYPED_TEST(LocalMatrixTest, Initialize2x2)
{
  using Number = TypeParam;
  LocalMatrix<Number> m({{1,2}, {3, 4}});

  ASSERT_EQ(m.rows(), 2);
  ASSERT_EQ(m.cols(), 2);
  EXPECT_EQ(m(0,0), Number(1));
  EXPECT_EQ(m(0,1), Number(2));
  EXPECT_EQ(m(1,0), Number(3));
  EXPECT_EQ(m(1,1), Number(4));
}

TYPED_TEST(LocalMatrixTest, InvalidInitialize)
{
  using Number = TypeParam;

  EXPECT_ANY_THROW(LocalMatrix<Number> m({{1,2}, {3, 4, 5}}));
}

TYPED_TEST(LocalMatrixTest, SetRow)
{
  using Number = TypeParam;
  LocalMatrix<Number> m(2, 2);
  LocalVector<Number> v(2);

  v[0] = Number(5);
  v[1] = Number(6);

  m.set_row(0, v);

  EXPECT_EQ(m(0,0), Number(5));
  EXPECT_EQ(m(0,1), Number(6));
}

TYPED_TEST(LocalMatrixTest, SetCol)
{
  using Number = TypeParam;
  LocalMatrix<Number> m(2, 2);
  LocalVector<Number> v(2);

  v[0] = Number(5);
  v[1] = Number(6);

  m.set_row(0, v);

  EXPECT_EQ(m(0,0), Number(5));
  EXPECT_EQ(m(0,1), Number(6));
}

TYPED_TEST(LocalMatrixTest, GetRowCol)
{
  using Number = TypeParam;
  LocalMatrix<Number> m({{1,2}, {3,4}});

  auto v = m.get_row(1);
  auto w = m.get_col(1);

  EXPECT_EQ(v[0], Number(3));
  EXPECT_EQ(v[1], Number(4));

  EXPECT_EQ(w[0], Number(2));
  EXPECT_EQ(w[1], Number(4));
}

TYPED_TEST(LocalMatrixTest, ApplyToVector)
{
  using Number = TypeParam;
  LocalMatrix<Number> m({{1,2},{3,4}});
  LocalVector<Number> v(2);

  v[0] = Number(5);
  v[1] = Number(6);

  auto w = m.apply(v);
  EXPECT_EQ(w.nrows(), 2);
  EXPECT_EQ(w[0], Number(17));
  EXPECT_EQ(w[1], Number(39));
}

