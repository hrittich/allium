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
#include <allium/la/vector.hpp>
#include <allium/la/petsc_vector.hpp>
#include <allium/la/eigen_vector.hpp>

using namespace allium;

// Types to test
typedef
  testing::Types<
    EigenVector<double>
    , EigenVector<std::complex<double>>
    #ifdef ALLIUM_USE_PETSC
      , PetscVector<double>
      , PetscVector<std::complex<double>>
    #endif
    >
  VectorTypes;

template <typename T>
struct VectorTest : public testing::Test {
  VectorTest()
    : comm(Comm::world()),
      spec_1d(comm, 1, 1),
      spec_2d(comm, 2, 2)
  {}

  Comm comm;
  VectorSpec spec_1d;
  VectorSpec spec_2d;
};

TYPED_TEST_CASE(VectorTest, VectorTypes);

TYPED_TEST(VectorTest, ReadWrite)
{
  TypeParam v(this->spec_1d);

  {
    auto v_loc = local_slice(v);
    v_loc[0] = 1.0;
  }
  {
    auto v_loc = local_slice(v);
    EXPECT_EQ(v_loc[0], 1.0);
  }
}

TYPED_TEST(VectorTest, BoundCheck)
{
  TypeParam v(this->spec_1d);
  auto v_loc = local_slice(v);
  #ifdef ALLIUM_BOUND_CHECKS
    EXPECT_ANY_THROW(v_loc[1]);
  #endif
}

TYPED_TEST(VectorTest, CopyConstruct) {
  auto v = TypeParam(this->spec_1d);

  local_slice(v) = { 1.0 };

  TypeParam w(v);

  {
    auto loc = local_slice(w);
    EXPECT_EQ(loc[0], 1.0);
  }
}

TYPED_TEST(VectorTest, MoveConstruct) {
  auto v = TypeParam(this->spec_1d);

  local_slice(v) ={ 1.0 };

  TypeParam w(std::move(v));

  {
    auto loc = local_slice(w);
    EXPECT_EQ(loc[0], 1.0);
  }
}

TYPED_TEST(VectorTest, Assign) {
  auto v = TypeParam(this->spec_1d);
  auto w = TypeParam(this->spec_1d);

  local_slice(v) = { 1.0 };

  w = v;

  {
    auto loc = local_slice(w);
    EXPECT_EQ(loc[0], 1.0);
  }
}

TYPED_TEST(VectorTest, LocalSliceAssign) {
  auto v = TypeParam(this->spec_2d);
  auto w = TypeParam(this->spec_2d);

  local_slice(v) = { 1.0, 2.0 };
  local_slice(w) = local_slice(v);
  {
    auto loc = local_slice(w);
    EXPECT_EQ(loc[0], 1.0);
    EXPECT_EQ(loc[1], 2.0);
  }
}

TYPED_TEST(VectorTest, Move) {
  auto v = TypeParam(this->spec_1d);
  auto w = TypeParam(this->spec_1d);

  local_slice(v) = { 1.0 };

  w = std::move(v);

  {
    auto loc = local_slice(w);
    EXPECT_EQ(loc[0], 1.0);
  }
}

TYPED_TEST(VectorTest, InitializerList) {
  VectorSpec vspec(this->comm, 3, 3);
  auto v = TypeParam(vspec);

  {
    auto v_loc = local_slice(v);
    v_loc = { 1.0, 3.0, 5.0 };
  }

  {
    auto v_loc = local_slice(v);
    EXPECT_EQ(v_loc[0], 1.0);
    EXPECT_EQ(v_loc[1], 3.0);
    EXPECT_EQ(v_loc[2], 5.0);
  }

  {
    auto v_loc = local_slice(v);
    EXPECT_ANY_THROW(v_loc = { 1.0 });
  }
}

TYPED_TEST(VectorTest, InplaceAdd) {
  auto v = TypeParam(this->spec_1d);
  auto w = TypeParam(this->spec_1d);

  local_slice(v) = { 2.0 };
  local_slice(w) = { 3.0 };

  v += w;

  { auto v_loc = local_slice(v);
    EXPECT_EQ(v_loc[0], 5.0);
  }
  { auto w_loc = local_slice(w);
    EXPECT_EQ(w_loc[0], 3.0);
  }
}

TYPED_TEST(VectorTest, Add) {
  auto v = TypeParam(this->spec_1d);
  auto w = TypeParam(this->spec_1d);

  local_slice(v) = { 2.0 };
  local_slice(w) = { 3.0 };

  auto u = v + w;

  { auto v_loc = local_slice(v);
    EXPECT_EQ(v_loc[0], 2.0);
  }
  { auto w_loc = local_slice(w);
    EXPECT_EQ(w_loc[0], 3.0);
  }
  { auto u_loc = local_slice(u);
    EXPECT_EQ(u_loc[0], 5.0);
  }
}

TYPED_TEST(VectorTest, InplaceSub) {
  auto v = TypeParam(this->spec_1d);
  auto w = TypeParam(this->spec_1d);

  local_slice(v) = { 3.0 };
  local_slice(w) = { 2.0 };

  v -= w;

  { auto v_loc = local_slice(v);
    EXPECT_EQ(v_loc[0], 1.0);
  }
  { auto w_loc = local_slice(w);
    EXPECT_EQ(w_loc[0], 2.0);
  }
}

TYPED_TEST(VectorTest, Sub) {
  auto v = TypeParam(this->spec_1d);
  auto w = TypeParam(this->spec_1d);

  local_slice(v) = { 3.0 };
  local_slice(w) = { 2.0 };

  auto u = v - w;

  { auto v_loc = local_slice(v);
    EXPECT_EQ(v_loc[0], 3.0);
  }
  { auto w_loc = local_slice(w);
    EXPECT_EQ(w_loc[0], 2.0);
  }
  { auto u_loc = local_slice(u);
    EXPECT_EQ(u_loc[0], 1.0);
  }
}

TYPED_TEST(VectorTest, InplaceScale) {
  auto v = TypeParam(this->spec_1d);

  local_slice(v) = { 2.0 };

  v *= 3;

  {
    auto loc = local_slice(v);
    EXPECT_EQ(loc[0], 6.0);
  }
}

TYPED_TEST(VectorTest, Scale) {
  auto v = TypeParam(this->spec_1d);

  local_slice(v) = { 2.0 };

  auto w = 3 * v;

  { auto loc = local_slice(v);
    EXPECT_EQ(loc[0], 2.0);
  }
  { auto loc = local_slice(w);
    EXPECT_EQ(loc[0], 6.0);
  }
}

TYPED_TEST(VectorTest, InplaceDiv) {
  auto v = TypeParam(this->spec_1d);

  local_slice(v) = { 6.0 };

  v /= 3;

  {
    auto loc = local_slice(v);
    EXPECT_EQ(loc[0], 2.0);
  }
}

TYPED_TEST(VectorTest, Div) {
  auto v = TypeParam(this->spec_1d);

  local_slice(v) = { 6.0 };

  auto w = v / 3;

  { auto loc = local_slice(v);
    EXPECT_EQ(loc[0], 6.0);
  }
  { auto loc = local_slice(w);
    EXPECT_EQ(loc[0], 2.0);
  }
}

TYPED_TEST(VectorTest, Norm) {
  VectorSpec spec(this->comm, 4, 4);
  auto v = TypeParam(spec);
  local_slice(v) = { 1.0, 1.0, 1.0, 1.0 };

  EXPECT_EQ(v.l2_norm(), 2.0);
}

TYPED_TEST(VectorTest, Dot) {
  auto v = TypeParam(this->spec_2d);
  auto w = TypeParam(this->spec_2d);

  // todo: test with complex numbers

  local_slice(v) = { 2, 4 };
  local_slice(w) = { 3, 1 };

  EXPECT_EQ(v.dot(w), 10.0);
}

