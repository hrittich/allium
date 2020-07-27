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

#include <chive/config.hpp>
#include <chive/la/vector.hpp>
#include <chive/la/petsc_vector.hpp>
#include <chive/la/eigen_vector.hpp>

using namespace chive;

// Types to test
typedef
  testing::Types<
    EigenVectorStorage<double>
    , EigenVectorStorage<std::complex<double>>
    #ifdef CHIVE_USE_PETSC
      , PetscVectorStorage
    #endif
    >
  VectorStorageTypes;

template <typename T>
class VectorStorageTest : public testing::Test {
  public:
    using Number = typename T::Number;
};

TYPED_TEST_CASE(VectorStorageTest, VectorStorageTypes);

TYPED_TEST(VectorStorageTest, Create) {
  using Number = typename TestFixture::Number;

  Comm comm = Comm::world();

  VectorSpec vspec(comm, 1, 1);

  TypeParam v(vspec);

  VectorBase<TypeParam> v2(vspec);
}

TYPED_TEST(VectorStorageTest, Fill) {
  using Number = typename TestFixture::Number;

  auto comm = Comm::world();

  VectorSpec vspec(comm, 1, 1);
  auto v = std::make_shared<TypeParam>(vspec);

  {
    auto v_loc = VectorSlice<Number>(v);
    v_loc[0] = 1.0;
  }

  {
    auto v_loc = VectorSlice<Number>(v);
    EXPECT_EQ(v_loc[0], 1.0);
  }

  #ifdef CHIVE_BOUND_CHECKS
  {
    auto v_loc = VectorSlice<Number>(v);
    EXPECT_ANY_THROW(v_loc[1] = 1);
  }
  #endif
}

TYPED_TEST(VectorStorageTest, Initializer) {
  using Number = typename TestFixture::Number;

  auto comm = Comm::world();

  VectorSpec vspec(comm, 3, 3);
  auto v = std::make_shared<TypeParam>(vspec);

  {
    auto v_loc = VectorSlice<Number>(v);
    v_loc = { 1.0, 3.0, 5.0 };
  }

  {
    auto v_loc = VectorSlice<Number>(v);
    EXPECT_EQ(v_loc[0], 1.0);
    EXPECT_EQ(v_loc[1], 3.0);
    EXPECT_EQ(v_loc[2], 5.0);
  }

  #ifdef CHIVE_BOUND_CHECKS
  {
    auto v_loc = VectorSlice<Number>(v);
    EXPECT_ANY_THROW(v_loc = { 1.0 });
  }
  #endif
}

TYPED_TEST(VectorStorageTest, Add) {
  using Number = typename TypeParam::Number;

  auto comm = Comm::world();
  VectorSpec vspec(comm, 1, 1);
  auto v = std::make_shared<TypeParam>(vspec);
  auto w = std::make_shared<TypeParam>(vspec);

  {
    auto v_loc = VectorSlice<Number>(v);
    auto w_loc = VectorSlice<Number>(w);

    v_loc[0] = 2;
    w_loc[0] = 3;
  }

  v->add(*w);

  {
    auto v_loc = VectorSlice<Number>(v);

    EXPECT_EQ(v_loc[0], 5.0);
  }
}

TYPED_TEST(VectorStorageTest, Scale) {
  using Number = typename TypeParam::Number;

  auto comm = Comm::world();
  VectorSpec vspec(comm, 1, 1);
  auto v = std::make_shared<TypeParam>(vspec);

  {
    auto loc = VectorSlice<Number>(v);
    loc[0] = 2;
  }

  v->scale(3);

  {
    auto loc = VectorSlice<Number>(v);
    EXPECT_EQ(loc[0], 6.0);
  }
}

TYPED_TEST(VectorStorageTest, Norm) {
  using Number = typename TypeParam::Number;

  auto comm = Comm::world();
  VectorSpec vspec(comm, 4, 4);
  auto v = std::make_shared<TypeParam>(vspec);

  {
    auto loc = VectorSlice<Number>(v);
    loc[0] = 1;
    loc[1] = 1;
    loc[2] = 1;
    loc[3] = 1;
  }

  EXPECT_EQ(v->l2_norm(), 2.0);
}

TYPED_TEST(VectorStorageTest, Assign) {
  auto comm = Comm::world();
  VectorSpec vspec(comm, 1, 1);
  auto v = VectorBase<TypeParam>(vspec);
  auto w = VectorBase<TypeParam>(vspec);

  {
    auto loc = local_slice(v);
    loc[0] = 1;
  }

  w.assign(v);

  {
    auto loc = local_slice(w);
    EXPECT_EQ(loc[0], 1.0);
  }
}

TYPED_TEST(VectorStorageTest, CastToGeneric) {
  using Number = typename TypeParam::Number;

  auto comm = Comm::world();
  VectorSpec vspec(comm, 1, 1);
  auto v = VectorBase<TypeParam>(vspec);
  auto v_gen = Vector<Number>(v);
}

TYPED_TEST(VectorStorageTest, SetZero) {
  auto comm = Comm::world();
  VectorSpec vspec(comm, 1, 1);
  auto v = VectorBase<TypeParam>(vspec);

  v.set_zero();
  {
    auto loc = local_slice(v);
    EXPECT_EQ(loc[0], 0.0);
  }
}

TYPED_TEST(VectorStorageTest, Dot) {
  auto comm = Comm::world();
  VectorSpec vspec(comm, 1, 1);
  auto v = VectorBase<TypeParam>(vspec);
  auto w = VectorBase<TypeParam>(vspec);

  // todo: test with complex numbers

  { auto loc = local_slice(v);
    loc[0] = 2; }
  { auto loc = local_slice(w);
    loc[0] = 3; }

  EXPECT_EQ(v.dot(w), 6.0);
}


