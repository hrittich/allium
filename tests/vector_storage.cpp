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

#include <allium/config.hpp>
#include <allium/la/vector_storage.hpp>
#include <allium/la/petsc_vector.hpp>
#include <allium/la/eigen_vector.hpp>
#include <allium/la/cuda_vector.hpp>
#include <allium/mesh/cuda_mesh.hpp>

#include <gtest/gtest.h>

using namespace allium;

typedef
  testing::Types<
    EigenVectorStorage<double>
    , EigenVectorStorage<std::complex<double>>
    #ifdef ALLIUM_USE_PETSC
      , PetscVectorStorage<double>
      #ifdef ALLIUM_PETSC_HAS_COMPLEX
      , PetscVectorStorage<std::complex<double>>
      #endif
    #endif
    #ifdef ALLIUM_USE_CUDA
      , CudaVector<double>
      , CudaVector<float>
      , CudaMesh<double, 2>
      , CudaMesh<float, 2>
    #endif
    >
  VectorStorageTypes;

template <typename T>
struct VectorStorageTest : public testing::Test {
  using Number = typename T::Number;

  VectorStorageTest ()
    : comm(Comm::world()),
      spec_1d(comm, 1, 1),
      spec_2d(comm, 2, 2)
  {}

  Comm comm;
  VectorSpec spec_1d;
  VectorSpec spec_2d;
};

TYPED_TEST_CASE(VectorStorageTest, VectorStorageTypes);

TYPED_TEST(VectorStorageTest, Create) {
  TypeParam v(this->spec_1d);
}

TYPED_TEST(VectorStorageTest, ReadWrite) {
  auto comm = Comm::world();

  VectorSpec vspec(comm, 1, 1);
  TypeParam v(vspec);

  {
    auto v_loc = LocalSlice<TypeParam*>(&v);
    v_loc[0] = 1.0;
  }

  {
    auto v_loc = LocalSlice<TypeParam*>(&v);
    EXPECT_EQ(v_loc[0], 1.0);
  }

  #ifdef ALLIUM_BOUND_CHECKS
  {
    auto v_loc = LocalSlice<TypeParam*>(&v);
    EXPECT_ANY_THROW(v_loc[1] = 1);
  }
  #endif
}

TYPED_TEST(VectorStorageTest, Fill) {
  auto comm = Comm::world();

  VectorSpec vspec(comm, 2, 2);
  TypeParam v(vspec);

  v.fill(42.0);

  {
    auto v_loc = LocalSlice<TypeParam*>(&v);
    EXPECT_EQ(v_loc[0], 42.0);
    EXPECT_EQ(v_loc[1], 42.0);
  }
}

TYPED_TEST(VectorStorageTest, InitializerLists) {
  auto comm = Comm::world();

  VectorSpec vspec(comm, 3, 3);
  TypeParam v(vspec);

  {
    auto v_loc = LocalSlice<TypeParam*>(&v);
    v_loc = { 1.0, 3.0, 5.0 };
  }

  {
    auto v_loc = LocalSlice<TypeParam*>(&v);
    EXPECT_EQ(v_loc[0], 1.0);
    EXPECT_EQ(v_loc[1], 3.0);
    EXPECT_EQ(v_loc[2], 5.0);
  }

  #ifdef ALLIUM_BOUND_CHECKS
  {
    auto v_loc = LocalSlice<TypeParam*>(&v);
    EXPECT_ANY_THROW(v_loc = { 1.0 });
  }
  #endif
}

TYPED_TEST(VectorStorageTest, Assign) {
  auto comm = Comm::world();
  VectorSpec vspec(comm, 1, 1);
  TypeParam v(vspec);
  TypeParam w(vspec);

  local_slice(v) = { 1.0 };

  w.assign(v);

  {
    auto loc = local_slice(w);
    EXPECT_EQ(loc[0], 1.0);
  }

  // make sure it hase been a true, deep copy
  local_slice(w) = { 2.0 };

  EXPECT_EQ(local_slice(v)[0], 1.0);
  EXPECT_EQ(local_slice(w)[0], 2.0);
}

TYPED_TEST(VectorStorageTest, AssignBase) {
  using Number = typename TypeParam::Number;

  auto comm = Comm::world();
  VectorSpec vspec(comm, 1, 1);
  TypeParam v_(vspec);
  VectorStorage<Number>& v(v_);
  TypeParam w(vspec);

  local_slice(v) = { 1.0 };

  w.assign(v);

  {
    auto loc = local_slice(w);
    EXPECT_EQ(loc[0], 1.0);
  }

  // make sure it hase been a true, deep copy
  local_slice(w) = { 2.0 };

  EXPECT_EQ(local_slice(v)[0], 1.0);
  EXPECT_EQ(local_slice(w)[0], 2.0);
}

TYPED_TEST(VectorStorageTest, LocalSliceAssign) {
  auto comm = Comm::world();
  VectorSpec spec(comm, 2, 2);
  TypeParam v(spec);
  TypeParam w(spec);

  local_slice(v) = { 1.0, 2.0 };
  local_slice(w) = local_slice(v);
  {
    auto loc = local_slice(w);
    EXPECT_EQ(loc[0], 1.0);
    EXPECT_EQ(loc[1], 2.0);
  }
} 

TYPED_TEST(VectorStorageTest, Add) {
  using Number = typename TypeParam::Number;

  auto comm = Comm::world();
  VectorSpec vspec(comm, 1, 1);
  TypeParam v(vspec);
  TypeParam w(vspec);

  {
    auto v_loc = LocalSlice<TypeParam*>(&v);
    auto w_loc = LocalSlice<TypeParam*>(&w);

    v_loc[0] = 2;
    w_loc[0] = 3;
  }

  v += w;
  static_cast<VectorStorage<Number>&>(v) += w;
  v += static_cast<VectorStorage<Number>&>(w);
  static_cast<VectorStorage<Number>&>(v)
    += static_cast<VectorStorage<Number>&>(w);

  {
    auto v_loc = LocalSlice<TypeParam*>(&v);
    EXPECT_EQ(v_loc[0], 14.0);
  }
  EXPECT_EQ(local_slice(w)[0], 3.0);
}

TYPED_TEST(VectorStorageTest, Scale) {
  auto comm = Comm::world();
  VectorSpec vspec(comm, 1, 1);
  TypeParam v(vspec);

  {
    auto loc = LocalSlice<TypeParam*>(&v);
    loc[0] = 2;
  }

  v *= 3;

  {
    auto loc = LocalSlice<TypeParam*>(&v);
    EXPECT_EQ(loc[0], 6.0);
  }
}

TYPED_TEST(VectorStorageTest, AddScaled) {
  auto comm = Comm::world();
  VectorSpec vspec(comm, 1, 1);
  TypeParam v(vspec), w(vspec);

  {
    auto loc = LocalSlice<TypeParam*>(&v);
    loc[0] = 2;
  }
  {
    auto loc = LocalSlice<TypeParam*>(&w);
    loc[0] = 3;
  }

  v.add_scaled(2, w);

  {
    auto loc = LocalSlice<TypeParam*>(&v);
    EXPECT_EQ(loc[0], 8.0);
  }
}

TYPED_TEST(VectorStorageTest, Norm) {
  auto comm = Comm::world();
  VectorSpec vspec(comm, 4, 4);
  TypeParam v(vspec);

  {
    auto loc = LocalSlice<TypeParam*>(&v);
    loc = { 1.0, 1.0, 1.0, 1.0 };
  }

  EXPECT_EQ(v.l2_norm(), 2.0);
}

TYPED_TEST(VectorStorageTest, LargeNorm) {
  using Number = typename TypeParam::Number;
  auto comm = Comm::world();
  const int m = 2000;

  VectorSpec vspec(comm, m, m);
  TypeParam v(vspec);

  {
    auto lv = local_slice(v);
    for (size_t i = 0; i < m; ++i) {
      lv[i] = sqrt(i+1);
    }
  }

  EXPECT_LE(
    std::abs(v.l2_norm() - sqrt(Number(m * (m+1) / 2))),
    1e-4);
}

TYPED_TEST(VectorStorageTest, SetZero) {
  TypeParam v(this->spec_1d);

  set_zero(v);
  {
    auto loc = local_slice(v);
    EXPECT_EQ(loc[0], 0.0);
  }
}

TYPED_TEST(VectorStorageTest, Dot) {
  using Number = typename TypeParam::Number;
  auto comm = Comm::world();
  VectorSpec vspec(comm, 1, 1);
  TypeParam v(vspec);
  TypeParam w(vspec);

  // @todo: test with complex numbers

  { auto loc = local_slice(v);
    loc[0] = 2; }
  { auto loc = local_slice(w);
    loc[0] = 3; }

  EXPECT_EQ(v.dot(w), 6.0);
  EXPECT_EQ(static_cast<VectorStorage<Number>&>(v).dot(w), 6.0);
  EXPECT_EQ(v.dot(static_cast<VectorStorage<Number>&>(w)), 6.0);
  EXPECT_EQ(static_cast<VectorStorage<Number>&>(v)
            .dot(static_cast<VectorStorage<Number>&>(w)), 6.0);
}

TYPED_TEST(VectorStorageTest, LargeDot) {
  using Number = typename TypeParam::Number;
  auto comm = Comm::world();
  const int m = 2000;

  VectorSpec vspec(comm, m, m);
  TypeParam v(vspec);
  TypeParam w(vspec);

  {
    auto lv = local_slice(v);
    for (size_t i = 0; i < m; ++i) {
      lv[i] = i+1;
    }
  }

  fill(w, 1.0);

  EXPECT_EQ(v.dot(w), Number(m * (m+1) / 2));
}

TYPED_TEST(VectorStorageTest, AllocateLike) {
  auto comm = Comm::world();
  VectorSpec vspec(comm, 1, 1);
  TypeParam v(vspec);
  local_slice(v) = { 2.0 };

  auto w = allocate_like(v);

  EXPECT_EQ(w->spec().local_size(), 1);
  EXPECT_EQ(w->spec().global_size(), 1);

  local_slice(*w) = { 3.0 };

  EXPECT_EQ(local_slice(v)[0], 2.0);
  EXPECT_EQ(local_slice(*w)[0], 3.0);
}

TYPED_TEST(VectorStorageTest, Clone) {
  auto comm = Comm::world();
  VectorSpec vspec(comm, 1, 1);
  TypeParam v(vspec);

  local_slice(v) = { 2.0 };

  auto w = clone(v);

  EXPECT_EQ(w->spec().local_size(), 1);
  EXPECT_EQ(w->spec().global_size(), 1);
  EXPECT_EQ(local_slice(*w)[0], 2.0);

  local_slice(*w) = { 3.0 };
  EXPECT_EQ(local_slice(v)[0], 2.0);
  EXPECT_EQ(local_slice(*w)[0], 3.0);
}

