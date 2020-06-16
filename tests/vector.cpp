#include <gtest/gtest.h>

#include <chive/la/vector.hpp>
#include <chive/la/petsc_vector.hpp>
#include <chive/la/eigen_vector.hpp>

using namespace chive;

// Types to test
typedef
  testing::Types<
    EigenVectorStorage<double>,
    EigenVectorStorage<std::complex<double>>,
    PetscVectorStorage>
  VectorStorageTypes;

template <typename T>
class VectorStorageTest : public ::testing::Test {
  public:
    using Number = typename T::Number;
};

TYPED_TEST_CASE(VectorStorageTest, VectorStorageTypes);

TYPED_TEST(VectorStorageTest, Create) {
  MpiComm comm = MpiComm::world();

  VectorSpec vspec(comm, 1, 1);

  TypeParam v(vspec);
}

TYPED_TEST(VectorStorageTest, Fill) {
  using Number = typename TestFixture::Number;

  auto comm = MpiComm::world();

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
    try {
      v_loc[1] = 1;
      FAIL() << "Expected out-of-bound exception";
    } catch (...) {}
  }
  #endif
}

TYPED_TEST(VectorStorageTest, Add) {
  auto comm = MpiComm::world();
  VectorSpec vspec(comm, 1, 1);
  auto v = std::make_shared<TypeParam>(vspec);
  auto w = std::make_shared<TypeParam>(vspec);

  {
    auto v_loc = local_slice(v);
    auto w_loc = local_slice(w);

    v_loc[0] = 2;
    w_loc[0] = 3;
  }

  v->add(*w);

  {
    auto v_loc = local_slice(v);

    EXPECT_EQ(v_loc[0], 5.0);
  }
}

TYPED_TEST(VectorStorageTest, Scale) {
  auto comm = MpiComm::world();
  VectorSpec vspec(comm, 1, 1);
  auto v = std::make_shared<TypeParam>(vspec);

  {
    auto loc = local_slice(v);
    loc[0] = 2;
  }

  v->scale(3);

  {
    auto loc = local_slice(v);
    EXPECT_EQ(loc[0], 6.0);
  }
}

TYPED_TEST(VectorStorageTest, Norm) {
  auto comm = MpiComm::world();
  VectorSpec vspec(comm, 4, 4);
  auto v = std::make_shared<TypeParam>(vspec);

  {
    auto loc = local_slice(v);
    loc[0] = 1;
    loc[1] = 1;
    loc[2] = 1;
    loc[3] = 1;
  }

  EXPECT_EQ(v->l2_norm(), 2.0);
}



