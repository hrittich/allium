#include <gtest/gtest.h>

#include <chive/la/petsc_vector.hpp>

using namespace chive;

TEST(PetscVector, Create) {
  MpiComm comm = MpiComm::world();

  VectorSpec vspec(comm, 10, 10);

  PetscVector v(vspec);
}

TEST(PetscVector, Fill) {
  auto comm = MpiComm::world();

  VectorSpec vspec(comm, 1, 1);
  auto v = std::make_shared<PetscVector>(vspec);

  {
    auto v_loc = v->local_slice();
    (*v_loc)[0] = 1.0;
  }

  {
    auto v_loc = v->local_slice();
    EXPECT_EQ((*v_loc)[0], 1.0);
  }

  {
    auto v_loc = v->local_slice();
    #ifdef CHIVE_BOUND_CHECKS
      try {
        (*v_loc)[1] = 1;
        FAIL() << "Expected out-of-bound exception";
      } catch (...) {

      }
    #endif
  }
}

TEST(PetscVector, Add) {
  auto comm = MpiComm::world();
  VectorSpec vspec(comm, 2, 2);
  auto v = std::make_shared<PetscVector>(vspec);
  auto w = std::make_shared<PetscVector>(vspec);

  {
    auto v_loc = v->local_slice();
    (*v_loc)[0] = 2;
    (*v_loc)[1] = 3;
  }

}


TEST(PetscVector, Foo) {
  auto comm = MpiComm::world();
  VectorSpec vspec(comm, 2, 2);
  auto v = std::make_shared<PetscVector>(vspec);

  Vector_<std::unique_ptr<PetscVector>> w(std::make_unique<PetscVector>(vspec));

  Vector_<std::shared_ptr<PetscVector>> u(std::move(w));

  Vector_<std::shared_ptr<PetscVector>> x(u);

  Vector_<std::shared_ptr<Vector<std::complex<double>>>> y(x);

}

