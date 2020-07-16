#include <gtest/gtest.h>

#include <chive/la/cg.hpp>
#include <chive/la/petsc_vector.hpp>
#include <chive/la/petsc_sparse_matrix.hpp>

using namespace chive;

TEST(CG, solve1)
{
  using Number = PetscVector::Number;

  VectorSpec spec(MpiComm::world(), 1, 1);
  PetscVector v(spec);

  LocalCooMatrix<Number> coo;
  coo.add(0, 0, 5);

  PetscSparseMatrix mat(spec, spec);
  mat.set_entries(coo);

  { auto loc = local_slice(v);
    loc[0] = 1;
  }

  auto w = cg(mat, v);

  { auto loc = local_slice(w);
    EXPECT_EQ(loc[0], 0.2);
  }
}

TEST(CG, solve4)
{
  using Number = PetscVector::Number;

  VectorSpec spec(MpiComm::world(), 4, 4);
  PetscVector v(spec);

  LocalCooMatrix<Number> coo;
  coo.add(0, 0,  2);
  coo.add(0, 1, -1);

  coo.add(1, 0, -1);
  coo.add(1, 1,  2);
  coo.add(1, 2, -1);

  coo.add(2, 1, -1);
  coo.add(2, 2,  2);
  coo.add(2, 3, -1);

  coo.add(3, 2, -1);
  coo.add(3, 3,  2);

  PetscSparseMatrix mat(spec, spec);
  mat.set_entries(coo);

  { auto loc = local_slice(v);
    loc[0] = 1.0;
    loc[1] = 0.0;
    loc[2] = 0.0;
    loc[3] = 1.0;
  }

  auto w = cg(mat, v);
  { auto loc = local_slice(w);
    EXPECT_EQ(loc[0], 1.0);
    EXPECT_EQ(loc[1], 1.0);
    EXPECT_EQ(loc[2], 1.0);
    EXPECT_EQ(loc[3], 1.0);
  }
}



