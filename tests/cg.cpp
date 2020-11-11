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

#include <allium/la/cg.hpp>
#include <allium/la/default.hpp>
#include <allium/la/linear_operator.hpp>

// @todo remove
#include <allium/la/vector.hpp>

using namespace allium;

TEST(CG, solve1)
{
  using Number = std::complex<double>;

  VectorSpec spec(Comm::world(), 1, 1);
  auto v = Vector<Number>(spec);

  LocalCooMatrix<Number> coo;
  coo.add(0, 0, 5);

  auto mat = make_sparse_matrix<Number>(spec, spec);
  mat.set_entries(coo);

  local_slice(v) = { 1.0 };

  auto w = cg(mat, v);

  { auto loc = local_slice(w);
    EXPECT_EQ(loc[0], 0.2);
  }
}

TEST(CG, solve4)
{
  using Number = std::complex<double>;

  VectorSpec spec(Comm::world(), 4, 4);
  auto v = Vector<Number>(spec);

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

  auto mat = make_sparse_matrix<Number>(spec, spec);
  mat.set_entries(coo);

  local_slice(v) = { 1.0, 0.0, 0.0, 1.0 };

  auto w = cg(mat, v);
  { auto loc = local_slice(w);
    EXPECT_EQ(loc[0], 1.0);
    EXPECT_EQ(loc[1], 1.0);
    EXPECT_EQ(loc[2], 1.0);
    EXPECT_EQ(loc[3], 1.0);
  }
}

TEST(CgSolver, solve1)
{
  using Number = std::complex<double>;
  using Vector = DefaultVector<Number>;

  class ScalingOp : public LinearOperator<Vector> {
    public:
      void apply(Vector& output,
                 const Vector& input) override
      {
        output.assign(input);
        output *= 5;
      }
  };
  auto mat = std::make_shared<ScalingOp>();

  VectorSpec spec(Comm::world(), 1, 1);
  auto v = std::make_shared<Vector>(spec);
  auto w = std::make_shared<Vector>(spec);

  LocalCooMatrix<Number> coo;
  coo.add(0, 0, 5);

  local_slice(*v) = { 1.0 };

  CgSolver<Vector> solver;
  solver.setup(mat);
  solver.solve(*v, *w);

  { auto loc = local_slice(*w);
    EXPECT_EQ(loc[0], 0.2);
  }
}

TEST(CgSolver, solve4)
{
  using Number = std::complex<double>;

  VectorSpec spec(Comm::world(), 4, 4);
  DefaultVector<Number> v(spec);

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

  auto mat = std::make_unique<DefaultSparseMatrix<Number>>(spec, spec);
  mat->set_entries(coo);

  local_slice(v) = { 1.0, 0.0, 0.0, 1.0 };

  CgSolver<DefaultVector<Number>> solver;
  DefaultVector<Number> w(spec);

  solver.setup(std::move(mat));
  solver.solve(v, w);

  { auto loc = local_slice(w);
    EXPECT_EQ(loc[0], 1.0);
    EXPECT_EQ(loc[1], 1.0);
    EXPECT_EQ(loc[2], 1.0);
    EXPECT_EQ(loc[3], 1.0);
  }
}


