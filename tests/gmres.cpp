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

#include <allium/la/gmres.impl.hpp>

#include <gtest/gtest.h>

using namespace allium;

TEST(GMRES, Givens1)
{
  double a = 1;
  double b = 0;
  double c, s;
  givens<double>(c, s, a, b);

  EXPECT_DOUBLE_EQ(c*c + s*s, 1);
  EXPECT_DOUBLE_EQ(s*a + c*b, 0);
}

TEST(GMRES, Givens2)
{
  double a = 1;
  double b = 2;
  double c, s;
  givens<double>(c, s, a, b);

  EXPECT_DOUBLE_EQ(c*c + s*s, 1);
  EXPECT_DOUBLE_EQ(s*a + c*b, 0);
}

TEST(GMRES, Givens3)
{
  double a = 2;
  double b = 1;
  double c, s;
  givens<double>(c, s, a, b);

  EXPECT_DOUBLE_EQ(c*c + s*s, 1);
  EXPECT_DOUBLE_EQ(s*a + c*b, 0);
}

TEST(GMRES, Givens4)
{
  auto a = std::complex<double>(1, 2);
  auto b = std::complex<double>(3, 4);
  std::complex<double> c, s;
  givens(c, s, a, b);

  EXPECT_DOUBLE_EQ(std::abs(c)*std::abs(c) + std::abs(s)*std::abs(s), 1.0);
  EXPECT_DOUBLE_EQ(std::abs(std::conj(s)*a + c*b - 0.0), 0);
}

TEST(GMRES, HessenbergQrSolve1)
{
  HessenbergQr<double> qr(1);

  LocalVector<double> col{0, 1};
  qr.add_column(col, 2);

  EXPECT_EQ(1, qr.residual_norm());

  LocalVector<double> solution = qr.solution();
  ASSERT_EQ(1, solution.rows());
  EXPECT_DOUBLE_EQ(2.0, solution[0]);
}

TEST(GMRES, HessenbergQrSolve2)
{
  LocalVector<double> rhs{(2-3*sqrt(6))/4,
                          (2*sqrt(3)+3*sqrt(2))/4,
                          3*sqrt(2)/2};

  LocalVector<double> col1{1.0/2,
                           sqrt(3)/2};

  LocalVector<double> col2{(4-3*sqrt(6))/4,
                           (4*sqrt(3)+3*sqrt(2))/4,
                           3*sqrt(2)/2};

  HessenbergQr<double> qr(rhs[0]);
  qr.add_column(col1, rhs[1]);
  qr.add_column(col2, rhs[2]);

  EXPECT_DOUBLE_EQ(qr.residual_norm(), 0);

  LocalVector<double> solution = qr.solution();
  ASSERT_EQ(solution.rows(), 2);
  EXPECT_DOUBLE_EQ(solution[0], -1);
  EXPECT_DOUBLE_EQ(solution[1],  1);
}

TEST(GMRES, HessenbergQrSolve3)
{
  // Test case with complex numbers
  using complex = std::complex<double>;

  LocalVector<complex> rhs{-1, complex(1,2), 7};
  LocalVector<complex> col1{0.0, 1.0};
  LocalVector<complex> col2{-2, complex(0,1), 2};

  HessenbergQr<complex> qr(rhs[0]);
  qr.add_column(col1, rhs[1]);
  qr.add_column(col2, rhs[2]);

  EXPECT_DOUBLE_EQ(qr.residual_norm(), sqrt(2)*3);

  LocalVector<complex> solution = qr.solution();
  ASSERT_EQ(solution.rows(), 2);
  EXPECT_DOUBLE_EQ(std::abs(solution[0] - 1.0), 0);
  EXPECT_DOUBLE_EQ(std::abs(solution[1] - 2.0), 0);
}

TEST(GMRES, HessenbergQrSolve4)
{
  // This test case requires a complex rotation
  using Number = std::complex<double>;

  LocalVector<Number> rhs{Number(0,1), -4.0};
  LocalVector<Number> col1{0, -2};

  HessenbergQr<Number> qr(rhs[0]);
  qr.add_column(col1, rhs[1]);

  EXPECT_DOUBLE_EQ(qr.residual_norm(), 1);

  LocalVector<Number> solution = qr.solution();
  ASSERT_EQ(solution.rows(), 1);
  EXPECT_DOUBLE_EQ(std::abs(solution[0] - 2.0), 0);
}

TEST(GMRES, InnerSolve1)
{
  using Number = std::complex<double>;

  VectorSpec spec(Comm::world(), 1, 1);
  DefaultVector<Number> v(spec);

  LocalCooMatrix<Number> coo;
  coo.add(0, 0, 5);

  auto mat = std::make_shared<DefaultSparseMatrix<Number>>(spec, spec);
  mat->set_entries(coo);

  local_slice(v) = { 1.0 };

  DefaultVector<Number> w(spec);
  gmres(w, mat, v, 1e-6);

  { auto loc = local_slice(w);
    EXPECT_EQ(loc[0], 0.2);
  }
}

TEST(GMRES, solve2)
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

  auto mat = std::make_shared<DefaultSparseMatrix<Number>>(spec, spec);
  mat->set_entries(coo);

  local_slice(v) = { 1.0, 0.0, 0.0, 1.0 };

  DefaultVector<Number> w(spec);
  gmres(w, mat, v, 1e-10);
  { auto loc = local_slice(w);
    EXPECT_LE(std::abs(loc[0] - 1.0), 1e-14);
    EXPECT_LE(std::abs(loc[1] - 1.0), 1e-14);
    EXPECT_LE(std::abs(loc[2] - 1.0), 1e-14);
    EXPECT_LE(std::abs(loc[3] - 1.0), 1e-14);
  }
}

