#include <gtest/gtest.h>
#include <chive/la/gmres.impl.hpp>

using namespace chive;

TEST(GMRES, Givens1)
{
  double a = 1;
  double b = 0;
  auto c_s = givens<double>(a, b);
  auto c = std::get<0>(c_s);
  auto s = std::get<1>(c_s);

  EXPECT_DOUBLE_EQ(c*c + s*s, 1);
  EXPECT_DOUBLE_EQ(s*a + c*b, 0);
}

TEST(GMRES, Givens2)
{
  double a = 1;
  double b = 2;
  auto c_s = givens<double>(a, b);
  auto c = std::get<0>(c_s);
  auto s = std::get<1>(c_s);

  EXPECT_DOUBLE_EQ(c*c + s*s, 1);
  EXPECT_DOUBLE_EQ(s*a + c*b, 0);
}

TEST(GMRES, Givens3)
{
  double a = 2;
  double b = 1;
  auto c_s = givens<double>(a, b);
  auto c = std::get<0>(c_s);
  auto s = std::get<1>(c_s);

  EXPECT_DOUBLE_EQ(c*c + s*s, 1);
  EXPECT_DOUBLE_EQ(s*a + c*b, 0);
}

TEST(GMRES, Givens4)
{
  auto a = std::complex<double>(1, 2);
  auto b = std::complex<double>(3, 4);
  auto c_s = givens(a, b);
  auto c = std::get<0>(c_s);
  auto s = std::get<1>(c_s);

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
  ASSERT_EQ(1, solution.nrows());
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
  ASSERT_EQ(solution.nrows(), 2);
  EXPECT_DOUBLE_EQ(solution[0], -1);
  EXPECT_DOUBLE_EQ(solution[1],  1);
}

TEST(GMRES, HessenbergQrSolve3)
{
  using complex = std::complex<double>;

  LocalVector<complex> rhs{-1, complex(1,2), 7};

  LocalVector<complex> col1{0.0, 1.0};

  LocalVector<complex> col2{-2, complex(0,1), 2};

  HessenbergQr<complex> qr(rhs[0]);
  qr.add_column(col1, rhs[1]);
  qr.add_column(col2, rhs[2]);

  EXPECT_DOUBLE_EQ(qr.residual_norm(), sqrt(2)*3);

  LocalVector<complex> solution = qr.solution();
  ASSERT_EQ(solution.nrows(), 2);
  EXPECT_DOUBLE_EQ(std::real(solution[0]), 1);
  EXPECT_DOUBLE_EQ(std::imag(solution[0]), 0);
  EXPECT_DOUBLE_EQ(std::real(solution[1]), 2);
  EXPECT_DOUBLE_EQ(std::imag(solution[1]), 0);
}



