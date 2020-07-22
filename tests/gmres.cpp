#include <gtest/gtest.h>
#include <chive/la/gmres.impl.hpp>

using namespace chive;

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



