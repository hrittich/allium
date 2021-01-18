// Copyright 2021 Hannah Rittich
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

#include <allium/util/polynomial.hpp>
#include <complex>

#include <gtest/gtest.h>

using namespace allium;

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;

template <typename T>
struct PolynomialTest : public testing::Test {};

TYPED_TEST_SUITE(PolynomialTest, TestTypes);

#ifdef ALLIUM_USE_GSL
TYPED_TEST(PolynomialTest, EvalConstant)
{
  using N = TypeParam;
  Polynomial<N> p({1});

  EXPECT_EQ(p.eval(-1), N(1.0));
  EXPECT_EQ(p.eval( 0), N(1.0));
  EXPECT_EQ(p.eval( 1), N(1.0));
}

TYPED_TEST(PolynomialTest, EvalLinear)
{
  using N = TypeParam;
  Polynomial<N> p({1, 2});

  EXPECT_EQ(p.eval(-1), N(-1.0));
  EXPECT_EQ(p.eval( 0), N(1.0));
  EXPECT_EQ(p.eval( 1), N(3.0));
}

TYPED_TEST(PolynomialTest, EvalQuadratic)
{
  using N = TypeParam;
  Polynomial<N> p({-2, 0, 1});

  EXPECT_EQ(p.eval(-2), N( 2.0));
  EXPECT_EQ(p.eval( 0), N(-2.0));
  EXPECT_EQ(p.eval( 2), N( 2.0));
}
#endif

TYPED_TEST(PolynomialTest, DerivativeLinear)
{
  using N = TypeParam;
  Polynomial<N> p({2, 3});

  auto d = p.derivative();
  ASSERT_EQ(d.deg(), 0);
  EXPECT_EQ(d.coeffs().at(0), N(3));
}

TYPED_TEST(PolynomialTest, DerivativeQuadratic)
{
  using N = TypeParam;
  Polynomial<N> p({2, 3, 4});

  auto d = p.derivative();
  ASSERT_EQ(d.deg(), 1);
  EXPECT_EQ(d.coeffs().at(0), N(3));
  EXPECT_EQ(d.coeffs().at(1), N(8));
}

TYPED_TEST(PolynomialTest, AntiDerConst)
{
  using N = TypeParam;
  Polynomial<N> p({2});

  auto d = p.anti_derivative();
  ASSERT_EQ(d.deg(), 1);
  EXPECT_EQ(d.coeffs().at(0), N(0));
  EXPECT_EQ(d.coeffs().at(1), N(2));
}

TYPED_TEST(PolynomialTest, AntiDerLinear)
{
  using N = TypeParam;
  Polynomial<N> p({2, 6});

  auto d = p.anti_derivative();
  ASSERT_EQ(d.deg(), 2);
  EXPECT_EQ(d.coeffs().at(0), N(0));
  EXPECT_EQ(d.coeffs().at(1), N(2));
  EXPECT_EQ(d.coeffs().at(2), N(3));
}

#ifdef ALLIUM_USE_GSL
TYPED_TEST(PolynomialTest, IntegrateLinear)
{
  using N = TypeParam;
  Polynomial<N> p({1, 1});

  auto d = p.integrate(-1);
  ASSERT_EQ(d.deg(), 2);
  EXPECT_EQ(d.coeffs().at(0), N(0.5));
  EXPECT_EQ(d.coeffs().at(1), N(1));
  EXPECT_EQ(d.coeffs().at(2), N(0.5));
}
#endif

TYPED_TEST(PolynomialTest, MultiplyConst)
{
  using N = TypeParam;
  Polynomial<N> p({2}), q({3});

  auto r = p * q;
  ASSERT_EQ(r.deg(), 0);
  ASSERT_EQ(r.coeffs().at(0), N(6));
}

TYPED_TEST(PolynomialTest, MultiplyLinear)
{
  using N = TypeParam;
  Polynomial<N> p({2, 3}), q({2, -3});

  auto r = p * q;
  ASSERT_EQ(r.deg(), 2);
  ASSERT_EQ(r.coeffs().at(0), N(4));
  ASSERT_EQ(r.coeffs().at(1), N(0));
  ASSERT_EQ(r.coeffs().at(2), N(-9));
}

TYPED_TEST(PolynomialTest, MultiplyLinearAndQuadratic)
{
  using N = TypeParam;
  Polynomial<N> p({2, 3}), q({4, 5, 6});

  auto r = p * q;
  ASSERT_EQ(r.deg(), 3);
  ASSERT_EQ(r.coeffs().at(0), N(8));
  ASSERT_EQ(r.coeffs().at(1), N(22));
  ASSERT_EQ(r.coeffs().at(2), N(27));
  ASSERT_EQ(r.coeffs().at(3), N(18));
}

TYPED_TEST(PolynomialTest, AddConstant)
{
  using N = TypeParam;
  Polynomial<N> p({2}), q({3});

  auto r = p + q;
  ASSERT_EQ(r.deg(), 0);
  ASSERT_EQ(r.coeffs().at(0), N(5));
}

TYPED_TEST(PolynomialTest, AddConstantAndLinear)
{
  using N = TypeParam;
  Polynomial<N> p({2}), q({3, 4});

  auto r = p + q;
  ASSERT_EQ(r.deg(), 1);
  ASSERT_EQ(r.coeffs().at(0), N(5));
  ASSERT_EQ(r.coeffs().at(1), N(4));
}

using RealTypes = ::testing::Types<float, double>;

template <typename T>
struct RealPolynomialTest : public testing::Test {};

TYPED_TEST_SUITE(RealPolynomialTest, RealTypes);


#ifdef ALLIUM_USE_GSL
TYPED_TEST(RealPolynomialTest, RootsLinear)
{
  using N = TypeParam;
  Polynomial<N> p({1, 2});

  auto r = p.roots();
  ASSERT_EQ(r.size(), 1);
  EXPECT_EQ(r.at(0), std::complex<N>(-0.5));
}

TYPED_TEST(RealPolynomialTest, RootsOfUnity)
{
  using N = TypeParam;
  Polynomial<N> p({-1, 0, 0, 0, 1});

  auto r = p.roots();
  ASSERT_EQ(r.size(), 4);

  EXPECT_EQ(std::count_if(
              r.begin(), r.end(),
              [](std::complex<N> z) { return abs(z - std::complex<N>(1.0)) < 1e-6; }),
            1);
  EXPECT_EQ(std::count_if(
              r.begin(), r.end(),
              [](std::complex<N> z) { return abs(z - std::complex<N>(0.0, 1.0)) < 1e-6; }),
            1);
  EXPECT_EQ(std::count_if(
              r.begin(), r.end(),
              [](std::complex<N> z) { return abs(z - std::complex<N>(-1.0)) < 1e-6; }),
            1);
  EXPECT_EQ(std::count_if(
              r.begin(), r.end(),
              [](std::complex<N> z) { return abs(z - std::complex<N>(0.0, -1.0)) < 1e-6; }),
            1);
}
#endif

