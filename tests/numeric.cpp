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

#include <allium/util/numeric.hpp>
#include <allium/util/warnings.hpp>

#include <gtest/gtest.h>

using namespace allium;

TEST(Numeric, test_compiler_signed) {
  // assert out assumptions on the C++ compiler
  ALLIUM_NO_SIGN_COMPARE_WARNING
  EXPECT_TRUE(-1 >= 1u);
  ALLIUM_RESTORE_WARNING
}

TEST(Numeric, safe_le) {
  // both positive
  EXPECT_TRUE(safe_le(1u, 2u));
  EXPECT_FALSE(safe_le(2u, 1u));

  // both negative
  EXPECT_FALSE(safe_le(-1, -2));
  EXPECT_TRUE(safe_le(-2, -1));

  // both equal
  EXPECT_TRUE(safe_le(1u, 1u));
  EXPECT_TRUE(safe_le(-1, -1));

  // positive and negative
  EXPECT_TRUE(safe_le(-1, 1u));
  EXPECT_FALSE(safe_le(1u, -1));
}

TEST(Numeric, safe_ge) {
  // both positive
  EXPECT_FALSE(safe_ge(1u, 2u));
  EXPECT_TRUE(safe_ge(2u, 1u));

  // both negative
  EXPECT_TRUE(safe_ge(-1, -2));
  EXPECT_FALSE(safe_ge(-2, -1));

  // both equal
  EXPECT_TRUE(safe_ge(1u, 1u));
  EXPECT_TRUE(safe_ge(-1, -1));

  // positive and negative
  EXPECT_FALSE(safe_ge(-1, 1u));
  EXPECT_TRUE(safe_ge(1u, -1));
}

TEST(Numeric, safe_lt) {
  // both positive
  EXPECT_TRUE(safe_lt(1u, 2u));
  EXPECT_FALSE(safe_lt(2u, 1u));

  // both negative
  EXPECT_FALSE(safe_lt(-1, -2));
  EXPECT_TRUE(safe_lt(-2, -1));

  // both equal
  EXPECT_FALSE(safe_lt(1u, 1u));
  EXPECT_FALSE(safe_lt(-1, -1));

  // positive and negative
  EXPECT_TRUE(safe_lt(-1, 1u));
  EXPECT_FALSE(safe_lt(1u, -1));
}

TEST(Numeric, safe_gt) {
  // both positive
  EXPECT_FALSE(safe_gt(1u, 2u));
  EXPECT_TRUE(safe_gt(2u, 1u));

  // both negative
  EXPECT_TRUE(safe_gt(-1, -2));
  EXPECT_FALSE(safe_gt(-2, -1));

  // both equal
  EXPECT_FALSE(safe_gt(1u, 1u));
  EXPECT_FALSE(safe_gt(-1, -1));

  // positive and negative
  EXPECT_FALSE(safe_gt(-1, 1u));
  EXPECT_TRUE(safe_gt(1u, -1));
}

