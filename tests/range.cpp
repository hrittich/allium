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

#include <allium/mesh/range.hpp>

#include <gtest/gtest.h>

using namespace allium;

TEST(Range, OneElement1D)
{
  Range<1> r({1}, {2});

  EXPECT_EQ(r.size(), 1);

  auto iter = r.begin();
  EXPECT_EQ(*iter, (Point<int, 1>{1}));

  ++iter;
  EXPECT_EQ(iter, r.end());
}

TEST(Range, TwoElement1D)
{
  Range<1> r({1}, {3});

  EXPECT_EQ(r.size(), 2);

  auto iter = r.begin();
  EXPECT_EQ(*iter, (Point<int, 1>{1}));

  ++iter;
  EXPECT_EQ(*iter, (Point<int, 1>{2}));

  ++iter;
  EXPECT_EQ(iter, r.end());
}

TEST(Range, SixElement2D)
{
  Range<2> r({1, 2}, {4, 4});

  EXPECT_EQ(r.size(), 6);

  auto iter = r.begin();
  EXPECT_EQ(*iter, (Point<int, 2>{1, 2}));

  ++iter;
  EXPECT_EQ(*iter, (Point<int, 2>{1, 3}));

  ++iter;
  EXPECT_EQ(*iter, (Point<int, 2>{2, 2}));

  ++iter;
  EXPECT_EQ(*iter, (Point<int, 2>{2, 3}));

  ++iter;
  EXPECT_EQ(*iter, (Point<int, 2>{3, 2}));

  ++iter;
  EXPECT_EQ(*iter, (Point<int, 2>{3, 3}));

  ++iter;
  EXPECT_EQ(iter, r.end());
}

TEST(Range, In)
{
  Range<2> r({1,2}, {4,7});

  EXPECT_TRUE(r.in({1,2}));
  EXPECT_TRUE(r.in({2,3}));
  EXPECT_FALSE(r.in({0, 3}));
  EXPECT_FALSE(r.in({3, 8}));
  EXPECT_FALSE(r.in({4,7}));
}

TEST(Range, RangeIndex1D)
{
  Range<1> r({1}, {3});

  EXPECT_EQ(0, r.index(Point<int, 1>{1}));
  EXPECT_EQ(1, r.index(Point<int, 1>{2}));
}

TEST(Range, RangeIndex2D)
{
  Range<2> r({3,1}, {5,4});

  EXPECT_EQ(0, r.index({3,1}));
  EXPECT_EQ(1, r.index({3,2}));
  EXPECT_EQ(2, r.index({3,3}));
  EXPECT_EQ(3, r.index({4,1}));
  EXPECT_EQ(4, r.index({4,2}));
  EXPECT_EQ(5, r.index({4,3}));
}

TEST(Range, RangeSize)
{
  Range<2> range({0,1},{2,4});
  EXPECT_EQ(range.size(), 6);
}

