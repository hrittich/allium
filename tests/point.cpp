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

#include <allium/mesh/point.hpp>

#include <gtest/gtest.h>

using namespace allium;

TEST(Point, CreateSetGet)
{
  Point<int, 1> p;
  p[0] = 99;

  EXPECT_EQ(p[0], 99);
}

TEST(Point, Initialize)
{
  Point<int, 3> p{4, 7, 12};
  EXPECT_EQ(p[0], 4);
  EXPECT_EQ(p[1], 7);
  EXPECT_EQ(p[2], 12);
}

TEST(Point, Add)
{
  Point<int, 1> p{2}, q{3};
  p += q;
  EXPECT_EQ(p[0], 5);
}

TEST(Point, Scale)
{
  Point<int, 1> p{2};
  p *= 3;
  EXPECT_EQ(p[0], 6);
}

TEST(Point, Join)
{
  Point<int, 2> p{2, 3};
  auto q = p.joined(4);

  EXPECT_EQ(q[0], 2);
  EXPECT_EQ(q[1], 3);
  EXPECT_EQ(q[2], 4);
}

TEST(Point, Prod)
{
  Point<int, 2> p{2, 3};

  EXPECT_EQ(p.prod(), 6);
}

