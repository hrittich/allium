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

#include <allium/la/local_vector.hpp>
#include <sstream>

#include <gtest/gtest.h>

using namespace allium;

TEST(LocalVectorTest, constant)
{
  auto v = LocalVector<double>::constant(3, 42);
  std::stringstream s;

  EXPECT_EQ(v.rows(), 3);
  EXPECT_EQ(v[0], 42);
  EXPECT_EQ(v[1], 42);
  EXPECT_EQ(v[2], 42);
}

TEST(LocalVectorTest, output)
{
  LocalVector<double> v(3);
  std::stringstream s;

  v[0] = 1;
  v[1] = 2;
  v[2] = 3;

  s << v;

  EXPECT_EQ(std::string("1 2 3"), s.str());
}

