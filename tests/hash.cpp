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

#include <allium/util/hash.hpp>

#include <gtest/gtest.h>

using namespace allium;

TEST(Hash, BitScatter)
{
  EXPECT_EQ(bit_scatter(0, 1), 0);
  EXPECT_EQ(bit_scatter(1, 1), 1);
  EXPECT_EQ(bit_scatter(0b00001011, 1), 0b01000101);
  EXPECT_EQ(bit_scatter(0b00001011, 2), 0b001000001001);
}

TEST(Hash, ZCurve)
{
  EXPECT_EQ(z_curve(0, 0), 0);
  EXPECT_EQ(z_curve(0x01, 0x01), 0x03);
  EXPECT_EQ(z_curve(0b1011, 0b0101), 0b01100111);
}

