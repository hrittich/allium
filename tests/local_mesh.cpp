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

#include <allium/mesh/local_mesh.hpp>

#include <gtest/gtest.h>

using namespace allium;

TEST(LocalMesh, CreateFillRead)
{
  Range<2> range({0,0}, {2,2});

  LocalMesh<double, 2> mesh(range);

  for (auto p : range) {
    mesh[p] = 100 * p[1] + p[0];
  }

  for (auto p : range) {
    EXPECT_EQ(mesh[p], 100 * p[1] + p[0]);
  }
}
