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

#include <allium/mesh/cuda_mesh.hpp>
#include <allium/mesh/cuda_mesh_algorithm.hpp>

#include <gtest/gtest.h>

using namespace allium;

TEST(CudaTest, MeshCopy)
{
  Range<2> range({0,0}, {2,2});

  LocalMesh<double, 2> host(range);

  host(0,0) = 2.0;
  host(0,1) = 4.0;
  host(1,0) = 8.0;
  host(1,1) = 16.0;

  CudaNArray<double, 2> device(range.shape());
  device.copy_from(host);

  LocalMesh<double, 2> h_check(range);
  device.copy_to(h_check);

  EXPECT_EQ(h_check(0,0), 2.0);
  EXPECT_EQ(h_check(0,1), 4.0);
  EXPECT_EQ(h_check(1,0), 8.0);
  EXPECT_EQ(h_check(1,1), 16.0);
}

TEST(CudaTest, MeshFivePointDirichlet)
{
  Range<2> range({0,0}, {4,4});

  LocalMesh<double, 2> host(range);

  for (auto p : range) {
    host[p] = 0.0;
  }

  host(1,1) = 1.0;
  host(1,2) = 2.0;
  host(2,1) = 3.0;
  host(2,2) = 4.0;

  std::array<double, 5> coeff = { 10, 20, 30, 40, 50 };

  CudaNArray<double, 2> d_in(range.shape());
  CudaNArray<double, 2> d_out(range.shape());

  d_in.copy_from(host);

  MeshDataLayout layout = {
    .pitch = d_in.pitch(),
    .start = { 1, 1 },
    .end = { range.shape()[0] - 1, range.shape()[1] }
  };

  cuda_mesh_five_point(d_out.ptr(), layout, coeff.data(), d_in.ptr());

  d_out.copy_to(host);

  EXPECT_EQ(host(1,1), 260);
  EXPECT_EQ(host(1,2), 280);
  EXPECT_EQ(host(2,1), 260);
  EXPECT_EQ(host(2,2), 200);
}

TEST(CudaTest, MeshMap)
{
  Range<2> range({0,0}, {1,1});

  LocalMesh<double, 2> host(range);

  host(0,0) = 3.0;

  CudaNArray<double, 2> device(range.shape());
  device.copy_from(host);

  MeshDataLayout layout {
    .pitch = device.pitch(),
    .start = { 0, 0 },
    .end = { range.shape()[0], range.shape()[1] }
  };

  cuda_mesh_map(device.ptr(), layout, cuda_op::Inc<double>());

  device.copy_to(host);

  EXPECT_EQ(host(0,0), 4.0);
}

TEST(CudaTest, MeshMapWithGhost)
{
  Range<2> range({0,0}, {4,4});

  LocalMesh<double, 2> host(range);

  host(0,0) = 1.0;
  host(0,1) = 2.0;
  host(0,2) = 3.0;
  host(0,3) = 4.0;

  host(1,0) = 5.0;
  host(1,1) = 6.0;
  host(1,2) = 7.0;
  host(1,3) = 8.0;

  host(2,0) = 9.0;
  host(2,1) = 10.0;
  host(2,2) = 11.0;
  host(2,3) = 12.0;

  host(3,0) = 13.0;
  host(3,1) = 14.0;
  host(3,2) = 15.0;
  host(3,3) = 16.0;

  CudaNArray<double, 2> device(range.shape());
  device.copy_from(host);

  MeshDataLayout layout {
    .pitch = device.pitch(),
    .start = { 1, 1 },
    .end = { range.shape()[0]-1, range.shape()[1]-1 }
  };

  cuda_mesh_map(device.ptr(), layout, cuda_op::Inc<double>());

  device.copy_to(host);

  EXPECT_EQ(host(0,0), 1.0);
  EXPECT_EQ(host(0,1), 2.0);
  EXPECT_EQ(host(0,2), 3.0);
  EXPECT_EQ(host(0,3), 4.0);

  EXPECT_EQ(host(1,0), 5.0);
  EXPECT_EQ(host(1,1), 7.0);
  EXPECT_EQ(host(1,2), 8.0);
  EXPECT_EQ(host(1,3), 8.0);

  EXPECT_EQ(host(2,0), 9.0);
  EXPECT_EQ(host(2,1), 11.0);
  EXPECT_EQ(host(2,2), 12.0);
  EXPECT_EQ(host(2,3), 12.0);

  EXPECT_EQ(host(3,0), 13.0);
  EXPECT_EQ(host(3,1), 14.0);
  EXPECT_EQ(host(3,2), 15.0);
  EXPECT_EQ(host(3,3), 16.0);
}

TEST(CudaTest, MeshMapReduce)
{
  Range<2> range({0,0}, {2,3});

  LocalMesh<double, 2> host(range);

  host(0,0) = 1.0;
  host(0,1) = 2.0;
  host(0,2) = 3.0;
  host(1,0) = 4.0;
  host(1,1) = 5.0;
  host(1,2) = 6.0;

  CudaNArray<double, 2> device(range.shape());
  device.copy_from(host);

  MeshDataLayout layout {
    .pitch = device.pitch(),
    .start = { 0, 0 },
    .end = { range.shape()[0], range.shape()[1] }
  };

  auto result
    = cuda_mesh_map_reduce<double, cuda_op::Sum<double>, cuda_op::Id<double>>
                          (layout, device.ptr());

  EXPECT_EQ(result, 21);
}

TEST(CudaTest, MeshMapReduceWithGhost)
{
  Range<2> range({0,0}, {3,4});

  LocalMesh<double, 2> host(range);

  host(0,0) = 1.0;
  host(0,1) = 2.0;
  host(0,2) = 3.0;
  host(0,3) = 4.0;
  host(1,0) = 5.0;
  host(1,1) = 6.0;
  host(1,2) = 7.0;
  host(1,3) = 8.0;
  host(2,0) = 9.0;
  host(2,1) = 10.0;
  host(2,2) = 11.0;
  host(2,3) = 12.0;

  CudaNArray<double, 2> device(range.shape());
  device.copy_from(host);

  MeshDataLayout layout {
    .pitch = device.pitch(),
    .start = { 1, 1 },
    .end = { range.shape()[0]-1, range.shape()[1]-1 }
  };

  auto result
    = cuda_mesh_map_reduce<double, cuda_op::Sum<double>, cuda_op::Id<double>>
                          (layout, device.ptr());

  EXPECT_EQ(result, 13);
}

TEST(CudaTest, MeshFill)
{
  Range<2> range({0,0}, {3,4});

  LocalMesh<double, 2> host(range);

  for (const auto p : range) {
    host[p] = 1.0;
  }

  CudaNArray<double, 2> device(range.shape());
  device.copy_from(host);

  MeshDataLayout layout {
    .pitch = device.pitch(),
    .start = { 0, 1 },
    .end = { range.shape()[0]-1, range.shape()[1]-1 }
  };

  cuda_mesh_fill<double>(device.ptr(), layout, 2.0);

  device.copy_to(host);

  EXPECT_EQ(host(0,0), 1.0);
  EXPECT_EQ(host(0,1), 2.0);
  EXPECT_EQ(host(0,2), 2.0);
  EXPECT_EQ(host(0,3), 1.0);

  EXPECT_EQ(host(1,0), 1.0);
  EXPECT_EQ(host(1,1), 2.0);
  EXPECT_EQ(host(1,2), 2.0);
  EXPECT_EQ(host(1,3), 1.0);

  EXPECT_EQ(host(2,0), 1.0);
  EXPECT_EQ(host(2,1), 1.0);
  EXPECT_EQ(host(2,2), 1.0);
  EXPECT_EQ(host(2,3), 1.0);
}

TEST(CudaTest, CudaMeshFillGhost)
{
  Range<2> range({0,0}, {1,2});
  Range<2> ghosted_range({0, 0}, {3, 4});

  CudaMesh<double, 2> mesh(range, 1);

  LocalMesh<double, 2> host(ghosted_range);

  for (const auto p : ghosted_range) {
    host[p] = 1.0;
  }

  mesh.narray().copy_from(host);
  mesh.fill_ghost_points(2.0);
  mesh.narray().copy_to(host);

  EXPECT_EQ(host(0,0), 2.0);
  EXPECT_EQ(host(0,1), 2.0);
  EXPECT_EQ(host(0,2), 2.0);
  EXPECT_EQ(host(0,3), 2.0);

  EXPECT_EQ(host(1,0), 2.0);
  EXPECT_EQ(host(1,1), 1.0);
  EXPECT_EQ(host(1,2), 1.0);
  EXPECT_EQ(host(1,3), 2.0);

  EXPECT_EQ(host(2,0), 2.0);
  EXPECT_EQ(host(2,1), 2.0);
  EXPECT_EQ(host(2,2), 2.0);
  EXPECT_EQ(host(2,3), 2.0);
}


