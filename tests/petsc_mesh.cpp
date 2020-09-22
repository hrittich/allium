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

#include <gtest/gtest.h>
#include <allium/config.hpp>

#ifdef ALLIUM_USE_PETSC

#include <allium/mesh/petsc_mesh.hpp>
#include <allium/util/hash.hpp>

using namespace allium;

TEST(PetscMesh, Simple4x4)
{
  auto comm = Comm::world();
  if (comm.size() < 4) {
    std::cerr << "WARNING: at least 4 ranks required" << std::endl;
    GTEST_SKIP();
  }

  // we use only 4 rank
  auto sub = comm.split(comm.rank() < 4 ? 0 : 1, 0);
  if (comm.rank() >= 4)
    return;

  auto spec = std::shared_ptr<PetscMeshSpec<2>>(
                new PetscMeshSpec<2>(
                  sub,
                  {DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED},
                  DMDA_STENCIL_BOX,
                  {2, 2}, // global size
                  {2, 2}, // processors per dim
                  1, // ndof
                  1)); // stencil_width

  PetscMesh<2> global(spec);

  auto p = spec->range().begin_pos();
  {
    PetscMeshValues<2> v(&global);
    v(p[0], p[1]) = 10 * p[0] + p[1];
  }

  PetscLocalMesh<2> local(spec);
  local.assign(global);

  {
    PetscMeshValues<2> v(&local);
    EXPECT_EQ(v(0,0), 00.0);
    EXPECT_EQ(v(0,1), 01.0);
    EXPECT_EQ(v(1,0), 10.0);
    EXPECT_EQ(v(1,1), 11.0);
  }
}

TEST(PetscMesh, FixedLocalSize2D) {
  auto comm = Comm::world();

  unsigned int N = (int)(100 * sqrt(comm.size()));

  auto spec = std::shared_ptr<PetscMeshSpec<2>>(
                new PetscMeshSpec<2>(
                  comm,
                  {DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED},
                  DMDA_STENCIL_BOX,
                  {N, N}, // global size
                  {PETSC_DECIDE, PETSC_DECIDE}, // processors per dim
                  1, // ndof
                  1)); // stencil_width

  PetscMesh<2> global(spec);
  PetscLocalMesh<2> local(spec);

  {
    PetscMeshValues<2> val(&global);

    std::cout <<
      "normal: " << spec->range() << " "
      "ghosted: " << spec->ghost_range() << std::endl;

    for (auto p : spec->range()) {
      //std::cout << p << std::endl;
      val(p[0], p[1]) = z_curve(p[0], p[1]);
    }
  }

  local.assign(global);

  {
    PetscMeshValues<2> val(&local);

    auto r = spec->range();
    auto local_begin_pos = r.begin_pos() - r.begin_pos();
    auto local_end_pos = r.end_pos() - r.begin_pos();

    for (auto p : spec->ghost_range()) {
      if (p[0] < 0 || p[1] < 0 || p[0] >= N || p[1] >= N)
        continue;

      EXPECT_EQ(val(p[0], p[1]), (double)z_curve(p[0], p[1]));
    }
  }
}

#endif
