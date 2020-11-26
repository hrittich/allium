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

#include <allium/main/init.hpp>
#include <allium/mesh/petsc_mesh_spec.hpp>
#include <allium/mesh/petsc_mesh.hpp>
#include <allium/ipc/comm.hpp>
#include <allium/util/memory.hpp>
#include <allium/ode/imex_euler.hpp>
#include <iomanip>

using namespace allium;

using Number = PetscScalar;
using Real = real_part_t<Number>;

/** Initialize the boundary of the mesh with zeros. */
void zero_boundary(PetscLocalMesh<2>& mesh)
{
  // the range of the whole mesh
  auto global_range = mesh.mesh_spec()->range();

  // the range associated to the current processor
  auto range = mesh.mesh_spec()->local_ghost_range();

  // Access the local data of the mesh.
  // Note, this can be a costly operation in the case that the data has to
  // be transferred from an accelerator.
  auto lmesh = local_mesh(mesh);

  // Iterate over all mesh points
  for (auto p : range) {
    if (p[0] == -1
        || p[1] == -1
        || p[0] == global_range.end_pos()[0]
        || p[1] == global_range.end_pos()[1]) {
      // set the boundary value to zero
      lmesh(p[0], p[1]) = 0.0;
    }
  }
}

/** Set the mesh values to the initial values of the function. */
void initial_values(PetscMesh<2>& result)
{
  auto range = result.mesh_spec()->local_range();

  auto lresult = local_mesh(result);
  for (auto p : range) {
    if (p[0] == 5 && p[1] == 5) {
      lresult(p[0], p[1]) = 1.0;
    } else {
      lresult(p[0], p[1]) = 0.0;
    }
  }
}

/** Returns the implicit right-hand side given the mesh-point distance h. */
auto make_f_impl(double h) {
  return
    [=](PetscMesh<2>& f, Real t, const PetscMesh<2>& u)
    {
      // PETSc require to create a "local mesh" to have access to the ghost
      // nodes.
      PetscLocalMesh<2> u_aux(u.mesh_spec());
      u_aux.assign(u); // copy to determine the ghost nodes

      // We set the boundary to zero such that we can apply the same stencil
      // everywhere (also at the boundary).
      zero_boundary(u_aux);

      auto range = u.mesh_spec()->local_range();
      auto lu = local_mesh(u_aux);
      auto lf = local_mesh(f);

      double diffusion = 0.1;

      // Apply the stencil
      // |      1     |
      // | 1   -4   1 |
      // |      1     |h
      for (auto p : range) {
        lf(p[0], p[1])
          = diffusion
            * (1.0 / (h*h))
            * ( -4 * lu(p[0],   p[1])
                + lu(p[0]-1, p[1])
                + lu(p[0],   p[1]-1)
                + lu(p[0],   p[1]+1)
                + lu(p[0]+1, p[1]));
      }
    };
};

/** The explicit part of the ODE. */
void f_expl(PetscMesh<2>& result, Real t, const PetscMesh<2>& u)
{
  auto range = u.mesh_spec()->local_range();
  auto lresult = local_mesh(result);
  auto lu = local_mesh(u);

  double rate = 2;

  for (auto p : range) {
    lresult(p[0], p[1]) = rate * lu(p[0], p[1]) * (1.0 - lu(p[0], p[1]));
  }
}

/** Print the mesh on STDOUT, sorted by MPI ranks. */
template <typename M>
void print_mesh(const M& mesh) {
  auto comm = mesh.mesh_spec()->comm();

  for (int i=0; i < comm.size(); ++i) {
    if (i == comm.rank()) {
      std::cout << "rank " << comm.rank() << std::endl;

      auto lmesh = local_mesh(mesh);
      auto range = mesh.local_range();
      for (int i = range.begin_pos()[0]; i < range.end_pos()[0]; i++) {
        for (int j = range.begin_pos()[1]; j < range.end_pos()[1]; ++j) {
          std::cout << std::setw(10) << lmesh(i, j) << " ";
        }
        std::cout << std::endl;
      }

    }

    comm.barrier();
  }
}

int main(int argc, char** argv)
{
  using namespace std::placeholders;

  Init init(argc, argv);  // initialize Allium

  const int N = 10;
  double h = 1.0 / (N-1);

  auto comm = Comm::world();

  if (comm.rank() == 0)
    std::cout << "Fisher 2D solver" << std::endl;

  // Create a mesh. For description of the parameters, see the PETSc manual
  auto spec = std::shared_ptr<PetscMeshSpec<2>>(
                new PetscMeshSpec<2>(
                  comm,
                  {DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED},
                  DMDA_STENCIL_STAR,
                  {N, N}, // global size
                  {PETSC_DECIDE, PETSC_DECIDE}, // processors per dim
                  1, // ndof
                  1)); // stencil_width

  PetscMesh<2> u(spec);

  // setup the integrator
  ImexEuler<PetscMesh<2>> integrator;
  integrator.setup(f_expl, make_f_impl(h));

  initial_values(u);
  integrator.initial_values(0, u);
  integrator.dt(0.001);

  print_mesh(u);

  integrator.integrate(u, 10);

  print_mesh(u);

  return 0;
}

