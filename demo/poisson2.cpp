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
#include <allium/la/linear_operator.hpp>
#include <allium/la/cg.hpp>
#include <allium/util/memory.hpp>
#include <iomanip>

using namespace allium;

void zero_boundary(PetscLocalMesh<2>& mesh)
{
  auto global_range = mesh.mesh_spec()->range();
  auto range = mesh.mesh_spec()->local_ghost_range();
  auto val = local_mesh(mesh);
  for (auto p : range) {
    if (p[0] == -1
        || p[1] == -1
        || p[0] == global_range.end_pos()[0]
        || p[1] == global_range.end_pos()[1]) {
      val(p[0], p[1]) = 0.0;
    }
  }
}

// harmonic function: f(x, y) = x^3 - 3 * x * v^2
double solution(double x, double y) {
  return x*x*x - 3 * x*y*y;
}

void initialize_exact_solution(PetscMesh<2>& mesh, double h)
{
  auto range = mesh.mesh_spec()->local_range();
  auto val = local_mesh(mesh);

  for (auto p : range) {
    double x = p[0] * h;
    double y = p[1] * h;
    val(p[0], p[1]) = solution(x, y);
  }
}

void initialize_boundary(PetscLocalMesh<2>& mesh, double h)
{
  auto global_range = mesh.mesh_spec()->range();
  auto range = mesh.mesh_spec()->local_ghost_range();
  auto val = local_mesh(mesh);

  std::cout << global_range << std::endl;

  for (auto p : range) {
    if (p[0] == -1
        || p[1] == -1
        || p[0] == global_range.end_pos()[0]
        || p[1] == global_range.end_pos()[1])
    {
      double x = p[0] * h;
      double y = p[1] * h;
      val(p[0], p[1]) = solution(x, y);
    } else {
      val(p[0], p[1]) = 0.0;
    }
  }
}

void apply_laplace(PetscMesh<2>& f, double h, const PetscLocalMesh<2>& u)
{
  auto range = u.mesh_spec()->local_range();

  auto u_val = local_mesh(u);
  auto f_val = local_mesh(f);

  for (auto p : range) {
    f_val(p[0], p[1]) = 1.0 / (h*h) * ( 4 * u_val(p[0],   p[1])
                                        -u_val(p[0]-1, p[1])
                                        -u_val(p[0],   p[1]-1)
                                        -u_val(p[0],   p[1]+1)
                                        -u_val(p[0]+1, p[1]));
  }
}

template <typename M>
void print_mesh(const M& mesh) {
  auto comm = mesh.mesh_spec()->comm();

  for (int i=0; i < comm.size(); ++i) {
    if (i == comm.rank()) {
      std::cout << "rank " << comm.rank() << std::endl;

      auto val = local_mesh(mesh);
      auto range = mesh.local_range();
      for (int i = range.begin_pos()[0]; i < range.end_pos()[0]; i++) {
        for (int j = range.begin_pos()[1]; j < range.end_pos()[1]; ++j) {
          std::cout << std::setw(10) << val(i, j) << " ";
        }
        std::cout << std::endl;
      }

    }

    comm.barrier();
  }
}

void initialize_rhs(double h, PetscMesh<2>& rhs)
{
  PetscLocalMesh<2> domain(rhs.mesh_spec());

  initialize_boundary(domain, h);

  print_mesh(domain); std::cout << std::endl;

  apply_laplace(rhs, h, domain);

  rhs *= (-1.0);
}

int main(int argc, char** argv)
{
  Init init(argc, argv);
  const int N = 10;
  double h = 1.0 / (N-1);

  auto comm = Comm::world();

  if (comm.rank() == 0)
    std::cout << "Poisson 2D solver" << std::endl;

  auto spec = std::shared_ptr<PetscMeshSpec<2>>(
                new PetscMeshSpec<2>(
                  comm,
                  {DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED},
                  DMDA_STENCIL_STAR,
                  {N, N}, // global size
                  {PETSC_DECIDE, PETSC_DECIDE}, // processors per dim
                  1, // ndof
                  1)); // stencil_width

  PetscMesh<2> rhs(spec);
  PetscMesh<2> solution(spec);

  CgSolver<PetscMesh<2>> solver(1e-10);
  auto op
    = shared_copy(make_linear_operator<PetscMesh<2>>(
        [h](PetscMesh<2>& result, const PetscMesh<2>& arg) {
          PetscLocalMesh<2> local_arg(arg.mesh_spec());
          local_arg.assign(arg);
          zero_boundary(local_arg);
          apply_laplace(result, h, local_arg);
        }));
  solver.setup(op);

  initialize_rhs(h, rhs);
  print_mesh(rhs); std::cout << std::endl;
  set_zero(solution);
  solver.solve(rhs, solution);

  PetscMesh<2> residual(spec);  // b - Ax
  PetscMesh<2> tmp1(spec);
  op->apply(tmp1, solution);
  residual.assign(rhs);
  residual.add_scaled(-1.0, tmp1);
  std::cout << "residual, l2 norm: " << residual.l2_norm() << std::endl;

  print_mesh(solution);
  std::cout << std::endl;

  PetscMesh<2> error(spec);
  initialize_exact_solution(error, h);
  print_mesh(error);
  error.add_scaled(-1.0, solution);

  std::cout << std::endl;
  print_mesh(error);

  std::cout << "l2 error: " << error.l2_norm() << std::endl;

  return 0;
}

