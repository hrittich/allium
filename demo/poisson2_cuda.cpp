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
#include <allium/mesh/cuda_mesh.hpp>
#include <allium/ipc/comm.hpp>
#include <allium/la/linear_operator.hpp>
#include <allium/la/cg.hpp>
#include <allium/util/memory.hpp>
#include <iostream>
#include <iomanip>

using namespace allium;

using Mesh = CudaMesh<double, 2>;

// harmonic function: f(x, y) = x^3 - 3 * x * v^2
double solution(double x, double y) {
  return x*x*x - 3 * x*y*y;
}

void initialize_exact_solution(Mesh& mesh, double h)
{
  auto range = mesh.local_range();
  LocalMesh<double, 2> l_mesh(mesh.ghosted_range());
  mesh.copy_ghosted_to(l_mesh);

  for (auto p : range) {
    double x = p[0] * h;
    double y = p[1] * h;
    l_mesh[p] = solution(x, y);
  }

  mesh.copy_ghosted_from(l_mesh);
}

void initialize_boundary(Mesh& mesh, double h)
{
  auto global_range = mesh.range();
  auto range = mesh.ghosted_range();
  LocalMesh<double, 2> l_mesh(range);
  mesh.copy_ghosted_to(l_mesh);

  std::cout << global_range << std::endl;

  for (auto p : range) {
    if (p[0] == -1
        || p[1] == -1
        || p[0] == global_range.end_pos()[0]
        || p[1] == global_range.end_pos()[1])
    {
      double x = p[0] * h;
      double y = p[1] * h;
      l_mesh[p] = solution(x, y);
    } else {
      l_mesh[p] = 0.0;
    }
  }

  mesh.copy_ghosted_from(l_mesh);
}

void apply_laplace(Mesh& f, double h, const Mesh& u)
{
  std::array<double, 5> coeff
    = { -1.0 / (h*h), -1.0 / (h*h), 4.0 / (h*h), -1.0 / (h*h), -1.0 / (h*h) };

  u.apply_five_point(f, coeff);
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

void initialize_rhs(double h, Mesh& rhs)
{
  Mesh domain(rhs.range(), rhs.ghost_width());

  initialize_boundary(domain, h);

  // print_mesh(domain); std::cout << std::endl;

  apply_laplace(rhs, h, domain);

  rhs *= (-1.0);
}

int main(int argc, char** argv)
{
  Init init(argc, argv);
  const int N = 100;
  Range<2> range({0,0}, {N, N});
  double h = 1.0 / (N-1);

  auto comm = Comm::world();

  if (comm.rank() == 0)
    std::cout << "Poisson 2D solver" << std::endl;

  Mesh rhs(range, 1);
  Mesh solution(range, 1);

  CgSolver<Mesh> solver(1e-10);
  auto op
    = shared_copy(make_linear_operator<Mesh>(
        [h](Mesh& result, const Mesh& arg) {
          const_cast<Mesh&>(arg).fill_ghost_points(0.0);
          apply_laplace(result, h, arg);
        }));
  solver.setup(op);

  initialize_rhs(h, rhs);
  // print_mesh(rhs); std::cout << std::endl;
  set_zero(solution);
  solver.solve(solution, rhs);

  Mesh residual(range, 1);  // b - Ax
  Mesh tmp1(range, 1);
  op->apply(tmp1, solution);
  residual.assign(rhs);
  residual.add_scaled(-1.0, tmp1);
  std::cout << "residual, l2 norm: " << residual.l2_norm() << std::endl;

  // print_mesh(solution);
  // std::cout << std::endl;

  Mesh error(range, 1);
  initialize_exact_solution(error, h);
  // print_mesh(error);
  error.add_scaled(-1.0, solution);

  // std::cout << std::endl;
  // print_mesh(error);

  std::cout << "l2 error: " << error.l2_norm() << std::endl;

  return 0;
}

