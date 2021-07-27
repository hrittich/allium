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
#include <allium/mesh/vtk_io.hpp>
#include <allium/ipc/comm.hpp>
#include <allium/util/memory.hpp>
#include <allium/ode/imex_euler.hpp>
#include <allium/la/gmres.hpp>
#include <sstream>
#include <iomanip>

using namespace allium;

using Number = double;
using Real = real_part_t<Number>;
using Mesh = PetscMesh<double, 2>;
using LocalMesh = PetscLocalMesh<double, 2>;

const double pi = 4.0 * atan(1.0);

/** Simulation of the heat equation. */
class Heat {
  public:
    Heat(int N);

    void simulate();

  private:
    const global_size_t m_N;
    const double m_h;

    static double exact_solution(double t, double x, double y);
    void set_solution(Mesh& result, double t);

    void apply_shifted_laplace(Mesh& f, Number a, const Mesh& u);

    void solve_f_impl(Mesh& y,
                      Real t,
                      Number a,
                      const Mesh& r,
                      InitialGuess initial_guess);

    static void f_expl(Mesh& result, Real t, const Mesh& u);
};

int main(int argc, char** argv)
{
  using namespace std::placeholders;

  Init init(argc, argv);  // initialize Allium

  Heat problem(64);
  problem.simulate();

  return 0;
}

Heat::Heat(int N)
  : m_N(N),
    m_h(1.0 / N) // in the periodic case h is 1.0 / N
{}

void Heat::simulate() {
  using namespace std::placeholders;

  auto comm = Comm::world();

  if (comm.rank() == 0)
    std::cout << "Periodic heat 2D solver" << std::endl;

  // Create a mesh. For description of the parameters, see the PETSc manual
  auto spec = std::shared_ptr<PetscMeshSpec<2>>(
                new PetscMeshSpec<2>(
                  comm,
                  {DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC},
                  DMDA_STENCIL_STAR,
                  {m_N, m_N}, // global size
                  {PETSC_DECIDE, PETSC_DECIDE}, // processors per dim
                  1, // ndof
                  1)); // stencil_width

  Mesh u(spec);
  Mesh error(spec);

  // setup the integrator
  ImexEuler<Mesh> integrator;
  integrator.setup(&f_expl,
                   [](Mesh& out, double t, const Mesh& in) {}, // not needed for Euler
                   std::bind(&Heat::solve_f_impl, this, _1, _2, _3, _4, _5));

  double t0 = 0;
  set_solution(u, t0);
  integrator.initial_values(t0, u);
  integrator.dt(1e-4);

  auto filename = [](int frame) {
    std::stringstream s;
    s << "mesh_" << frame << ".pvti";
    return s.str();
  };

  write_vtk(filename(0), u);

  for (int i = 0; i < 200; ++i) {
    double t0 = i * 2e-4;
    double t1 = (i+1)*2e-4;
    integrator.initial_values(t0, u);
    integrator.integrate(u, t1);

    write_vtk(filename(i+1), u);

    // error = exact - u
    set_solution(error, t1);
    error.add_scaled(-1.0, u);

    auto e = error.l2_norm();
    if (comm.rank() == 0) {
      std::cout << "t = " << t1 << ", ‖e‖ = " << e << std::endl;
    }
  } 
}

/**
  The exact analytic solution of the problem.
  The solution for this particular case is known, hence, we can check the
  correctness of our code.
*/
double Heat::exact_solution(double t, double x, double y)
{
  return exp(-8*pi*pi*t) * sin(x*pi*2) * sin(y*pi*2);
}

/** Set the mesh values to the exact solution. */
void Heat::set_solution(Mesh& result, double t)
{
  auto range = result.mesh_spec()->local_range();

  auto lresult = local_mesh(result);
  for (auto p : range) {
    double x = m_h * p[0];
    double y = m_h * p[1];
    lresult(p[0], p[1]) = exact_solution(t, x, y);
  }
}

/**
 Compute `f = (-Δ + a I) u`.
 */
void Heat::apply_shifted_laplace(Mesh& f, Number a, const Mesh& u)
{
  // PETSc require to create a "local mesh" to have access to the ghost
  // nodes.
  ::LocalMesh u_aux(u.mesh_spec());
  u_aux.assign(u); // copy to determine the ghost nodes

  auto range = u.mesh_spec()->local_range();
  auto lu = local_mesh(u_aux);
  auto lf = local_mesh(f);

  // Apply the stencil
  //         |     -1     |
  // (1/h^2) | -1   4  -1 |
  //         |     -1     |h
  for (auto p : range) {
    lf(p[0], p[1])
      = (1.0 / (m_h*m_h))
        * ( (4+a*(m_h*m_h)) * lu(p[0], p[1])
            - lu(p[0]-1, p[1])
            - lu(p[0],   p[1]-1)
            - lu(p[0],   p[1]+1)
            - lu(p[0]+1, p[1]));
  }
}

/**
 Solves y - a f_i(t, y) = r, where f_i(t, y) = Δy.
*/
void Heat::solve_f_impl(Mesh& y,
                        Real t,
                        Number a,
                        const Mesh& r,
                        InitialGuess initial_guess) {
  using namespace std::placeholders;

  // rhs = (1/a) r + Δ^b u^b
  Mesh rhs(r.mesh_spec());
  rhs.assign(r);
  rhs *= (1.0/a);

  // solve (-Δ + (1/a) I) y = (1/a) r + Δ^b y^b
  CgSolver<Mesh> solver;
  auto op = std::bind(&Heat::apply_shifted_laplace, this, _1, 1.0/a, _2);
  solver.setup(shared_copy(make_linear_operator<Mesh>(op)));
  solver.solve(y, rhs, initial_guess);
}

/** The explicit part of the ODE, f_e(y) = 0 */
void Heat::f_expl(Mesh& result, Real t, const Mesh& u)
{
  auto range = u.mesh_spec()->local_range();
  auto lresult = local_mesh(result);

  for (auto p : range) {
    lresult(p[0], p[1]) = 0.0;
  }
}

