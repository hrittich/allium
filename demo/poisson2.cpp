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
#include <memory>

using namespace allium;

void initialize_boundary(PetscLocalMesh<2>& mesh)
{
  auto range = mesh.mesh_spec()->ghost_range();
  PetscMeshValues<2> val(&mesh);
  for (auto p : range) {
    if (p[0] == -1
        || p[1] == -1
        || p[0] == range.end_pos()[0]
        || p[1] == range.end_pos()[1]) {
      val(p[0], p[1]) = 0.0;
    }
  }
}

void apply_laplace(double h, PetscLocalMesh<2>& u, PetscMesh<2>& f)
{
  auto range = u.mesh_spec()->range();

  PetscMeshValues<2> u_val(&u), f_val(&u);

  for (auto p : range) {
    f_val(p[0], p[1]) = 1 / (h*h) * (4 * u_val(p[0], p[1]),
                                     -u_val(p[0]-1, p[1]),
                                     -u_val(p[0],   p[1]-1),
                                     -u_val(p[0],   p[1]+1),
                                     -u_val(p[0]+1, p[1]));
  }
}

class LaplaceOperator : public AbstractLinearOperator<PetscScalar>
{
  public:
    Vector<PetscScalar> apply(const VectorStorage<PetscScalar>& x) const& override {
      const PetscMesh<2>* mesh = dynamic_cast<const PetscMesh<2>*>(&x);
      if (mesh == nullptr)
        throw std::runtime_error("Invalid Vector");

      auto result = allocate_like(*mesh);

      PetscLocalMesh<2> input(mesh->mesh_spec());
      input.assign(*mesh);

      // @todo: correct h
      apply_laplace(1.0, input, *result);

      return Vector<PetscScalar>(std::move(result));
    }
};

int main(int argc, char** argv)
{
  Init init(argc, argv);

  auto comm = Comm::world();

  if (comm.rank() == 0)
    std::cout << "Poisson 2D solver" << std::endl;

  auto spec = std::shared_ptr<PetscMeshSpec<2>>(
                new PetscMeshSpec<2>(
                  comm,
                  {DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED},
                  DMDA_STENCIL_STAR,
                  {100, 100}, // global size
                  {PETSC_DECIDE, PETSC_DECIDE}, // processors per dim
                  1, // ndof
                  1)); // stencil_width

  PetscMesh<2> mesh(spec);
  PetscLocalMesh<2> local(spec);

  initialize_boundary(local);

}

