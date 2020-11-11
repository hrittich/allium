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

#ifndef ALLIUM_MESH_PETSC_MESH_SPEC_HPP
#define ALLIUM_MESH_PETSC_MESH_SPEC_HPP

#include <allium/config.hpp>
#ifdef ALLIUM_USE_PETSC

#include <allium/util/extern.hpp>
#include <allium/util/types.hpp>
#include <allium/ipc/comm.hpp>
#include <allium/la/petsc_object_ptr.hpp>
#include <allium/la/petsc_util.hpp>
#include <allium/la/vector_spec.hpp>

#include "range.hpp"

#include <petscdm.h>
#include <petscdmda.h>

namespace allium {

template <int D>
class PetscMeshSpec {};

template <>
class PetscMeshSpec<2> {
  public:
    PetscMeshSpec(Comm comm,
                  std::array<DMBoundaryType, 2> boundary_type,
                  DMDAStencilType stencil_type,
                  std::array<global_size_t, 2> global_size,
                  std::array<PetscInt, 2> ranks_per_dim,
                  int ndof,
                  int stencil_width)
      : m_spec(comm, 0, 0),
        m_ndof(ndof)
    {
      using namespace petsc;
      PetscErrorCode ierr;

      ierr = DMDACreate2d(
               comm.handle(), // comm
               boundary_type[0], boundary_type[1],
               stencil_type,
               global_size[0], global_size[1],
               ranks_per_dim[0], ranks_per_dim[1],
               ndof,
               stencil_width,
               nullptr, // PetscInt lx[],
               nullptr, // PetscInt ly[],
               m_dm.writable_ptr());
      chkerr(ierr);

      ierr = DMSetUp(m_dm); chkerr(ierr);

      DMDALocalInfo info;
      ierr = DMDAGetLocalInfo(m_dm, &info); chkerr(ierr);

      m_spec = VectorSpec(comm,
                          info.xm * info.ym * ndof,
                          info.mx * info.my * ndof);
    }

    PetscMeshSpec(const PetscMeshSpec&) = delete;
    PetscMeshSpec& operator= (const PetscMeshSpec&) = delete;

    PetscObjectPtr<DM> dm() { return m_dm; }
    VectorSpec vector_spec() const { return m_spec; }
    Comm comm() { return m_spec.comm(); }

    int ndof() { return m_ndof; }

    Range<2> range() const {
      using namespace petsc;
      PetscErrorCode ierr;

      DMDALocalInfo info;
      ierr = DMDAGetLocalInfo(m_dm, &info); chkerr(ierr);

      Point<int, 2> begin{0,0};
      Point<int, 2> end{info.mx, info.my};

      return Range<2>(begin, end);
    }

    Range<2> local_range() const {
      using namespace petsc;
      PetscErrorCode ierr;

      Point<PetscInt, 2> begin;
      Point<PetscInt, 2> extent;
      ierr = DMDAGetCorners(m_dm,
                            &begin[0], &begin[1], nullptr,
                            &extent[0], &extent[1], nullptr);

      chkerr(ierr);

      return Range<2>(begin, begin+extent);
    }

    Range<2> local_ghost_range() const {
      using namespace petsc;
      PetscErrorCode ierr;

      Point<PetscInt, 2> begin;
      Point<PetscInt, 2> extent;
      ierr = DMDAGetGhostCorners(m_dm,
                                 &begin[0], &begin[1], nullptr,
                                 &extent[0], &extent[1], nullptr);

      chkerr(ierr);

      return Range<2>(begin, begin+extent);
    }
  private:
    PetscObjectPtr<DM> m_dm;
    VectorSpec m_spec;
    int m_ndof;
};

#define ALLIUM_PETSC_MESH_SPEC_DECL(template, D) \
  template class PetscMeshSpec<D>;
ALLIUM_EXTERN_D(ALLIUM_PETSC_MESH_SPEC_DECL)

}

#endif
#endif
