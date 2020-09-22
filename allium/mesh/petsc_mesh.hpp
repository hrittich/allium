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

#ifndef ALLIUM_MESH_PETSC_MESH_HPP
#define ALLIUM_MESH_PETSC_MESH_HPP

#include <allium/config.hpp>
#ifdef ALLIUM_USE_PETSC

#include <allium/ipc/comm.hpp>
#include <allium/la/petsc_abstract_vector.hpp>
#include "petsc_mesh_spec.hpp"
#include "range.hpp"

#include <petscdm.h>
#include <petscdmda.h>

namespace allium {

template <int D>
class IPetscMesh {
  public:
    virtual std::shared_ptr<PetscMeshSpec<D>> mesh_spec() const = 0;
    virtual PetscObjectPtr<Vec> petsc_vec() = 0;
    virtual Range<D> range() const = 0;
};

template <int D>
class PetscMesh {};

template <>
class PetscMesh<2>
  : public PetscAbstractVectorStorage<PetscScalar>,
    public IPetscMesh<2>
{
  public:
    using PetscAbstractVectorStorage<PetscScalar>::PetscAbstractVectorStorage;

    explicit PetscMesh(std::shared_ptr<PetscMeshSpec<2>> spec);
    PetscMesh(const PetscMesh&) = delete;
    PetscMesh& operator= (const PetscMesh&) = delete;

    PetscMesh(std::shared_ptr<PetscMeshSpec<2>> spec,
              PetscObjectPtr<Vec> ptr);

    std::shared_ptr<PetscMeshSpec<2>> mesh_spec() const override { return m_spec; }
    PetscObjectPtr<Vec> petsc_vec() override { return m_ptr; }
    Range<2> range() const override { return m_spec->range(); }
  private:
    std::shared_ptr<PetscMeshSpec<2>> m_spec;

    PetscMesh* allocate_like() const& override;
    PetscMesh* clone() const& override;
};

template <int D>
class PetscLocalMesh {};

template <>
class PetscLocalMesh<2> : public IPetscMesh<2>
{
  public:
    explicit PetscLocalMesh(std::shared_ptr<PetscMeshSpec<2>> spec)
      : m_spec(spec)
    {
      using namespace petsc;
      PetscErrorCode ierr;

      ierr = DMCreateLocalVector(spec->dm(), m_ptr.writable_ptr());
      chkerr(ierr);
    }

    PetscLocalMesh(const PetscLocalMesh&) = delete;
    PetscLocalMesh& operator= (const PetscLocalMesh&) = delete;

    void assign(const PetscMesh<2>& global_mesh)
    {
      using namespace petsc;
      PetscErrorCode ierr;

      ierr = DMGlobalToLocal(m_spec->dm(), global_mesh.native(), INSERT_VALUES, m_ptr);
      chkerr(ierr);
    }

    std::shared_ptr<PetscMeshSpec<2>> mesh_spec() const override { return m_spec; }
    PetscObjectPtr<Vec> petsc_vec() override { return m_ptr; }
    Range<2> range() const override { return m_spec->ghost_range(); }
  private:
    std::shared_ptr<PetscMeshSpec<2>> m_spec;
    PetscObjectPtr<Vec> m_ptr;
};

template <int D> struct PetscArrayType {
  using type = typename PetscArrayType<D-1>::type*;
};
template <> struct PetscArrayType<0> { using type = PetscScalar; };

template <int D>
class PetscMeshValues {};

template <>
class PetscMeshValues<2>
{
  public:
    explicit PetscMeshValues(IPetscMesh<2>* mesh);
    ~PetscMeshValues();

    PetscMeshValues(const PetscMeshValues&) = delete;
    PetscMeshValues& operator= (const PetscMeshValues&) = delete;

    PetscScalar& operator() (int i, int j, int dof = 0) {
      #ifdef ALLIUM_BOUND_CHECKS
      if (!m_range.in({i,j}))
        throw std::logic_error("Out of range access");
      #endif
      return m_values[j][i * m_ndof + dof];
    }

  private:
    typename PetscArrayType<2>::type m_values;
    IPetscMesh<2>* m_mesh;
    int m_ndof;
    #ifdef ALLIUM_BOUND_CHECKS
      Range<2> m_range;
    #endif
};

}

#endif
#endif
