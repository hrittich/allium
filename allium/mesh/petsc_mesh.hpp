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
#include "local_mesh.hpp"

#include <petscdm.h>
#include <petscdmda.h>

namespace allium {

template <int D>
class IPetscMesh {
  public:
    virtual std::shared_ptr<PetscMeshSpec<D>> mesh_spec() const = 0;
    virtual PetscObjectPtr<Vec> petsc_vec() = 0;
    virtual Range<D> local_range() const = 0;
};

template <typename N, int D>
class PetscMesh
  : public PetscAbstractVectorStorage<N>,
    public IPetscMesh<D>
{
  public:
    using PetscAbstractVectorStorage<N>::PetscAbstractVectorStorage;

    explicit PetscMesh(std::shared_ptr<PetscMeshSpec<D>> spec);
    PetscMesh(const PetscMesh&) = delete;
    PetscMesh& operator= (const PetscMesh&) = delete;

    PetscMesh(std::shared_ptr<PetscMeshSpec<D>> spec,
              PetscObjectPtr<Vec> ptr);

    std::shared_ptr<PetscMeshSpec<D>> mesh_spec() const override { return m_spec; }
    PetscObjectPtr<Vec> petsc_vec() override { return this->native(); }
    Range<D> local_range() const override { return m_spec->local_range(); }
  private:
    std::shared_ptr<PetscMeshSpec<D>> m_spec;

    PetscMesh* allocate_like() const& override;
    PetscMesh* clone() const& override;
};

template <typename N, int D>
class PetscLocalMesh : public IPetscMesh<D>
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

    void assign(const PetscMesh<N,D>& global_mesh)
    {
      using namespace petsc;
      PetscErrorCode ierr;

      ierr = DMGlobalToLocal(m_spec->dm(), global_mesh.native(), INSERT_VALUES, m_ptr);
      chkerr(ierr);
    }

    std::shared_ptr<PetscMeshSpec<D>> mesh_spec() const override { return m_spec; }
    PetscObjectPtr<Vec> petsc_vec() override { return m_ptr; }
    Range<D> local_range() const override { return m_spec->local_ghost_range(); }
  private:
    std::shared_ptr<PetscMeshSpec<D>> m_spec;
    PetscObjectPtr<Vec> m_ptr;
};

template <typename N, int D, bool is_mutable=true>
class PetscMeshValues;

/// @cond INTERNAL
template <int D> struct PetscArrayType {
  using type = typename PetscArrayType<D-1>::type*;
};
template <> struct PetscArrayType<0> { using type = PetscScalar; };

/// @endcond


template <bool is_mutable>
class PetscMeshValues<PetscScalar, 2, is_mutable>
{
  public:
    using Mesh
      = typename std::conditional<is_mutable, IPetscMesh<2>, const IPetscMesh<2>>::type;
    using Reference
      = typename std::conditional<is_mutable, PetscScalar, const PetscScalar>::type;

    explicit PetscMeshValues(Mesh& mesh)
      : m_mesh(&mesh)
    {
      using namespace petsc;

      PetscErrorCode ierr;
      ierr = DMDAVecGetArray(m_mesh->mesh_spec()->dm(),
                             const_cast<IPetscMesh<2>*>(m_mesh)->petsc_vec(),
                             &m_values);
      chkerr(ierr);

      m_ndof = mesh.mesh_spec()->ndof();

      #ifdef ALLIUM_BOUND_CHECKS
        m_range = mesh.local_range();
      #endif
    }
    ~PetscMeshValues() {
      using namespace petsc;

      if (m_mesh != nullptr) {
        PetscErrorCode ierr;
        ierr = DMDAVecRestoreArray(m_mesh->mesh_spec()->dm(),
                                   const_cast<IPetscMesh<2>*>(m_mesh)->petsc_vec(),
                                   &m_values);
        chkerr(ierr);
      }
    }

    PetscMeshValues(const PetscMeshValues&) = delete;
    PetscMeshValues& operator= (const PetscMeshValues&) = delete;

    PetscMeshValues(PetscMeshValues&& other) {
      m_values = other.m_values;
      m_mesh = other.m_mesh;
      m_ndof = other.m_ndof;
      #ifdef ALLIUM_BOUND_CHECKS
        m_range = other.m_range;
      #endif

      other.m_mesh = nullptr;
    }

    PetscScalar& operator() (int i, int j, int dof = 0) {
      #ifdef ALLIUM_BOUND_CHECKS
      if (!m_range.in({i,j}))
        throw std::logic_error("Out of range access");
      #endif
      return m_values[j][i * m_ndof + dof];
    }

    int dof_count() const { return m_ndof; }
    Range<2> local_range() const { return m_mesh->local_range(); }

  private:
    typename PetscArrayType<2>::type m_values;
    Mesh* m_mesh;
    int m_ndof;
    #ifdef ALLIUM_BOUND_CHECKS
      Range<2> m_range;
    #endif
};

template <typename N, int D, bool is_mutable>
class PetscMeshValues
{
  public:
    using Mesh = typename PetscMeshValues<PetscScalar, 2, is_mutable>::Mesh;

    PetscMeshValues(Mesh& mesh)
      : m_native_values(mesh),
        m_converted_values(
          Range<D+1>(mesh.local_range().begin_pos().joined(0),
                     mesh.local_range().end_pos().joined(mesh.mesh_spec()->ndof())))
    {
      for (auto p : mesh.local_range()) {
        for (int i_dof=0; i_dof < m_native_values.dof_count(); ++i_dof) {
          m_converted_values[p.joined(i_dof)]
            = narrow_number<N, PetscScalar>()(m_native_values(p[0], p[1], i_dof));
        }
      }
    }

    PetscMeshValues(PetscMeshValues&& other) = default;

    ~PetscMeshValues()
    {
      for (auto p : m_native_values.local_range()) {
        for (int i_dof=0; i_dof < m_native_values.dof_count(); ++i_dof) {
          m_native_values(p[0], p[1], i_dof) = m_converted_values[p.joined(i_dof)];
        }
      }
    }

    N& operator() (int i, int j, int dof = 0) {
      return m_converted_values[Point<int, D+1>({i, j, dof})];
    }

  private:
    PetscMeshValues<PetscScalar, 2, is_mutable> m_native_values;
    LocalMesh<N, D+1> m_converted_values;
};

template <typename N, int D>
PetscMeshValues<N, D, true> local_mesh(PetscMesh<N, D>& mesh) {
  return PetscMeshValues<N, D, true>(mesh);
}

template <typename N, int D>
PetscMeshValues<N, D, false> local_mesh(const PetscMesh<N, D>& mesh) {
  return PetscMeshValues<N, D, false>(mesh);
}

template <typename N, int D>
PetscMeshValues<N, D, true> local_mesh(PetscLocalMesh<N, D>& mesh) {
  return PetscMeshValues<N, D, true>(mesh);
}

template <typename N, int D>
PetscMeshValues<N, D, false> local_mesh(const PetscLocalMesh<N, D>& mesh) {
  return PetscMeshValues<N, D, false>(mesh);
}

#define ALLIUM_PETSC_MESH_DECL(extern, N) \
  extern template class PetscMesh<N, 2>; \
  extern template class PetscLocalMesh<N, 2>;
ALLIUM_EXTERN_N(ALLIUM_PETSC_MESH_DECL)

}

#endif
#endif
