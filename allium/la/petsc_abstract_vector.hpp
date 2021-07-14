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

#ifndef ALLIUM_LA_PETSC_ABSTRACT_VECTOR_HPP
#define ALLIUM_LA_PETSC_ABSTRACT_VECTOR_HPP

#include <allium/config.hpp>
namespace allium {}
#ifdef ALLIUM_USE_PETSC

#include "petsc_object_ptr.hpp"
#include <petscvec.h>
#include "petsc_util.hpp"
#include "vector_storage.hpp"
#include <allium/util/except.hpp>
#include <allium/util/assert.hpp>
#include <allium/config.hpp>
#include <algorithm>

namespace allium {

  /**
   @brief A vector based on a PETSc vector.
   */
  template <typename N> class PetscAbstractVectorStorage;

  template <>
  class PetscAbstractVectorStorage<PetscScalar>
      : public VectorStorageTrait<PetscAbstractVectorStorage<PetscScalar>, PetscScalar>
  {
    public:
      template <typename> friend class LocalSlice;
      template <typename N> friend class PetscAbstractVectorStorage;

      using typename VectorStorageTrait<PetscAbstractVectorStorage, PetscScalar>::Number;
      using typename VectorStorageTrait<PetscAbstractVectorStorage, PetscScalar>::Real;
      using VectorStorageTrait<PetscAbstractVectorStorage, PetscScalar>::spec;

    protected:
      PetscAbstractVectorStorage(VectorSpec spec)
        : VectorStorageTrait<PetscAbstractVectorStorage, PetscScalar>(spec)
        #ifdef ALLIUM_DEBUG
          , m_dirty(false)
        #endif
      {}

    public:
      PetscAbstractVectorStorage(const PetscObjectPtr<Vec>& vec)
        : VectorStorageTrait<PetscAbstractVectorStorage, PetscScalar>(
            VectorSpec(petsc::object_comm(vec),
                       petsc::vec_local_size(vec),
                       petsc::vec_global_size(vec))),
          m_ptr(vec)
        #ifdef ALLIUM_DEBUG
          , m_dirty(false)
        #endif
      {}

      PetscAbstractVectorStorage(const PetscAbstractVectorStorage&) = delete;
      PetscAbstractVectorStorage& operator= (const PetscAbstractVectorStorage&) = delete;

      using VectorStorageTrait<PetscAbstractVectorStorage, PetscScalar>::operator+=;
      PetscAbstractVectorStorage& operator+= (const PetscAbstractVectorStorage& other) {
        allium_assert(!m_dirty && !other.m_dirty);

        PetscErrorCode ierr;
        ierr = VecAXPY(m_ptr, 1.0, other.m_ptr); petsc::chkerr(ierr);
        return *this;
      }

      void add_scaled(Number factor, const PetscAbstractVectorStorage& other) {
        allium_assert(!m_dirty && !other.m_dirty);

        PetscErrorCode ierr;
        ierr = VecAXPY(m_ptr, factor, other.m_ptr); petsc::chkerr(ierr);
      }

      PetscAbstractVectorStorage& operator*=(const Number& factor) override {
        allium_assert(!m_dirty);

        PetscErrorCode ierr;

        ierr = VecScale(m_ptr, factor); petsc::chkerr(ierr);
        return *this;
      }

      using VectorStorageTrait<PetscAbstractVectorStorage, PetscScalar>::dot;
      PetscScalar dot(const PetscAbstractVectorStorage& other) const {
        allium_assert(!m_dirty && !other.m_dirty);

        PetscErrorCode ierr;
        PetscScalar result;
        ierr = VecDot(m_ptr, other.m_ptr, &result); petsc::chkerr(ierr);
        return result;
      }

      Real l2_norm() const override {
        allium_assert(!m_dirty);

        PetscErrorCode ierr;
        PetscReal result;

        ierr = VecNorm(m_ptr, NORM_2, &result); petsc::chkerr(ierr);

        return result;
      }

      void native(PetscObjectPtr<Vec> ptr) { m_ptr = ptr; }
      PetscObjectPtr<Vec> native() const { return m_ptr; }
    protected:

      Number* aquire_data_ptr() override
      {
        allium_assert(!m_dirty);
        #ifdef ALLIUM_DEBUG
          m_dirty = true;
        #endif
        PetscErrorCode ierr;

        PetscScalar* data = nullptr;
        ierr = VecGetArray(m_ptr, &data); petsc::chkerr(ierr);
        return data;
      }

      void release_data_ptr(Number* data) override
      {
        allium_assert(m_dirty);
        #ifdef ALLIUM_DEBUG
          m_dirty = false;
        #endif
        PetscErrorCode ierr;

        ierr = VecRestoreArray(m_ptr, &data); petsc::chkerr(ierr);
      }

      PetscObjectPtr<Vec> m_ptr;
    private:
      #ifdef ALLIUM_DEBUG
        bool m_dirty;
      #endif

    protected:
      PetscObjectPtr<Vec> petsc_allocate_like() const& {
        PetscErrorCode ierr;

        PetscObjectPtr<Vec> new_vec;
        ierr = VecDuplicate(m_ptr, new_vec.writable_ptr()); petsc::chkerr(ierr);

        return new_vec;
      }

      PetscObjectPtr<Vec> petsc_clone() const& {
        PetscErrorCode ierr;

        PetscObjectPtr<Vec> new_vec = petsc_allocate_like();
        ierr = VecCopy(m_ptr, new_vec); petsc::chkerr(ierr);

        return new_vec;
      }

    private:
      PetscAbstractVectorStorage* allocate_like() const& override {
        return new PetscAbstractVectorStorage(this->petsc_allocate_like());
      }

      PetscAbstractVectorStorage* clone() const& override {
        return new PetscAbstractVectorStorage(this->petsc_clone());
      }
  };


  template <typename N>
  class PetscAbstractVectorStorage
      : public VectorStorageTrait<PetscAbstractVectorStorage<N>, N>
  {
    public:
      template <typename> friend class LocalSlice;

      using typename VectorStorageTrait<PetscAbstractVectorStorage, N>::Number;
      using typename VectorStorageTrait<PetscAbstractVectorStorage, N>::Real;
      using VectorStorageTrait<PetscAbstractVectorStorage, N>::spec;

    protected:
      PetscAbstractVectorStorage(VectorSpec spec)
        : VectorStorageTrait<PetscAbstractVectorStorage, N>(spec),
          m_native(spec),
          m_entries(nullptr)
      {}

    public:
      PetscAbstractVectorStorage(const PetscObjectPtr<Vec>& vec)
        : VectorStorageTrait<PetscAbstractVectorStorage, N>(
            VectorSpec(petsc::object_comm(vec),
                       petsc::vec_local_size(vec),
                       petsc::vec_global_size(vec))),
          m_native(vec),
          m_entries(nullptr)
      {}

      PetscAbstractVectorStorage(const PetscAbstractVectorStorage&) = delete;
      PetscAbstractVectorStorage& operator= (const PetscAbstractVectorStorage&) = delete;

      using VectorStorageTrait<PetscAbstractVectorStorage, N>::operator+=;
      PetscAbstractVectorStorage& operator+= (const PetscAbstractVectorStorage& other) {
        m_native += other.m_native;
        return *this;
      }

      void add_scaled(Number factor, const PetscAbstractVectorStorage& other) {
        m_native.add_scaled(factor, other.m_native);
      }

      PetscAbstractVectorStorage& operator*=(const Number& factor) override {
        m_native *= factor;
        return *this;
      }

      using VectorStorageTrait<PetscAbstractVectorStorage, N>::dot;
      Number dot(const PetscAbstractVectorStorage& other) const {
        return narrow_number<N, PetscScalar>()(m_native.dot(other.m_native));
      }

      Real l2_norm() const override {
        return narrow_number<Real, PetscReal>()(m_native.l2_norm());
      }

      void native(PetscObjectPtr<Vec> v) { m_native.native(v); }
      PetscObjectPtr<Vec> native() const { return m_native.native(); }

      PetscAbstractVectorStorage<PetscScalar>& native_scalar() { return m_native; }
      const PetscAbstractVectorStorage<PetscScalar>& native_scalar() const { return m_native; }
    protected:

      Number* aquire_data_ptr() override
      {
        m_entries = m_native.aquire_data_ptr();

        // convert to desired type
        N* converted = new N[spec().local_size()];
        std::transform(m_entries, m_entries+spec().local_size(),
                       converted,
                       narrow_number<N, PetscScalar>());

        return converted;
      }

      void release_data_ptr(Number* data) override
      {
        // convert to PETSc (we should not need a cast, because we want that
        // the conversion is not narrowing)
        std::copy(data, data+spec().local_size(), m_entries);
        m_native.release_data_ptr(m_entries);
        m_entries = nullptr;

        delete[] data; // delete the buffer for the converted data
      }

    private:
      PetscAbstractVectorStorage<PetscScalar> m_native;
      PetscScalar *m_entries;

    protected:
      PetscObjectPtr<Vec> petsc_allocate_like() const& {
        return m_native.petsc_allocate_like();
      }

      PetscObjectPtr<Vec> petsc_clone() const& {
        return m_native.petsc_clone();
      }
  };
}
#endif
#endif
