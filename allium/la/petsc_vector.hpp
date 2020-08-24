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

#ifndef ALLIUM_LA_PETSC_VECTOR_HPP
#define ALLIUM_LA_PETSC_VECTOR_HPP

#include <allium/config.hpp>

#ifdef ALLIUM_USE_PETSC

#include "petsc_object_ptr.hpp"
#include <petscvec.h>
#include "petsc_util.hpp"
#include "vector.hpp"
#include <allium/util/except.hpp>
#include <allium/util/assert.hpp>
#include <allium/config.hpp>
#include <algorithm>

namespace allium {
  template <typename To, typename From> struct numerical_cast {
    To operator() (From x) {
      return static_cast<To>(x);
    }
  };
  template <typename T> struct numerical_cast<T, T> {
    T operator() (T x) { return x; }
  };
  template <> struct numerical_cast<double, std::complex<double>> {
    double operator() (std::complex<double> x) {
      allium_assert(x.imag() == 0);
      return x.real();
    }
  };
  template <> struct numerical_cast<float, std::complex<double>> {
    float operator() (std::complex<double> x) {
      return numerical_cast<double, std::complex<double>>()(x);
    }
  };

  template <typename T>
  inline T* convert_if_needed(PetscScalar* begin, PetscScalar* end) {
    T* converted = new T[end - begin];
    std::transform(begin, end, converted, numerical_cast<T, PetscScalar>());
    return converted;
  }
  template <>
  inline PetscScalar* convert_if_needed(PetscScalar* begin, PetscScalar* end) {
    return begin;
  }

  template <typename T>
  inline void copy_back_if_needed(T* begin, T* end, PetscScalar* original) {
    std::transform(begin, end, original, numerical_cast<PetscScalar, T>());
    delete [] begin;
  }
  template <>
  inline void copy_back_if_needed(PetscScalar* begin, PetscScalar* end, PetscScalar* original) {
  }

  template <typename N>
  class PetscVectorStorage final
      : public VectorStorageTrait<PetscVectorStorage<N>, N>
  {
    public:
      template <typename> friend class LocalSlice;

      using typename VectorStorageTrait<PetscVectorStorage, N>::Number;
      using typename VectorStorageTrait<PetscVectorStorage, N>::Real;
      using VectorStorageTrait<PetscVectorStorage, N>::spec;

      PetscVectorStorage(VectorSpec spec)
        : VectorStorageTrait<PetscVectorStorage, N>(spec),
          m_dirty(false),
          m_entries(nullptr)
      {
        PetscErrorCode ierr;

        ierr = VecCreateMPI(spec.comm().handle(),
                            spec.local_size(),
                            spec.global_size(),
                            m_ptr.writable_ptr()); petsc::chkerr(ierr);
      }

      PetscVectorStorage(const PetscObjectPtr<Vec>& vec)
        : VectorStorageTrait<PetscVectorStorage, N>(
            VectorSpec(petsc::object_comm(vec),
                       petsc::vec_local_size(vec),
                       petsc::vec_global_size(vec))),
          m_ptr(vec),
          m_dirty(false),
          m_entries(nullptr)
      {}

      PetscVectorStorage(const PetscVectorStorage&) = delete;
      PetscVectorStorage& operator= (const PetscVectorStorage&) = delete;

      using VectorStorageTrait<PetscVectorStorage, N>::operator+=;
      PetscVectorStorage& operator+= (const PetscVectorStorage& other) {
        allium_assert(!m_dirty && !other.m_dirty);

        PetscErrorCode ierr;
        ierr = VecAXPY(m_ptr, 1.0, other.m_ptr); petsc::chkerr(ierr);
        return *this;
      }

      PetscVectorStorage& operator*=(const Number& factor) override {
        allium_assert(!m_dirty);

        PetscErrorCode ierr;

        ierr = VecScale(m_ptr, factor); petsc::chkerr(ierr);
        return *this;
      }

      using VectorStorageTrait<PetscVectorStorage, N>::dot;
      Number dot(const PetscVectorStorage& other) const {
        allium_assert(!m_dirty && !other.m_dirty);

        PetscErrorCode ierr;
        PetscScalar result;
        ierr = VecDot(m_ptr, other.m_ptr, &result); petsc::chkerr(ierr);
        return numerical_cast<Number, PetscScalar>()(result);
      }

      Real l2_norm() const override {
        allium_assert(!m_dirty);

        PetscErrorCode ierr;
        PetscReal result;

        ierr = VecNorm(m_ptr, NORM_2, &result); petsc::chkerr(ierr);

        return result;
      }

      PetscObjectPtr<Vec> native() const { return m_ptr; }
    protected:
      Number* aquire_data_ptr() override
      {
        allium_assert(!m_dirty);
        #ifdef ALLIUM_DEBUG
          m_dirty = true;
        #endif
        PetscErrorCode ierr;

        ierr = VecGetArray(m_ptr, &m_entries); petsc::chkerr(ierr);
        return convert_if_needed<Number>(m_entries, m_entries+spec().local_size());
      }

      void release_data_ptr(Number* data) override
      {
        allium_assert(m_dirty);
        #ifdef ALLIUM_DEBUG
          m_dirty = false;
        #endif
        PetscErrorCode ierr;

        copy_back_if_needed<Number>(data, data+spec().local_size(), m_entries);

        ierr = VecRestoreArray(m_ptr, &m_entries); petsc::chkerr(ierr);
      }
    private:
      PetscObjectPtr<Vec> m_ptr;
      #ifdef ALLIUM_DEBUG
        bool m_dirty;
      #endif
      PetscScalar* m_entries;

      PetscVectorStorage* allocate_like() const& override {
        PetscErrorCode ierr;

        PetscObjectPtr<Vec> new_vec;
        ierr = VecDuplicate(m_ptr, new_vec.writable_ptr()); petsc::chkerr(ierr);

        return new PetscVectorStorage(new_vec);
      }

      Cloneable* clone() const& override {
        PetscErrorCode ierr;

        std::unique_ptr<PetscVectorStorage> new_storage(this->allocate_like());
        ierr = VecCopy(m_ptr, new_storage->m_ptr); petsc::chkerr(ierr);

        return new_storage.release();
      }
  };

  template <typename N>
  using PetscVector = VectorBase<PetscVectorStorage<N>>;
}

#endif
#endif
