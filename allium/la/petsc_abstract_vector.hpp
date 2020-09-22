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
          m_dirty(false),
          m_entries(nullptr)
      {}

    public:
      PetscAbstractVectorStorage(const PetscObjectPtr<Vec>& vec)
        : VectorStorageTrait<PetscAbstractVectorStorage, N>(
            VectorSpec(petsc::object_comm(vec),
                       petsc::vec_local_size(vec),
                       petsc::vec_global_size(vec))),
          m_ptr(vec),
          m_dirty(false),
          m_entries(nullptr)
      {}

      PetscAbstractVectorStorage(const PetscAbstractVectorStorage&) = delete;
      PetscAbstractVectorStorage& operator= (const PetscAbstractVectorStorage&) = delete;

      using VectorStorageTrait<PetscAbstractVectorStorage, N>::operator+=;
      PetscAbstractVectorStorage& operator+= (const PetscAbstractVectorStorage& other) {
        allium_assert(!m_dirty && !other.m_dirty);

        PetscErrorCode ierr;
        ierr = VecAXPY(m_ptr, 1.0, other.m_ptr); petsc::chkerr(ierr);
        return *this;
      }

      PetscAbstractVectorStorage& operator*=(const Number& factor) override {
        allium_assert(!m_dirty);

        PetscErrorCode ierr;

        ierr = VecScale(m_ptr, factor); petsc::chkerr(ierr);
        return *this;
      }

      using VectorStorageTrait<PetscAbstractVectorStorage, N>::dot;
      Number dot(const PetscAbstractVectorStorage& other) const {
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

      PetscObjectPtr<Vec> m_ptr;
    private:
      #ifdef ALLIUM_DEBUG
        bool m_dirty;
      #endif
      PetscScalar* m_entries;

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
  };
}
#endif
#endif
