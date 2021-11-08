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

#ifndef ALLIUM_LA_VECTOR_STORAGE_HPP
#define ALLIUM_LA_VECTOR_STORAGE_HPP

#include <allium/util/cloneable.hpp>
#include <allium/util/numeric.hpp>
#include <allium/util/crtp.hpp>
#include <allium/util/except.hpp>
#include <allium/util/extern.hpp>
#include "vector_spec.hpp"

namespace allium {
  /**
   @defgroup vector Vector
   @brief Generic vector interface.
  */

  /**
   @addtogroup vector
   @{
  */

  template <typename> class LocalSlice;

  /** @brief Abstract base class for any vector storage. */
  template <typename N>
  class VectorStorage : public Cloneable
  {
    public:
      template <typename> friend class LocalSlice;

      // type traits
      using Number = N;
      using Real = real_part_t<Number>;

      VectorStorage(VectorSpec spec) : m_spec(spec) {}
      virtual ~VectorStorage();

      // assignment
      virtual void assign(const VectorStorage& rhs) = 0;

      // Public virtual vector interface
      virtual VectorStorage& operator+=(const VectorStorage& rhs) = 0;
      virtual VectorStorage& operator*=(const Number& factor) = 0;

      virtual void add_scaled(Number factor, const VectorStorage& other) = 0;
      virtual Real l2_norm() const = 0;
      virtual Number dot(const VectorStorage& rhs) const = 0;
      virtual void fill(N value) = 0;

      VectorSpec spec() const { return m_spec; }
    private:
      // Private virtual interface
      virtual VectorStorage* allocate_like() const& = 0;

      template <typename T>
        std::enable_if_t<std::is_base_of<VectorStorage<typename T::Number>, T>::value,
                         std::unique_ptr<T>>
        friend allocate_like(const T& o);

      virtual Number* aquire_data_ptr() = 0;
      const Number* aquire_data_ptr() const {
        return const_cast<VectorStorage*>(this)->aquire_data_ptr();
      }

      virtual void release_data_ptr(Number* data) = 0;
      void release_data_ptr(const Number* data) const {
        const_cast<VectorStorage*>(this)->release_data_ptr(const_cast<Number*>(data));
      }

    private:
      VectorSpec m_spec;
  };

  template <typename C, typename T=void>
  using enable_if_vector_t =
    std::enable_if_t<std::is_base_of<VectorStorage<typename C::Number>, C>::value, T>;

  template <typename T>
    enable_if_vector_t<T, std::unique_ptr<T>>
    allocate_like(const T& o)
  {
    using Number = typename T::Number;

    VectorStorage<Number>* allocated =
      static_cast<const VectorStorage<Number>&>(o).allocate_like();;
    allium_assert(dynamic_cast<T*>(allocated) != nullptr,
                  std::string("The method allocate_like is not overridden in")
                  + typeid(*allocated).name()
                  + ".");

    return std::unique_ptr<T>(static_cast<T*>(allocated));
  }

  /**
   @brief Accessor for the locally stored part of a vector.
   */
  template <typename P>
  class LocalSlice final {
    public:
      using Pointer = P;
      using Vector = typename std::pointer_traits<P>::element_type;
      using Number = typename Vector::Number;
      using Real = typename Vector::Real;
      using DataPointer =
              typename std::conditional<std::is_const<Vector>::value,
                                        const Number*,
                                        Number*>::type;
      using Reference = std::conditional_t<std::is_const<Vector>::value,
                                           Number,
                                           Number&>;
      using Base = std::conditional_t<std::is_const<Vector>::value,
                                      const VectorStorage<Number>,
                                      VectorStorage<Number>>;

      explicit LocalSlice(const Pointer& ptr)
        : m_data(nullptr), m_size(0)
      {
        reset(ptr);
      }

      LocalSlice(const LocalSlice&) = delete;

      ~LocalSlice() {
        release();
      }

      LocalSlice(LocalSlice&& other)
        : m_data(other.m_data),
          m_size(other.m_size),
          m_ptr(other.m_ptr)
      {
        other.m_data = nullptr;
        other.m_size = 0;
        other.m_ptr = nullptr;
      }

      void operator= (const LocalSlice& rhs) {
        if (rhs.size() != size()) {
          throw std::logic_error("Invalid length of initilizer list.");
        }

        for (size_t i_element = 0; i_element < size(); ++i_element) {
          (*this)[i_element] = rhs[i_element];
        }
      }

      template <typename P2>
      void operator= (const LocalSlice<P2>& rhs) {
        if (rhs.size() != size()) {
          throw std::logic_error("Invalid length of initilizer list.");
        }

        for (size_t i_element = 0; i_element < size(); ++i_element) {
          (*this)[i_element] = rhs[i_element];
        }
      }

      void reset(const Pointer& ptr) {
        release();

        m_ptr = ptr;
        m_data = static_cast<Base*>(m_ptr)->aquire_data_ptr();
        m_size = m_ptr->spec().local_size();
      }

      void release() {
        if (m_data != nullptr) {
          static_cast<Base*>(m_ptr)->release_data_ptr(m_data);
          m_data = nullptr;
          m_size = 0;
        }
      }

      LocalSlice& operator= (const std::initializer_list<Number>& rhs) {
        if (rhs.size() != m_size) {
          throw std::logic_error("Invalid length of initilizer list.");
        }

        size_t i_element = 0;
        for (auto element : rhs) {
          (*this)[i_element] = element;
          ++i_element;
        }
        return *this;
      }

      Reference operator[] (size_t index) const {
        #ifdef ALLIUM_BOUND_CHECKS
          if (index >= m_size) {
            throw std::logic_error("Index out of bounds.");
          }
        #endif

        return m_data[index];
      }

      size_t size() const { return m_size; }
    protected:
      DataPointer m_data;
      size_t m_size;
    private:
      Pointer m_ptr;
  };

  template <typename N>
    LocalSlice<VectorStorage<N>*> local_slice(VectorStorage<N>& storage)
  {
    return LocalSlice<VectorStorage<N>*>(&storage);
  }

  template <typename N>
    LocalSlice<const VectorStorage<N>*> local_slice(const VectorStorage<N>& storage)
  {
    return LocalSlice<const VectorStorage<N>*>(&storage);
  }

  template <typename T1, typename T2>
  enable_if_vector_t<T1>
  copy(T1& dest, T2& src) {
    local_slice(dest) = local_slice(src);
  }

  /**
   @brief Implements a set of common vector operations.
   */
  template <typename Derived, typename N>
  class VectorStorageTrait
    : public VectorStorage<N>,
      public CrtpTrait<Derived>
  {
    public:
      using typename VectorStorage<N>::Number;
      using typename VectorStorage<N>::Real;
      using VectorStorage<N>::spec;

      VectorStorageTrait(VectorSpec spec) : VectorStorage<N>(spec) {}
      VectorStorageTrait(const VectorStorageTrait&) = delete;
      VectorStorageTrait& operator= (const VectorStorageTrait&) = delete;

      void assign(const VectorStorage<N>& other) {
        copy(*this, other);
      }

      // @todo: Turn these calls into static casts and add assertions
      VectorStorageTrait& operator+=(const VectorStorage<N>& rhs) override {
        const Derived* derived_rhs = dynamic_cast<const Derived*>(&rhs);
        if (derived_rhs != nullptr) {
          return derived(this) += (*derived_rhs);
        } else {
          throw not_implemented();
        }
      }

      N dot(const VectorStorage<N>& rhs) const override {
        const Derived* derived_rhs = dynamic_cast<const Derived*>(&rhs);
        if (derived_rhs != nullptr) {
          return derived(this).dot(*derived_rhs);
        } else {
          throw not_implemented();
        }
      }

      void add_scaled(Number factor, const VectorStorage<N>& other) override {
        allium_assert(dynamic_cast<const Derived*>(&other));
        derived(this).add_scaled(factor, static_cast<const Derived&>(other));
      }

  };

  template <typename V>
  enable_if_vector_t<V> fill(V& vec, typename V::Number value)
  {
    vec.fill(value);
  }

  template <typename N>
  void set_zero(VectorStorage<N>& vec)
  {
    fill(vec, N(0.0));
  }

  template <typename V>
  std::unique_ptr<V> zeros_like(const V& vec)
  {
    auto v = allocate_like(vec);
    set_zero(*v);
    return v;
  }

  /// @}

  #define ALLIUM_LA_VECTOR_STORAGE_DECL(extern, N) \
    extern template class VectorStorage<N>;
  ALLIUM_EXTERN_N(ALLIUM_LA_VECTOR_STORAGE_DECL)
}

#endif
