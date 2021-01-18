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

#ifndef ALLIUM_LA_LOCAL_VECTOR_HPP
#define ALLIUM_LA_LOCAL_VECTOR_HPP

#include <Eigen/Core>
#include "vector_trait.hpp"
#include <allium/util/extern.hpp>

namespace allium {

  /**
    A (mathematical) vector stored locally on one processor.
  */
  template <typename N>
  class LocalVector
    : public VectorTrait<LocalVector<N>, N>
  {
    public:
      using Number = N;
      using Real = real_part_t<Number>;

      explicit LocalVector(size_t nrows)
        : m_storage(nrows)
      {}

      explicit LocalVector(std::initializer_list<N> entries)
        : m_storage(entries.size())
      {
        size_t i_entry = 0;
        for (auto entry : entries) {
          m_storage(i_entry) = entry;
          ++i_entry;
        }
      }

      LocalVector& operator+= (const LocalVector& rhs);
      LocalVector& operator*= (Number rhs);

      Number dot(const LocalVector& rhs) const {
        return rhs.m_storage.dot(m_storage);
      }

      Number& operator[] (size_t i_element) {
        return m_storage(i_element);
      }
      Number operator[] (size_t i_element) const {
        return (*const_cast<LocalVector*>(this))[i_element];
      }

      size_t nrows() const { return m_storage.rows(); }
    private:
      Eigen::Matrix<N, Eigen::Dynamic, 1> m_storage;
  };

  template <typename N>
  LocalVector<N>& LocalVector<N>::operator+= (const LocalVector& rhs)
  {
    m_storage += rhs.m_storage;
    return *this;
  }

  template <typename N>
  LocalVector<N>& LocalVector<N>::operator*= (Number rhs)
  {
    m_storage *= rhs;
    return *this;
  }

  template <typename N>
  LocalVector<N> operator* (N s, const LocalVector<N>& v) {
    LocalVector<N> w = v;
    w *= s;
    return w;
  }

  template <typename N>
  std::ostream& operator<< (std::ostream& os, const LocalVector<N>& v)
  {
    bool first = true;
    for (size_t i = 0; i < v.nrows(); ++i) {
      if (first) first = false;
      else os << " ";

      os << v[i];
    }

    return os;
  }

  #define ALLIUM_LOCAL_VECTOR_DECL(extern, N) \
    extern template class LocalVector<N>;
  ALLIUM_EXTERN_N(ALLIUM_LOCAL_VECTOR_DECL)
}

#endif
