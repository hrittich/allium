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

#ifndef ALLIUM_LA_LINEAR_OPERATOR_HPP
#define ALLIUM_LA_LINEAR_OPERATOR_HPP

#include "vector.hpp"
#include <memory>
#include <allium/util/cloning_ptr.hpp>

namespace allium {
  template <typename N>
  class AbstractLinearOperator {
    public:
      virtual Vector<N> apply(const VectorStorage<N>& x) const& = 0;
  };

  template <typename N, bool reference = false>
  class LinearOperator {
    public:
      using Number = N;
      using Real = real_part_t<Number>;

      // Create
      template <typename T>
      LinearOperator(T storage)
        : m_ptr(std::forward<T>(storage))
      {}

      // Methods
      Vector<N> operator() (const Vector<N>& x) {
        return m_ptr->apply(x);
      }

      Vector<N> operator* (const Vector<N>& x) {
        return m_ptr->apply(x);
      }

    private:
      std::shared_ptr<AbstractLinearOperator<N>> m_ptr;
  };

  template <typename N>
  using LinearOperatorReference = LinearOperator<N, true>;
}

#endif
