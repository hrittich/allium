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

#ifndef ALLIUM_LA_VECTOR_TRAIT_HPP
#define ALLIUM_LA_VECTOR_TRAIT_HPP

#include <allium/util/crtp.hpp>
#include <allium/util/numeric.hpp>

namespace allium {

  /**
   @brief Generates common vector operations automatically.

   If += and *= are implemented, all other vector operations are
   automatically generated.
  */
  template <typename Derived, typename N>
  class VectorTrait
    : public CrtpTrait<Derived> {
    public:
      using Number = N;
      using Real = real_part_t<Number>;

      template <typename Other>
      Derived& operator-= (const Other& other) {
        derived(this) += (-1.0 * other);
        return derived(this);
      }

      Derived& operator/= (Number divisor) {
        derived(this) *= (Number(1.0) / divisor);
        return derived(this);
      }

      template <typename Other>
      Derived operator+ (const Other& rhs) const {
        Derived result(derived(this));
        result += rhs;
        return result;
      }

      template <typename Other>
      Derived operator- (const Other& rhs) const {
        Derived result(derived(this));
        result -= rhs;
        return result;
      }

      Derived operator* (Number factor) const {
        Derived result(derived(this));
        result *= factor;
        return result;
      }

      Derived operator/ (Number divisor) const {
        Derived result(derived(this));
        result /= divisor;
        return result;
      }
  };

  template <typename T, typename R = void>
  using enable_if_vector
    = std::enable_if_t<
        std::is_base_of<
          VectorTrait<T, typename T::Number>, T>::value, R>;

  /** Scalar multiplication from the left. */
  template <typename T>
  enable_if_vector<T, T>
  operator* (typename T::Number factor, const T& rhs) {
    return rhs * factor;
  }

  //template <typename T>
  //enable_if_vector<T, T>
  //T operator* (typename T::Real factor, const T& rhs) {
  //  return rhs * factor;
  //}

}

#endif
