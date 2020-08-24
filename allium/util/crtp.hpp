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

#ifndef ALLIUM_UTIL_CRTP_HPP
#define ALLIUM_UTIL_CRTP_HPP

#include <memory>

namespace allium {

  template <typename T>
  typename T::DerivedType& derived(T* p) {
    return static_cast<typename T::DerivedType&>(*p);
  }
  template <typename T>
  const typename T::DerivedType& derived(const T* p) {
    return static_cast<const typename T::DerivedType&>(*p);
  }

  /** Trait that implements the self helper routine for the curiously
   * recurring template pattern. */
  template <typename Derived>
  struct CrtpTrait {
      using DerivedType = Derived;
  };
}

#endif
