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

#ifndef ALLIUM_UTIL_MEMORY_HPP
#define ALLIUM_UTIL_MEMORY_HPP

#include <memory>

namespace allium {

  /**
   @brief Creates a heap-allocated, unique_ptr-stored copy of an object.
   */
  template <typename T>
    std::unique_ptr<typename std::remove_reference<T>::type>
    unique_copy(T&& v) {
      return std::make_unique<typename std::remove_reference<T>::type>(std::forward<T>(v));
    }

  /**
   @brief Creates a heap-allocated, shared_ptr-stored copy of an object.
   */
  template <typename T>
    std::shared_ptr<typename std::remove_reference<T>::type>
    shared_copy(T&& v) {
      return std::make_shared<typename std::remove_reference<T>::type>(std::forward<T>(v));
    }

}

#endif
