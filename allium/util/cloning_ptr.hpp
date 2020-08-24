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

#ifndef ALLIUM_UTIL_CLONING_PTR_HPP
#define ALLIUM_UTIL_CLONING_PTR_HPP

#include <memory>
#include "cloneable.hpp"

namespace allium {
  template <typename T>
  class CloningPtr {
    public:
      // Create
      CloningPtr() : m_ptr(nullptr) {}

      operator bool() const {
        return (m_ptr != nullptr);
      }

      CloningPtr(T* value)
        : m_ptr(value)
      {}

      CloningPtr(std::unique_ptr<T> ptr)
        : m_ptr(std::move(ptr))
      {}

      // Copy
      CloningPtr(const CloningPtr& other)
        : m_ptr(other.m_ptr ? clone(*other.m_ptr) : nullptr)
      {}

      template <typename R>
      CloningPtr(const CloningPtr<R>& other)
        : m_ptr(clone(*other.m_ptr))
      {}

      CloningPtr& operator= (const CloningPtr& other) {
        m_ptr = clone(*other.m_ptr);
        return *this;
      }

      // Move
      CloningPtr(CloningPtr&& other)
        : m_ptr(std::move(other.m_ptr))
      {}

      CloningPtr& operator= (CloningPtr&& other) {
        m_ptr = std::move(other.m_ptr);
        return *this;
      }

      // Methods
      T& operator* () const {
        return *m_ptr;
      }

      T* operator-> () const {
        return m_ptr.get();
      }

      T* get() const {
        return m_ptr.get();
      }

    public:
      std::unique_ptr<T> m_ptr;
  };
}

#endif
