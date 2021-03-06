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

#ifndef ALLIUM_LA_VECTOR_SPEC_HPP
#define ALLIUM_LA_VECTOR_SPEC_HPP

#include <allium/util/types.hpp>
#include <allium/ipc/comm.hpp>

namespace allium {

  /**
   @brief The specification of a vector, needed to create distributed vectors.
   */
  class VectorSpec final {
    public:
      VectorSpec(const VectorSpec&) = default;
      VectorSpec(VectorSpec&&) = default;
      VectorSpec(Comm comm, size_t local_size, global_size_t global_size);

      VectorSpec& operator= (const VectorSpec&) = default;
      VectorSpec& operator= (VectorSpec&&) = default;

      bool operator!= (const VectorSpec& other) {
        return (m_comm != other.m_comm)
               || (m_global_size != other.m_global_size)
               || (m_local_size != other.m_local_size);
      }

      Comm comm() { return m_comm; }
      global_size_t global_size() { return m_global_size; }
      size_t local_size() { return m_local_size; }

      global_size_t local_start() { return m_local_start; }
      global_size_t local_end() { return m_local_end; }
    private:
      Comm m_comm;
      global_size_t m_global_size;
      size_t m_local_size;
      global_size_t m_local_start;
      global_size_t m_local_end;
  };
}

#endif
