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

#include "vector_spec.hpp"
#include <allium/util/warnings.hpp>

#include <iostream>

namespace allium {
  VectorSpec::VectorSpec(Comm comm, size_t local_size, global_size_t global_size)
    : m_comm(comm), m_global_size(global_size), m_local_size(local_size)
  {
    ALLIUM_NO_NARROWING_WARNING
    std::vector<long long> local_size_v = { local_size };
    ALLIUM_RESTORE_WARNING
    local_size_v = m_comm.sum_exscan(local_size_v);

    m_local_start = local_size_v[0];
    m_local_end = m_local_start + local_size;
  }


}

