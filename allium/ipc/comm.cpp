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

#include "comm.hpp"

namespace allium {

  Comm::Comm(::MPI_Comm handle)
    : m_handle(handle)
  {}

  Comm Comm::world() {
    return Comm(MPI_COMM_WORLD);
  }

  int Comm::rank() const {
    int rank;
    MPI_Comm_rank(m_handle, &rank);
    return rank;
  }

  int Comm::size() const {
    int size;
    MPI_Comm_size(m_handle, &size);
    return size;
  }

  Comm Comm::dup() const {
    MPI_Comm new_handle;
    MPI_Comm_dup(m_handle, &new_handle);
    return Comm(new_handle);
  }

  Comm Comm::split(int color, int key) const {
    MPI_Comm new_handle;
    MPI_Comm_split(m_handle, color, key, &new_handle);
    return Comm(new_handle);
  }

  void Comm::free() {
    MPI_Comm_free(&m_handle);
    m_handle = MPI_COMM_NULL;
  }

  void Comm::barrier(void) {
    MPI_Barrier(m_handle);
  }

  std::vector<long long> Comm::sum_exscan(std::vector<long long> buf)
  {
    std::vector<long long> result(buf.size());
    MPI_Exscan(buf.data(), result.data(), buf.size(), MPI_LONG_LONG, MPI_SUM, m_handle);
    return result;
  }

}
