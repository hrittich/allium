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

  auto Comm::recv(Range<2>& p, int src, int tag) -> RecvInfo {
    std::array<int, 4> buf;
    auto info = recv(buf.data(), buf.size(), src, tag);

    p = Range<2>({buf[0], buf[1]}, {buf[2], buf[3]});
    return info;
  }

  auto Comm::recv(int* data, int max_elements, int src, int tag) -> RecvInfo {
    MPI_Status stat;
    MPI_Recv(data, max_elements, MPI_INT, src, tag, m_handle, &stat);

    RecvInfo info;
    info.source = stat.MPI_SOURCE;
    info.tag = stat.MPI_TAG;
    MPI_Get_count(&stat, MPI_INT, &info.elements);

    return info;
  }

  void Comm::send(const Range<2>& p, int dest, int tag) {
    std::vector<int> buf = { p.begin_pos()[0], p.begin_pos()[1],
                             p.end_pos()[0], p.end_pos()[1] };
    send(buf.data(), buf.size(), dest, tag);
  }

  void Comm::send(int* data, int elements, int dest, int tag) {
    MPI_Send(data, elements, MPI_INT, dest, tag, m_handle);
  }

  std::vector<long long> Comm::sum_exscan(std::vector<long long> buf)
  {
    std::vector<long long> result(buf.size());
    MPI_Exscan(buf.data(), result.data(), buf.size(), MPI_LONG_LONG, MPI_SUM, m_handle);
    return result;
  }

}
