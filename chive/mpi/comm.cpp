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

namespace chive {

  MpiComm::MpiComm(::MPI_Comm handle)
    : handle(handle)
  {}

  MpiComm MpiComm::world() {
    return MpiComm(MPI_COMM_WORLD);
  }

  int MpiComm::get_rank() {
    int rank;
    MPI_Comm_rank(handle, &rank);
    return rank;
  }

  int MpiComm::get_size() {
    int size;
    MPI_Comm_size(handle, &size);
    return size;
  }

  void MpiComm::barrier(void) {
    MPI_Barrier(handle);
  }

  std::vector<long long> MpiComm::sum_exscan(std::vector<long long> buf)
  {
    std::vector<long long> result(buf.size());
    MPI_Exscan(buf.data(), result.data(), buf.size(), MPI_LONG_LONG, MPI_SUM, handle);
    return result;
  }

}
