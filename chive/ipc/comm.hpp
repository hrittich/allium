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

#ifndef CHIVE_IPC_COMM_HPP
#define CHIVE_IPC_COMM_HPP

#include <mpi.h>
#include <vector>

namespace chive {
  /** The communicator class. */
  class Comm {
    public:
      Comm(::MPI_Comm handle);

      bool operator!= (const Comm& other) {
        return m_handle != other.m_handle;
      }

      static Comm world();

      int rank();
      int size();

      void barrier(void);

      std::vector<long long> sum_exscan(std::vector<long long> buf);

      MPI_Comm handle() { return m_handle; }
    private:
      MPI_Comm m_handle;
  };
}

#endif
