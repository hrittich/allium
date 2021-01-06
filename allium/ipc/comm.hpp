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

#ifndef ALLIUM_IPC_COMM_HPP
#define ALLIUM_IPC_COMM_HPP

#include <mpi.h>
#include <vector>
#include <allium/mesh/range.hpp>

namespace allium {
  /** The communicator class.
   This class just wraps an MPI communicator to make MPI easier to use.
  */
  class Comm {
    public:
      Comm(::MPI_Comm handle);

      bool operator!= (const Comm& other) {
        return m_handle != other.m_handle;
      }

      static Comm world();

      int rank() const;
      int size() const;

      // === Communicator Management =========================================

      Comm dup() const;
      Comm split(int color, int key) const;
      void free();

      // === IPC =============================================================
      void barrier(void);

      struct RecvInfo {
        int source;
        int tag;
        int elements;
      };

      RecvInfo recv(Range<2>& p, int src, int tag);
      RecvInfo recv(int* data, int max_elements, int src, int tag);

      void send(const Range<2>& p, int dest, int tag);
      void send(int* data, int elements, int dest, int tag);

      std::vector<long long> sum_exscan(std::vector<long long> buf);

      MPI_Comm handle() { return m_handle; }
    private:
      MPI_Comm m_handle;
  };
}

#endif
