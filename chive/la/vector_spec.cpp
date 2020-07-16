#include "vector_spec.hpp"

#include <iostream>

namespace chive {
  VectorSpec::VectorSpec(MpiComm comm, size_t local_size, global_size_t global_size)
    : m_comm(comm), m_global_size(global_size), m_local_size(local_size)
  {
    std::vector<long long> local_size_v = { local_size };
    local_size_v = m_comm.sum_exscan(local_size_v);

    m_local_start = local_size_v[0];
    m_local_end = m_local_start + local_size;
  }


}

