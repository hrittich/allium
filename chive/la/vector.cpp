#include "vector.hpp"

namespace chive {

  VectorSpec::VectorSpec(MpiComm comm, global_size_t global_size, size_t local_size)
    : m_comm(comm), m_global_size(global_size), m_local_size(local_size)
  {}

}

