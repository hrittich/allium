#include "vector.hpp"

namespace chive {

  VectorSpec::VectorSpec(MpiComm comm, global_size_t global_size, size_t local_size)
    : comm(comm), global_size(global_size), local_size(local_size)
  {}

}

