#ifndef CHIVE_MPI_COMM_HPP
#define CHIVE_MPI_COMM_HPP

#include <mpi.h>

namespace chive {
  class MpiComm {
    public:
      MpiComm(::MPI_Comm handle);

      bool operator!= (const MpiComm& other) {
        return handle != other.handle;
      }

      static MpiComm world();

      int get_rank();
      int get_size();

      MPI_Comm get_handle() { return handle; }
    private:
      MPI_Comm handle;
  };
}

#endif
