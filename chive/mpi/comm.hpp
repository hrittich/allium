#ifndef CHIVE_MPI_COMM_HPP
#define CHIVE_MPI_COMM_HPP

#include <mpi.h>
#include <vector>

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

      void barrier(void);

      std::vector<long long> sum_exscan(std::vector<long long> buf);

      MPI_Comm get_handle() { return handle; }
    private:
      MPI_Comm handle;
  };
}

#endif
