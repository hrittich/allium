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

}
