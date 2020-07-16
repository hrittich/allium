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
