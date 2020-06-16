#include "init.hpp"

#include <mpi.h>
#include <stdexcept>

namespace chive {

  MpiInit* MpiInit::instance = nullptr;

  MpiInit::MpiInit(int& argc, char** &argv) {
    if (instance != nullptr) {
      throw std::runtime_error("MPI is already initialized.");
    }

    MPI_Init(&argc, &argv);
  }

  MpiInit::~MpiInit() {
    MPI_Finalize();

    instance = nullptr;
  }

}

