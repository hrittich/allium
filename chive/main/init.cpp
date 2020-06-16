#include "init.hpp"

#include <chive/la/petsc_util.hpp>

namespace chive {
  using petsc::chkerr;

  Init::Init(int& argc, char** &argv)
    : mpi(argc, argv)
  {
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &argv, nullptr, nullptr); chkerr(ierr);
  }

  Init::~Init() {
    PetscErrorCode ierr;
    ierr = PetscFinalize(); chkerr(ierr);
  }
}

