#include "init.hpp"

#include <chive/config.hpp>
#include <chive/la/petsc_util.hpp>

namespace chive {
  #ifdef CHIVE_USE_PETSC
  using petsc::chkerr;
  #endif

  Init::Init(int& argc, char** &argv)
    : mpi(argc, argv)
  {
    #ifdef CHIVE_USE_PETSC
      PetscErrorCode ierr;
      ierr = PetscInitialize(&argc, &argv, nullptr, nullptr); chkerr(ierr);
    #endif
  }

  Init::~Init() {
    #ifdef CHIVE_USE_PETSC
      PetscErrorCode ierr;
      ierr = PetscFinalize(); chkerr(ierr);
    #endif
  }
}

