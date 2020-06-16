#ifndef CHIVE_LA_PETSC_UTIL_HPP
#define CHIVE_LA_PETSC_UTIL_HPP

#include <petscsys.h>

namespace chive {
  namespace petsc {
    inline void chkerr(PetscErrorCode ierr) {
      if (PetscUnlikely(ierr)) {
        throw std::runtime_error("PETSc error");
      }
    }
  }
}

#endif
