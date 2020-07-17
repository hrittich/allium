#ifndef CHIVE_LA_PETSC_UTIL_HPP
#define CHIVE_LA_PETSC_UTIL_HPP

#include <chive/config.hpp>

#ifdef CHIVE_USE_PETSC
#include <petscsys.h>
#endif

namespace chive {
  namespace petsc {

    #ifdef CHIVE_USE_PETSC
    inline void chkerr(PetscErrorCode ierr) {
      if (PetscUnlikely(ierr)) {
        throw std::runtime_error("PETSc error");
      }
    }
    #endif

  }
}

#endif
