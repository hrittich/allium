#include "petsc_vector.hpp"

#ifdef CHIVE_USE_PETSC

#include <petscsys.h>
#include "petsc_util.hpp"

namespace chive {

  using petsc::chkerr;

  PetscVectorStorage::PetscVectorStorage(VectorSpec spec)
    : VectorStorageBase(spec)
  {
    PetscErrorCode ierr;

    ierr = VecCreateMPI(spec.comm().get_handle(),
                        spec.local_size(),
                        spec.global_size(),
                        ptr.writable_ptr()); chkerr(ierr);

  }

  void PetscVectorStorage::add(const VectorStorage& rhs) {
    PetscErrorCode ierr;

    const PetscVectorStorage* petsc_rhs = dynamic_cast<const PetscVectorStorage*>(&rhs);
    if (petsc_rhs != nullptr) {
      ierr = VecAXPY(ptr, 1.0, petsc_rhs->ptr); chkerr(ierr);
    } else {
      throw std::logic_error("Not implemented");
    }
  }

  void PetscVectorStorage::scale(const Number& factor) {
    PetscErrorCode ierr;

    ierr = VecScale(ptr, factor); chkerr(ierr);
  }

  PetscVectorStorage::Number
    PetscVectorStorage::dot(const VectorStorage<Number>& rhs) {
      PetscErrorCode ierr;

      const PetscVectorStorage* petsc_rhs = dynamic_cast<const PetscVectorStorage*>(&rhs);
      if (petsc_rhs != nullptr) {
        PetscScalar result;
        ierr = VecDot(ptr, petsc_rhs->ptr, &result), chkerr(ierr);
        return result;
      } else {
        throw std::logic_error("Not implemented");
      }
    }

  PetscVectorStorage::Real PetscVectorStorage::l2_norm() const {
    PetscErrorCode ierr;
    Real result;

    ierr = VecNorm(ptr, NORM_2, &result); chkerr(ierr);

    return result;
  }

  PetscVectorStorage::Number* PetscVectorStorage::aquire_data_ptr() {
    PetscErrorCode ierr;

    Number *result;
    ierr = VecGetArray(ptr, &result); chkerr(ierr);
    return result;
  }

  void PetscVectorStorage::release_data_ptr(Number* data) {
    PetscErrorCode ierr;

    ierr = VecRestoreArray(ptr, &data); chkerr(ierr);
  }

}

#endif
