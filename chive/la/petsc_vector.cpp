#include "petsc_vector.hpp"

#include <petscsys.h>
#include "petsc_util.hpp"

namespace chive {

  using petsc::chkerr;

  PetscVector::PetscVector(VectorSpec spec)
    : spec(spec)
  {
    PetscErrorCode ierr;

    ierr = VecCreateMPI(spec.get_comm().get_handle(),
                        spec.get_local_size(),
                        spec.get_global_size(),
                        ptr.writable_ptr()); chkerr(ierr);

  }

  void PetscVector::add(const Vector& rhs) {
    PetscErrorCode ierr;

    const PetscVector* petsc_rhs = dynamic_cast<const PetscVector*>(&rhs);
    if (petsc_rhs != nullptr) {
      ierr = VecAXPY(ptr, 1.0, petsc_rhs->ptr); chkerr(ierr);
    } else {
      throw std::logic_error("Not implemented");
    }
  }

  void PetscVector::scale(const Number& factor) {
    PetscErrorCode ierr;

    ierr = VecScale(ptr, factor); chkerr(ierr);
  }

  PetscVector::Real PetscVector::l2_norm() const {
    PetscErrorCode ierr;
    Real result;

    ierr = VecNorm(ptr, NORM_2, &result); chkerr(ierr);

    return result;
  }

  std::unique_ptr<VectorSlice<PetscVector::Number>> PetscVector::local_slice() {
    auto slice = std::make_unique<PetscVectorSlice>(shared_from_this());
    return slice;
  }

  //=== PetscVectorSlice =====================================================

  PetscVectorSlice::PetscVectorSlice(std::shared_ptr<PetscVector> vec)
    : vec(vec)
  {
    PetscErrorCode ierr;

    ierr = VecGetArray(vec->ptr, &data); chkerr(ierr);
    size = vec->spec.get_local_size();
  }

  PetscVectorSlice::~PetscVectorSlice()
  {
    PetscErrorCode ierr;

    ierr = VecRestoreArray(vec->ptr, &data); chkerr(ierr);
  }

}

