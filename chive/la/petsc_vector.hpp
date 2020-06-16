#ifndef CHIVE_LA_PETSC_VECTOR_HPP
#define CHIVE_LA_PETSC_VECTOR_HPP

#include "petsc_object_ptr.hpp"
#include <petscvec.h>
#include "vector.hpp"

namespace chive {

  class PetscVector
      : public Vector<PetscScalar>,
        public std::enable_shared_from_this<PetscVector>
  {
    public:
      PetscVector(VectorSpec spec);

      void add(const Vector& rhs) override;
      void scale(const Number& factor) override;
      Real l2_norm() const override;

      std::unique_ptr<VectorSlice<Number>> local_slice() override;

      friend class PetscVectorSlice;
    private:
      VectorSpec spec;
      PetscObjectPtr<Vec> ptr;
  };

  class PetscVectorSlice : public VectorSlice<PetscScalar> {
    public:
      PetscVectorSlice(std::shared_ptr<PetscVector> vec);
      ~PetscVectorSlice();

    private:
      std::shared_ptr<PetscVector> vec;
  };

}

#endif
