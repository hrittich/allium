#ifndef CHIVE_LA_PETSC_VECTOR_HPP
#define CHIVE_LA_PETSC_VECTOR_HPP

#include "petsc_object_ptr.hpp"
#include <petscvec.h>
#include "vector.hpp"

namespace chive {
  class PetscVectorStorage final
      : public VectorStorage<PetscScalar>
  {
    public:
      template <typename S> friend class VectorSlice;

      PetscVectorStorage(VectorSpec spec);

      void add(const VectorStorage& rhs) override;
      void scale(const Number& factor) override;
      Number dot(const VectorStorage<Number>& rhs) override; 
      Real l2_norm() const override;
      std::shared_ptr<VectorStorage<Number>> allocate(VectorSpec spec) override;

      PetscObjectPtr<Vec> native() const { return ptr; }
    protected:
      Number* aquire_data_ptr() override;
      void release_data_ptr(Number* data) override;
    private:
      PetscObjectPtr<Vec> ptr;
  };

  using PetscVector = VectorBase<PetscVectorStorage>;
}

#endif
