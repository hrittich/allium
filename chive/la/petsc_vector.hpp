#ifndef CHIVE_LA_PETSC_VECTOR_HPP
#define CHIVE_LA_PETSC_VECTOR_HPP

#include "petsc_object_ptr.hpp"
#include <petscvec.h>
#include "vector.hpp"

namespace chive {
  class PetscVectorStorage final
      : public VectorStorage<PetscScalar>,
        public std::enable_shared_from_this<PetscVectorStorage>
  {
    public:
      template <typename N, typename S> friend class VectorSlice;

      PetscVectorStorage(VectorSpec spec);

      void add(const VectorStorage& rhs) override;
      void scale(const Number& factor) override;
      Real l2_norm() const override;

    protected:
      Number* aquire_data_ptr() override;
      void release_data_ptr(Number* data) override;
    private:
      PetscObjectPtr<Vec> ptr;
  };
}

#endif
