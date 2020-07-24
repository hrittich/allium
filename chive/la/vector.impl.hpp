#include "vector.hpp"

#include "eigen_vector.hpp"
#include "petsc_vector.hpp"

namespace chive {

  #if defined(CHIVE_DEFAULT_BACKEND_EIGEN)
    template <typename N>
      Vector<N> make_vector(VectorSpec spec) {
        return EigenVector<N>(spec);
      }
  #elif defined(CHIVE_DEFAULT_BACKEND_PETSC)
    template <typename N>
      Vector<N> make_vector_(VectorSpec spec) {
        throw std::logic_error("Not implemented");
      }

    template <>
      Vector<PetscScalar> make_vector_(VectorSpec spec) {
        return PetscVector(spec);
      }

    template <typename N>
      Vector<N> make_vector(VectorSpec spec) {
        return make_vector_<N>(spec);
      }
  #else
    #error "No default linear algebra backend specified."
  #endif
}
