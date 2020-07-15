#include "eigen_vector.hpp"

namespace chive {

  template <typename N>
  void EigenVectorStorage<N>::add(const VectorStorage<N>& rhs)
  {
    const EigenVectorStorage* eigen_rhs
      = dynamic_cast<const EigenVectorStorage*>(&rhs);
    if (eigen_rhs != nullptr) {
      vec += eigen_rhs->vec;
    } else {
      throw std::logic_error("Not implemented");
    }
  }

  template <typename Number>
  void EigenVectorStorage<Number>::scale(const Number& factor) {
    vec *= factor;
  }

  template <typename N>
  N EigenVectorStorage<N>::dot(const VectorStorage<N>& rhs) {
    const EigenVectorStorage* eigen_rhs
      = dynamic_cast<const EigenVectorStorage*>(&rhs);
    if (eigen_rhs != nullptr) {
      // eigen has a dot product wich is linear in the FIRST argument
      return eigen_rhs->vec.dot(vec);
    } else {
      throw std::logic_error("Not implemented");
    }
  }

  template <typename Number>
    real_part_t<Number>
    EigenVectorStorage<Number>::l2_norm() const
  {
    return vec.norm();
  }

  template <typename Number>
    typename VectorStorage<Number>::Number* EigenVectorStorage<Number>::aquire_data_ptr()
  {
    return vec.data();
  }

  template <typename Number>
    void EigenVectorStorage<Number>::release_data_ptr(Number* data)
  {}

}
