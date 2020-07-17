#ifndef CHIVE_LA_EIGEN_VECTOR_HPP
#define CHIVE_LA_EIGEN_VECTOR_HPP

#include <chive/util/extern.hpp>
#include "vector.hpp"
#include <Eigen/Core>

namespace chive {
  template <typename NumberT>
  class EigenVectorStorage final
    : public VectorStorageBase<EigenVectorStorage<NumberT>, NumberT>
  {
    public:
      using BaseVector = Eigen::Matrix<NumberT, Eigen::Dynamic, 1>;
      using typename VectorStorage<NumberT>::Number;
      using Real = real_part_t<NumberT>;

      explicit EigenVectorStorage(VectorSpec spec);

      void add(const VectorStorage<NumberT>& rhs) override;
      void scale(const Number& factor) override;
      NumberT dot(const VectorStorage<NumberT>& rhs) override;
      Real l2_norm() const override;

      BaseVector& native() { return vec; }

      EigenVectorStorage* allocate(VectorSpec spec) {
        return new EigenVectorStorage(spec);
      }
    protected:
      Number* aquire_data_ptr() override;
      void release_data_ptr(Number* data) override;

    private:
      BaseVector vec;
  };

  template <typename N>
  using EigenVector = VectorBase<EigenVectorStorage<N>>;

  #define CHIVE_EIGEN_VECTOR_DECL(T, N) \
    T class EigenVectorStorage<N>; \
    T class VectorBase<EigenVectorStorage<N>>;
  CHIVE_EXTERN(CHIVE_EIGEN_VECTOR_DECL)
}

#endif
