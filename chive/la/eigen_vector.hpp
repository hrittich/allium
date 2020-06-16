#ifndef CHIVE_LA_EIGEN_VECTOR
#define CHIVE_LA_EIGEN_VECTOR

#include <Eigen/Core>
#include "vector.hpp"

namespace chive {
  template <typename NumberT>
  class EigenVectorStorage : public VectorStorage<NumberT>
  {
    public:
      template <typename N, typename S> friend class VectorSlice;

      using EVector = Eigen::Matrix<NumberT, Eigen::Dynamic, 1>;
      using typename VectorStorage<NumberT>::Number;
      using Real = real_part_t<NumberT>;

      EigenVectorStorage(VectorSpec spec)
        : VectorStorage<NumberT>(spec),
          vec(spec.get_global_size())
      {}

      void add(const VectorStorage<NumberT>& rhs) override;
      void scale(const Number& factor) override;
      Real l2_norm() const override;
    protected:
      Number* aquire_data_ptr() override;
      void release_data_ptr(Number* data) override;

    private:
      EVector vec;
  };

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

#endif
