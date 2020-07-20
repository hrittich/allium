#ifndef CHIVE_LA_LOCAL_VECTOR_HPP
#define CHIVE_LA_LOCAL_VECTOR_HPP

#include <Eigen/Core>

namespace chive {

  template <typename N>
  class LocalVector
  {
    using Number = N;
    using Real = real_part_t<Number>;

    LocalVector& operator+= (const LocalVector& rhs) {
      m_storage += rhs.m_storage;
    }

    LocalVector& operator*= (Number rhs) {
      m_storage *= rhs;
      return *this;
    }

    Number dot(const LocalVector& rhs) const {
      return rhs->m_storage.dot(m_storage);
    }

    Number& operator() (size_t i_element) {
      return m_storage(i_element);
    }
    Number operator() (size_t i_element) const {
      return const_cast<LocalVector<N>>(this)(i_element);
    }


  private:
    Eigen::Matrix<N, Eigen::Dynamic, 1> m_storage;
  };

  template <typename N>
  LocalVector<N> operator* (N s, const Vector<N>& v) {
    LocalVector<N> w = v;
    w *= s;
    return w;
  }

}

#endif
