#ifndef CHIVE_LA_CG_HPP
#define CHIVE_LA_CG_HPP

#include <chive/util/extern.hpp>
#include "vector.hpp"
#include "sparse_matrix.hpp"

namespace chive {
  template <typename N>
    Vector<N> cg_(SparseMatrix<N> mat, Vector<N> rhs, real_part_t<N> tol = 1e-6);

  template <typename Mat, typename... Args>
    Vector<typename Mat::Number> cg(Mat mat, Args&&... args)
    {
      return cg_<typename Mat::Number>(mat, std::forward<Args>(args)...);
    }

  #define CHIVE_CG_DECL(T, N) \
    T Vector<N> cg_<N>(SparseMatrix<N>, Vector<N>, real_part_t<N>);
  CHIVE_EXTERN(CHIVE_CG_DECL)
}

#endif
