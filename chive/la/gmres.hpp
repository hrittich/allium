#ifndef CHIVE_LA_GMRES_HPP
#define CHIVE_LA_GMRES_HPP

#include "vector.hpp"
#include "sparse_matrix.hpp"
#include <chive/util/extern.hpp>

namespace chive {

  template <typename N>
    Vector<N> gmres(SparseMatrix<N> mat, Vector<N> rhs, real_part_t<N> tol = 1e-6);

  #define CHIVE_LA_GMRES_DECL(T, N) \
    T Vector<N> gmres(SparseMatrix<N>, Vector<N>, real_part_t<N> tol);
  CHIVE_EXTERN(CHIVE_LA_GMRES_DECL)
}

#endif
