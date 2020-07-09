#include "cg.hpp"

#include <complex>

namespace chive {

  template <typename N>
    Vector<N> cg(SparseMatrix<N> mat, Vector<N> rhs, real_part_t<N> tol)
  {
    using Number = N;
    using Real = real_part_t<N>;

    Vector<N> residual;
    Vector<N> new_residual;
    auto x = rhs.zeros_like();

    residual = rhs - mat * x;
    Real residual_norm_sq = std::real(residual.dot(residual));
    Vector<N> p = residual;

    Real rel_tol = rhs.l2_norm() * tol;

    while (true) {
      Vector<N> Ap = mat * p;
      Real alpha = std::real(residual.dot(residual)) / std::real(p.dot(Ap));
      x = x + alpha * p;
      new_residual = residual - alpha * Ap;
      Real new_residual_norm_sq
        = std::real(new_residual.dot(new_residual));

      if (sqrt(new_residual_norm_sq) <= rel_tol) break;

      Real beta = new_residual_norm_sq / residual_norm_sq;
      p = new_residual + beta * p;

      residual = new_residual;
      residual_norm_sq = new_residual_norm_sq;
    }

    return x;
  }

}
