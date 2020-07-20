#include "gmres.hpp"

#include "local_vector.hpp"

#include <vector>

namespace chive {

  /** Overrides u by the vector obtained by applying a rotation in the i1-i2
   * plane to the vector u.
   *
   * The numbers c and s are the sin and cos of the corresponding angle. */
  template <typename N>
    void gmres_rotate(size_t i1, size_t i2, real_part_t<N> c, real_part_t<N> s, LocalVector<N>& u)
  {
    using Number = N;

    Number new_entry1 = c * u[i1] - s * u[i2];
    Number new_entry2 = s * u[i1] + c * u[i2];

    u[i1] = new_entry1;
    u[i2] = new_entry2;
  }

  template <typename N>
    Vector<N> gmres_inner(SparseMatrix<N> mat, Vector<N> rhs, real_part_t<N> abs_tol, Vector<N> x0)
  {
    using Number = N;
    using Real = real_part_t<Number>;

    std::vector<Vector<N>> krylov_base;

    auto x = x0;
    auto r = rhs - mat * x;

    Real beta = r.l2_norm();
    krylov_base.push_back(r / beta);

    std::vector<LocalVector<N>> upper_triangular;

    std::vector<Real> c_list;
    std::vector<Real> s_list;

    size_t i_iteration = 0;
    while (true) {

      Vector<N> v_hat = mat * krylov_base.at(0);

      // current column of the Hessenberg matrix
      LocalVector<N> hessenberg_column(i_iteration + 2);

      // orthogonalize and store coefficients
      for (size_t i_base = 0; i_base <= i_iteration; ++i_base) {
        hessenberg_column[i_base] = v_hat.dot(krylov_base.at(i_base));
        v_hat -= hessenberg_column[i_base] * krylov_base[i_base];
      }

      // entry (i_iteration+2, i_iteration+1) in the Hessenberg matrix
      Real hessenberg_extra = v_hat.l2_norm();
      hessenberg_column[i_iteration+1] = hessenberg_extra;

      // normalize new basis vector
      krylov_base.push_back(v_hat / hessenberg_extra);

      // --- reduce to triangular form via rotations

      // first, apply old rotations
      for (size_t i_rot = 0; i_rot < i_iteration; ++i_rot) {
        gmres_rotate(i_rot, i_rot+1, c_list[i_rot], s_list[i_rot], hessenberg_column);
      }

      Number r = hessenberg_column[i_iteration];
      // second, compute new coefficients
      Real new_c = r / sqrt(r*r + hessenberg_extra*hessenberg_extra);
      Real new_s = -hessenberg_extra / sqrt(r*r + hessenberg_extra*hessenberg_extra);
      c_list.push_back(new_c);
      s_list.push_back(new_s);

      // third, apply new rotation
      gmres_rotate(i_iteration, i_iteration+1, new_c, new_s, hessenberg_column);

      // fourth, apply new rotation to rhs
      // todo: ...


      ++i_iteration;
    }

  }

  /** Implementation of the GMRES algorithm.
   *
   * Saad, Y. & Schultz, M. H.
   * GMRES: A generalized minimal residual algorithm for solving nonsymmetric 
   * linear systems 
   * SIAM J. Sci. Statist. Comput., 1986, 7, 856-869
   */
  template <typename N>
    Vector<N> gmres(SparseMatrix<N> mat, Vector<N> rhs, real_part_t<N> tol)
  {
    using Real = real_part_t<N>;

    auto x = rhs.zeros_like();
    auto r = rhs - mat * x;

    Real abs_tol = tol * r.l2_norm();


    return x;
  }
}

