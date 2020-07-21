#include "gmres.hpp"

#include "local_vector.hpp"

#include <vector>

namespace chive {

  /** Overrides u by the vector obtained by applying a rotation in the i1-i2
   * plane to the vector u.
   *
   * The numbers c and s are the sin and cos of the corresponding angle. */
  template <typename N, typename V>
    void gmres_rotate(size_t i1, size_t i2, real_part_t<N> c, real_part_t<N> s, V& u)
  {
    using Number = N;

    Number new_entry1 = c * u[i1] - s * u[i2];
    Number new_entry2 = s * u[i1] + c * u[i2];

    u[i1] = new_entry1;
    u[i2] = new_entry2;
  }

  /** Incrementally computes the QR decomposition of an upper Hessenberg
   * matrix. */
  template <typename N>
  class HessenbergQr {
    public:
      using Number = N;
      using Real = real_part_t<N>;

      HessenbergQr(Number first_rhs_entry);

      void add_column(LocalVector<N> next_column, Number next_rhs_entry);

      /** The residual norm of the least-squares problem. */
      Real residual_norm() const;
    private:
      std::vector<LocalVector<N>> m_r_columns;  // columns of the matrix R
      std::vector<Real> m_cos_list;
      std::vector<Real> m_sin_list;
      std::vector<Number> m_rhs;
  };

  template <typename N>
  HessenbergQr<N>::HessenbergQr(Number first_rhs_entry)
    : m_rhs({first_rhs_entry})
  {}

  template <typename N>
  void HessenbergQr<N>::add_column(LocalVector<N> next_column, Number next_rhs_entry)
  {
    size_t n_columns = m_r_columns.size();

    assert(next_column.nrows() == n_columns + 2);

    // First, apply previous rotations to the new column
    for (size_t i_rot = 0; i_rot < n_columns; ++i_rot) {
      gmres_rotate(i_rot, i_rot+1, m_cos_list[i_rot], m_sin_list[i_rot], next_column);
    }

    // Second, compute the new rotation coefficients

    // Currently, we assume that the last entry in the next_column is a real
    // number, which is true for the GMRES method.
    assert(std::imag(next_column[n_columns+1]) == 0.0);

    Number r = next_column[n_columns];
    Real h = std::real(next_column[n_columns+1]);

    Real new_c = r / sqrt(r*r + h*h);
    Real new_s = -h / sqrt(r*r + h*h);

    m_cos_list.push_back(new_c);
    m_sin_list.push_back(new_s);

    // Third, apply new rotation
    gmres_rotate(n_columns, n_columns+1, new_c, new_s, next_column);

    // Fourth, apply the new rotation to rhs
    m_rhs.push_back(next_rhs_entry);
    gmres_rotate(n_columns, n_columns+1, new_c, new_s, m_rhs);
  }

  /** The residual norm of the least-squares problem. */
  template <typename N>
  typename HessenbergQr<N>::Real HessenbergQr<N>::residual_norm() const
  {
    size_t n_columns = m_r_columns.size();
    return abs(m_rhs[n_columns]);
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

