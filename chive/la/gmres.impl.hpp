// Copyright 2020 Hannah Rittich
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CHIVE_LA_GMRES_IMPL_HPP
#define CHIVE_LA_GMRES_IMPL_HPP

#include "gmres.hpp"

#include "local_vector.hpp"

#include <vector>

namespace chive {

  /** Compute the Givens rotation coefficients. Only works for real numbers.
   * Compute c, s, such that
   *
   * | c  -s |   | a |   | r |
   * | s   c | * | b | = | 0 |
   *
   * */
  template <typename N>
    void real_givens(N& c, N& s, N a, N b)
  {
    if (b == 0.0) {
      c = 1;
      s = 0;
    } else {
      if (std::abs(b) > std::abs(a)) {
        N tau = -a/b;
        s = 1.0 / sqrt(1.0 + tau*tau);
        c = s * tau;
      } else {
        N tau = -b/a;
        c = 1.0 / sqrt(1.0 + tau*tau);
        s = c * tau;
      }
    }
  }

  template <typename N> void givens(N& c, N& s, N a, N b)
  {
    using Real = real_part_t<N>;

    Real c_alpha, s_alpha;
    real_givens(c_alpha, s_alpha, std::real(a), std::imag(a));
    auto r_a = c_alpha * std::real(a) - s_alpha * std::imag(a);

    Real c_beta, s_beta;
    real_givens(c_beta, s_beta, std::real(b), std::imag(b));
    auto r_b = c_beta * std::real(b) - s_beta * std::imag(b);

    Real c_theta, s_theta;
    real_givens(c_theta, s_theta, r_a, r_b);

    c = c_theta;
    s = s_theta * N(c_alpha*c_beta+s_alpha*s_beta,
                    c_alpha*s_beta-c_beta*s_alpha);
  }

  template <> void givens(float& c, float& s, float a, float b)
  { return real_givens<float>(c, s, a, b); }
  template <> void givens(double& c, double& s, double a, double b)
  { return real_givens<double>(c, s, a, b); }

  template <typename N>
  N conj(N z) {
    return std::conj(z);
  }
  template <>
  double conj(double z) { return z; }
  template <>
  float conj(float z) { return z; }

  /** Overrides u by the vector obtained by applying a rotation in the i1-i2
   * plane to the vector u.
   *
   * The numbers c and s are the sin and cos of the corresponding angle. */
  template <typename N, typename V>
    void gmres_rotate(size_t i1, size_t i2, N c, N s, V& u)
  {
    using Number = N;

    Number new_entry1 = c * u[i1] - s * u[i2];
    Number new_entry2 = conj<N>(s) * u[i1] + c * u[i2];

    u[i1] = new_entry1;
    u[i2] = new_entry2;
  }

  /** Incrementally computes the QR decomposition of an upper Hessenberg
   * matrix. */
  template <typename N>
  class HessenbergQr {
    public:
      using Number = N;
      using Real = real_part_t<Number>;
      using Vector = LocalVector<Number>;

      HessenbergQr(Number first_rhs_entry);

      void add_column(LocalVector<N> next_column, Number next_rhs_entry);

      /** The residual norm of the least-squares problem. */
      Real residual_norm() const;

      /** The solution vector. */
      Vector solution() const;
    private:
      std::vector<LocalVector<N>> m_r_columns;  // columns of the matrix R
      std::vector<Number> m_cos_list;
      std::vector<Number> m_sin_list;
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
      gmres_rotate<Number>(i_rot, i_rot+1, m_cos_list[i_rot], m_sin_list[i_rot], next_column);
    }

    // Second, compute the new rotation coefficients

    // Currently, we assume that the last entry in the next_column is a real
    // number, which is true for the GMRES method.
    //assert(std::imag(next_column[n_columns+1]) == 0.0);

    Number r = next_column[n_columns];
    Number h = next_column[n_columns+1];

    Number new_c, new_s;
    givens(new_c, new_s, r, h);

    m_cos_list.push_back(new_c);
    m_sin_list.push_back(new_s);

    // Third, apply new rotation
    gmres_rotate<Number>(n_columns, n_columns+1, new_c, new_s, next_column);

    // Fourth, apply the new rotation to rhs
    m_rhs.push_back(next_rhs_entry);
    gmres_rotate<Number>(n_columns, n_columns+1, new_c, new_s, m_rhs);

    // Store new column of R
    m_r_columns.push_back(std::move(next_column));
  }

  template <typename N>
  typename HessenbergQr<N>::Vector HessenbergQr<N>::solution() const
  {
    size_t n_columns = m_r_columns.size();

    Vector result(n_columns);

    // upper triangular backward solve
    for (int i_row = n_columns-1; i_row >= 0; i_row--)
    {
      Number acc = m_rhs[i_row];
      for (int i_col = i_row+1; i_col < n_columns; ++i_col) {
        acc -= m_r_columns[i_col][i_row] * result[i_col];
      }
      result[i_row] = acc / m_r_columns[i_row][i_row];
    }

    return result;
  }

  /** The residual norm of the least-squares problem. */
  template <typename N>
  typename HessenbergQr<N>::Real HessenbergQr<N>::residual_norm() const
  {
    assert(m_rhs.size() == m_r_columns.size()+1);
    size_t n_columns = m_r_columns.size();
    return abs(m_rhs[n_columns]);
  }


  template <typename N>
    Vector<N> gmres_inner(SparseMatrix<N> mat,
                          Vector<N> rhs,
                          real_part_t<N> abs_tol,
                          size_t max_iter,
                          Vector<N> x0)
  {
    using Number = N;
    using Real = real_part_t<Number>;

    std::vector<Vector<N>> krylov_base;

    auto x = x0;
    auto r = rhs - mat * x;

    Real beta = r.l2_norm();
    krylov_base.push_back(r / beta);

    HessenbergQr<N> qr(beta);

    for (size_t i_iteration = 0; i_iteration < max_iter; ++i_iteration) {
      Vector<N> v_hat = mat * krylov_base.at(i_iteration);

      // current column of the Hessenberg matrix
      LocalVector<N> hessenberg_column(i_iteration + 2);

      // orthogonalize and store coefficients using modified Gram-Schmidt
      // todo: The last orthogonalization does not need to be computed.
      //       Hence, we could save some work here.
      for (size_t i_base = 0; i_base <= i_iteration; ++i_base) {
        hessenberg_column[i_base] = v_hat.dot(krylov_base.at(i_base));
        v_hat -= hessenberg_column[i_base] * krylov_base.at(i_base);
      }

      // entry (i_iteration+2, i_iteration+1) in the Hessenberg matrix
      Real hessenberg_extra = v_hat.l2_norm();
      hessenberg_column[i_iteration+1] = hessenberg_extra;

      // normalize new basis vector
      //
      // todo: normalization can be avoided if the scaling coefficient is
      //       stored instead
      krylov_base.push_back(v_hat / hessenberg_extra);

      // add new column to Hessenberg-QR decomposition and new RHS entry 0
      qr.add_column(std::move(hessenberg_column), 0.0);

      if (qr.residual_norm() <= abs_tol)
        break;
    }

    LocalVector<N> y = qr.solution();
    // compute linear combination of basis vectors to approximate the
    // solution of the LGS
    assert(y.nrows() == krylov_base.size() - 1);
    for (size_t i_base = 0; i_base < y.nrows(); ++i_base) {
      x += y[i_base] * krylov_base[i_base];
    }

    return x;
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

#endif
