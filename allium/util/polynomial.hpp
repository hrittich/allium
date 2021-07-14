// Copyright 2021 Hannah Rittich
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

#ifndef ALLIUM_UTIL_POLYNOMIAL_HPP
#define ALLIUM_UTIL_POLYNOMIAL_HPP

#include <allium/util/assert.hpp>
#include <allium/config.hpp>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <complex>
#include "numeric.hpp"

#ifdef ALLIUM_USE_GSL
#include <gsl/gsl_poly.h>
#endif

namespace allium {

  /**
   @brief A polynomial given by their coefficients.

   The polynomial is given in the form
   @f[
    P(x) = c_0 + c_1 x + c_2 x^2 + \cdots + c_n x^{n-1}
   @f]

   This class requires the use of the GNU Scientific Library.
   */
  template <typename N>
  class Polynomial final {
    public:
      using Number = N;
      using Real = real_part_t<Number>;

      /**
       Construct a polynomial.

       @param [in] coeffs The coefficients @f$ c @f$. The lenght of the vector
         determines n.
      */
      Polynomial(std::vector<N> coeffs)
        : m_coeffs(coeffs)
      {}

      #ifdef ALLIUM_USE_GSL
      template <typename N1 = N,
                std::enable_if_t<std::is_floating_point<N1>::value, int> = 0>
      N eval(N x) const {
        std::vector<double> coeffs_d(m_coeffs.begin(), m_coeffs.end());
        return gsl_poly_eval(coeffs_d.data(), coeffs_d.size(), x);
      }

      template <typename N1 = N,
                std::enable_if_t<is_complex<N1>::value, int> = 0>
      N eval(N x) const {
        std::vector<gsl_complex> coeffs_z(m_coeffs.size());
        std::transform(m_coeffs.begin(), m_coeffs.end(),
                       coeffs_z.begin(),
                       [](N c) { return gsl_complex{c.real(), c.imag()}; });

        gsl_complex result
          = gsl_complex_poly_complex_eval(coeffs_z.data(), coeffs_z.size(),
                                          gsl_complex{x.real(), x.imag()});

        return N(GSL_REAL(result), GSL_IMAG(result));
      }
      #endif

      #ifdef ALLIUM_USE_GSL
      /**
       Compute the roots of the polynomial.

       This function is only implemented for real polynomial coefficients.
      */
      template <typename N1 = N,
                std::enable_if_t<std::is_floating_point<N1>::value, int> = 0>
      std::vector<std::complex<Real>> roots() const
      {
        std::vector<double> coeffs_d(m_coeffs.begin(), m_coeffs.end());
        std::vector<double> result(2*deg());

        gsl_poly_complex_workspace* work
            = gsl_poly_complex_workspace_alloc(coeffs_d.size());

        gsl_poly_complex_solve(coeffs_d.data(), coeffs_d.size(), work, result.data());
        gsl_poly_complex_workspace_free(work);

        std::vector<std::complex<Real>> converted_result(deg());
        for (int i=0; i < deg(); ++i) {
          converted_result.at(i) = std::complex<Real>(result.at(2*i),
                                                      result.at(2*i+1));
        }

        return converted_result;
      }
      #endif

      /**
       Returns the derivative of the polynomial.
      */
      Polynomial derivative() const {
        std::vector<N> der_coeffs(m_coeffs.size()-1);
        for (size_t i=0; i < der_coeffs.size(); ++i) {
          der_coeffs.at(i) = N(i+1) * m_coeffs.at(i+1);
        }

        return Polynomial(der_coeffs);
      }

      /**
       Computes the anti-derivative @f$ Q @f$ of the polynomial with
       @f$ Q(0) = 0 @f$, i.e., with vanishing constant coefficient.
      */
      Polynomial anti_derivative() const {
        std::vector<N> anti_coeffs(m_coeffs.size()+1);

        anti_coeffs.at(0) = 0;
        for (size_t i=1; i < anti_coeffs.size(); ++i) {
          anti_coeffs.at(i) = (N(1)/N(i)) * m_coeffs.at(i-1);
        }

        return Polynomial(anti_coeffs);
      }

      #ifdef ALLIUM_USE_GSL
      /**
        Returns a polynomial @f$ Q @f$ such that
        @f[
          Q(x_0) = \int_a^{x_0} P(x)\,dx
          \,.
        @f]

        @param [in] lower_bound The lower bound a.
       */
      Polynomial integrate(N lower_bound) const {
        auto anti = anti_derivative();

        auto int_coeffs = anti.coeffs();
        allium_assert(int_coeffs.at(0) == N(0));
        int_coeffs.at(0) = -anti.eval(lower_bound);

        return Polynomial(int_coeffs);
      }
      #endif

      Polynomial operator+ (const Polynomial& other) const {
        std::vector<N> sum_coeffs(std::max(m_coeffs.size(), other.m_coeffs.size()),
                                  N(0));

        for (size_t i = 0; i < m_coeffs.size(); ++i)
          sum_coeffs.at(i) += m_coeffs.at(i);

        for (size_t i = 0; i < other.m_coeffs.size(); ++i)
          sum_coeffs.at(i) += other.m_coeffs.at(i);

        return Polynomial(sum_coeffs);
      }

      Polynomial operator* (const Polynomial& other) const {
        // Initialize coeffs with zero
        std::vector<N> mul_coeffs(deg() + other.deg() + 1, N(0.0));

        for (size_t i=0; i < m_coeffs.size(); ++i) {
          for (size_t j=0; j < other.m_coeffs.size(); ++j) {
            mul_coeffs.at(i+j) += m_coeffs.at(i) * other.m_coeffs.at(j);
          }
        }

        return Polynomial(mul_coeffs);
      }

      /** The degree of the polynomial. */
      int deg() const { return m_coeffs.size()-1; }

      /**
       The coefficients @f$ c @f$ of the polynomial.
      */
      std::vector<N> coeffs() const { return m_coeffs; }
    private:
      std::vector<N> m_coeffs;
  };

}

#endif
