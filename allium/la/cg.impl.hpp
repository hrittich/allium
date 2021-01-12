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

#include "cg.hpp"

#include <complex>

namespace allium {

  template <typename N>
  void CgSolverBase<N>::solve(VectorStorage<N>& solution, const VectorStorage<N>& rhs)
  {
    auto residual = allocate_like(rhs);
    auto new_residual = allocate_like(rhs);
    auto x = allocate_like(rhs);
    auto tmp1 = allocate_like(rhs);
    set_zero(*x);

    //residual = rhs - m_mat * x;
    matvec(*tmp1, *x);
    residual->assign(rhs);
    residual->add_scaled(-1, *tmp1);

    Real residual_norm_sq = std::real(residual->dot(*residual));
    auto p = allocate_like(rhs);
    p->assign(*residual);

    Real rel_tol = rhs.l2_norm() * m_tol;

    if (sqrt(residual_norm_sq) > rel_tol)
    {
      auto Ap = allocate_like(rhs);
      while (true) {
        matvec(*Ap, *p);

        Real alpha = std::real(residual->dot(*residual)) / std::real(p->dot(*Ap));

        // x = x + alpha * p;
        x->add_scaled(alpha, *p);

        // new_residual = residual - alpha * Ap;
        new_residual->assign(*residual);
        new_residual->add_scaled(-alpha, *Ap);

        Real new_residual_norm_sq
          = std::real(new_residual->dot(*new_residual));

        if (sqrt(new_residual_norm_sq) <= rel_tol) break;

        Real beta = new_residual_norm_sq / residual_norm_sq;
        // p = beta * p + new_residual
        *p *= beta;
        *p += *new_residual;

        // @todo Remove unecessary copying
        residual->assign(*new_residual);
        residual_norm_sq = new_residual_norm_sq;
      }
    }

    solution.assign(*x);
  }

}
