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
  CgSolverBase<N>::~CgSolverBase()
  {}

  template <typename N>
  void CgSolverBase<N>::solve(VectorStorage<N>& solution,
                              const VectorStorage<N>& rhs,
                              InitialGuess initial_guess)
  {
    auto residual = allocate_like(rhs);
    auto new_residual = allocate_like(rhs);
    auto x = allocate_like(rhs);
    auto tmp1 = allocate_like(rhs);

    switch (initial_guess) {
      case InitialGuess::NOT_PROVIDED:
        set_zero(*x);
      break;
      case InitialGuess::PROVIDED:
        x->assign(solution);
      break;
    }

    //residual = rhs - m_mat * x;
    matvec(*tmp1, *x);
    residual->assign(rhs);
    residual->add_scaled(-1, *tmp1);

    Real residual_norm_sq = std::real(residual->dot(*residual));
    auto p = allocate_like(rhs);
    p->assign(*residual);

    Real abs_tol = rhs.l2_norm() * m_tol;

    m_iteration_count = 0;
    if (sqrt(residual_norm_sq) > abs_tol)
    {
      auto Ap = allocate_like(rhs);
      while (true) {
        m_iteration_count++;

        matvec(*Ap, *p);

        Real alpha = std::real(residual->dot(*residual)) / std::real(p->dot(*Ap));

        // x = x + alpha * p;
        x->add_scaled(alpha, *p);

        // new_residual = residual - alpha * Ap;
        new_residual->assign(*residual);
        new_residual->add_scaled(-alpha, *Ap);

        Real new_residual_norm_sq
          = std::real(new_residual->dot(*new_residual));

        if (sqrt(new_residual_norm_sq) <= abs_tol) break;

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
