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

#include "imex_euler.hpp"

namespace allium {

  template <typename N>
  void ImexEulerBase<N>::integrate(Vector& y1, Real t0, const Vector& y0, Real t1) {

    auto y_old = allocate_like(y0);
    auto y_new = allocate_like(y0);
    auto rhs = allocate_like(y0);
    auto shift = allocate_like(y0);

    y_old->assign(y0);

    Real t_old = t0;

    while (t_old < t1) {
      Real t_new = t_old + m_dt;
      t_new = std::min(t1, t_new);

      Real h = t_new - t_old;

      // rhs = y_old + h * f_ex(t_old, y_old);
      apply_f_ex(*rhs, t_old, *y_old);
      *rhs *= h;
      *rhs += *y_old;

      // y_old is a good initial guess to the solver
      y_new->assign(*y_old);

      // solve y_new - h * f_im(t_new, y_new) = rhs
      solve_implicit(*y_new, t_new, h, *rhs, InitialGuess::PROVIDED);

      y_old.swap(y_new);
      t_old = t_new;
    }

    y1.assign(*y_old);
  }

}
