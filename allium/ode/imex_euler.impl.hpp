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
  void ImexEulerBase<N>::integrate(Real t1) {
    allium_assert(t1 >= m_t_cur, "new time value larger than old");

    auto y_new = allocate_like(*m_y_cur);
    auto rhs = allocate_like(*m_y_cur);

    while (m_t_cur < t1) {
      Real t_new = m_t_cur + m_dt;
      t_new = std::min(t1, t_new);

      Real h = t_new - m_t_cur;

      // rhs = y_cur + h * f_ex(t_cur, y_cur);
      apply_f_ex(*rhs, m_t_cur, *m_y_cur);
      *rhs *= h;
      *rhs += *m_y_cur;

      // y_cur is a good initial guess to the solver
      y_new->assign(*m_y_cur);

      // solve y_new - h * f_im(t_new, y_new) = rhs
      solve_implicit(*y_new, t_new, h, *rhs, InitialGuess::PROVIDED);

      m_y_cur.swap(y_new);
      m_t_cur = t_new;
    }
  }

}
