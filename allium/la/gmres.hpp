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

#ifndef ALLIUM_LA_GMRES_HPP
#define ALLIUM_LA_GMRES_HPP

#include "vector_storage.hpp"
#include "linear_operator.hpp"
#include "iterative_solver.hpp"
#include <allium/util/extern.hpp>
#include "default.hpp"

namespace allium {

  /**
    @brief Implementation of the GMRES algorithm (Saad and Schultz, 1986).

    @ingroup linear_solver

    This class should be used using the GmresSolver type.

    Saad, Youcef, and Martin H. Schultz. 1986. "GMRES: A Generalized Minimal
    Residual Algorithm for Solving Nonsymmetric Linear Systems."
    SIAM J. Sci. Statist. Comput., 7 (3): 856–69.
    https://doi.org/10.1137/0907058.
   */
  template <typename N>
  class GmresSolverBase : public IterativeSolverBase<N>
  {
    public:
      using Number = N;
      using Real = real_part_t<N>;
      using IterativeSolverBase<N>::tolerance;

      void solve(VectorStorage<N>& result, const VectorStorage<N>& rhs);
    private:
      bool inner_solve(VectorStorage<N>& result,
                       const VectorStorage<N>& residual,
                       real_part_t<N> residual_norm,
                       real_part_t<N> abs_tol);

      size_t m_max_krylov_size;
  };

  /**
   @ingroup linear_solver
   */
  template <typename V>
  using GmresSolver = IterativeSolverMixin<GmresSolverBase<typename V::Number>, V>;

  /**
    Convenice method to use the GmresSolver.

    @ingroup linear_solver
  */
  template <typename V, typename M>
    void gmres(V& result,
               std::shared_ptr<M> mat,
               const V& rhs,
               real_part_t<typename V::Number> tol = 1e-6)
  {
    GmresSolver<typename M::Vector> solver;
    solver.tolerance(tol);
    solver.setup(mat);
    solver.solve(result, rhs);
  }

  #define ALLIUM_LA_GMRES_DECL(extern, N) \
    extern template class GmresSolverBase<N>; \
    extern template class IterativeSolverMixin<GmresSolverBase<N>, DefaultVector<N>>;
  ALLIUM_EXTERN_N(ALLIUM_LA_GMRES_DECL)
}

#endif
