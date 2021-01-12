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

#ifndef ALLIUM_LA_CG_HPP
#define ALLIUM_LA_CG_HPP

#include <allium/util/extern.hpp>
#include "vector_storage.hpp"
#include "sparse_matrix.hpp"
#include "linear_solver.hpp"
#include <allium/util/warnings.hpp>

namespace allium {

  template <typename N>
  class CgSolverBase {
    public:
      using Number = N;
      using Real = real_part_t<N>;

      CgSolverBase(Real tolerance = 1e-8)
      {
        m_tol = tolerance;
      }

      void solve(VectorStorage<N>& solution, const VectorStorage<N>& rhs);
    protected:
      Real m_tol;

      virtual void matvec(VectorStorage<N>& out, const VectorStorage<N>& in) = 0;
  };

  /**
   The conjugate gradient method, solves @f$ A x = b @f$.
   @ingroup linear_solver
   */
  template <typename V>
  class CgSolver
      : public CgSolverBase<typename V::Number>,
        public LinearSolver<V> {
    public:
      using Matrix = LinearOperator<V>;
      using typename LinearSolver<V>::Vector;
      using Number = typename Vector::Number;
      using Real = typename Vector::Real;
      using CgSolverBase<Number>::m_tol;

      using CgSolverBase<Number>::CgSolverBase;

      void setup(std::shared_ptr<Matrix> mat) override {
        m_mat = mat;
      }

      void solve(Vector& solution, const Vector& rhs) override {
        CgSolverBase<Number>::solve(solution, rhs);
      }

    private:
      std::shared_ptr<Matrix> m_mat;

      void matvec(VectorStorage<Number>& out, const VectorStorage<Number>& in) override {
        ALLIUM_NO_NONNULL_WARNING
        allium_assert(dynamic_cast<const V*>(&in) != nullptr);
        allium_assert(dynamic_cast<V*>(&out) != nullptr);
        ALLIUM_RESTORE_WARNING

        m_mat->apply(static_cast<V&>(out),
                     static_cast<const V&>(in));
      }
  };

  /**
    The conjugate gradient method.

    Convenience method for using CgSolver.

    @ingroup linear_solver
   */
  template <typename V, typename M>
    void cg(V& solution, std::shared_ptr<M> mat, const V& rhs, real_part_t<typename V::Number> tol = 1e-8)
  {
    CgSolver<V> solver(tol);
    solver.setup(mat);
    solver.solve(solution, rhs);
  }

  #define ALLIUM_CG_DECL(extern, N) \
    extern template class CgSolverBase<N>;
  ALLIUM_EXTERN_N(ALLIUM_CG_DECL)
}

#endif
