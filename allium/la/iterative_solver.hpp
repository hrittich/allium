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

#ifndef ALLIUM_LA_ITERATIVE_SOLVER_HPP
#define ALLIUM_LA_ITERATIVE_SOLVER_HPP

#include <allium/util/numeric.hpp>
#include "vector_storage.hpp"
#include "linear_solver.hpp"
#include "linear_operator.hpp"

namespace allium {

  /**
   @brief Base class for iterative solvers.
   */
  template <typename N>
  class IterativeSolverBase {
    public:
      using Number = N;

      IterativeSolverBase(real_part_t<N> tol = 1e-8) : m_tol(tol) {}
      virtual ~IterativeSolverBase() {};

      real_part_t<N> tolerance() { return m_tol; }
      void tolerance(real_part_t<N> t) { m_tol = t; }
    protected:
      virtual void apply_matrix(VectorStorage<Number>& out, const VectorStorage<Number>& in) = 0;

    private:
      real_part_t<N> m_tol;
  };

  /**
   @brief Mixin for iterative solvers, which implements the apply_matrix method.
   */
  template <typename Base, typename V>
  class IterativeSolverMixin :
    public Base,
    public LinearSolver<V>
  {
    public:
      using Base::Base;
      using Vector = V;
      using Number = typename Vector::Number;
      using Matrix = LinearOperator<V>;

      void setup(std::shared_ptr<Matrix> mat) override {
        m_mat = mat;
      }

      void solve(V& solution,
                 const V& rhs,
                 InitialGuess initial_guess = InitialGuess::NOT_PROVIDED) override {
        Base::solve(solution, rhs, initial_guess);
      }

    private:
      std::shared_ptr<Matrix> m_mat;

      void apply_matrix(VectorStorage<Number>& out, const VectorStorage<Number>& in)
      {
        ALLIUM_NO_NONNULL_WARNING
        allium_assert(dynamic_cast<const V*>(&in) != nullptr);
        allium_assert(dynamic_cast<V*>(&out) != nullptr);
        ALLIUM_RESTORE_WARNING

        m_mat->apply(static_cast<V&>(out),
                     static_cast<const V&>(in));
      }
  };

};

#endif
