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
#include "vector.hpp"
#include "sparse_matrix.hpp"
#include "linear_solver.hpp"

namespace allium {
  template <typename N>
    Vector<N> cg_(SparseMatrix<N> mat, Vector<N> rhs, real_part_t<N> tol = 1e-6);

  template <typename Mat, typename... Args>
    Vector<typename Mat::Number> cg(Mat mat, Args&&... args)
    {
      return cg_<typename Mat::Number>(mat, std::forward<Args>(args)...);
    }

  template <typename N>
  class CgSolverBase {
    public:
      using Number = N;
      using Real = real_part_t<N>;

      void solve(const VectorStorage<N>& rhs, VectorStorage<N>& solution);
    protected:
      Real m_tol;

      virtual void matvec(VectorStorage<N>& out, const VectorStorage<N>& in) = 0;
  };

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

      CgSolver(Real tolerance = 1e-8)
      {
        m_tol = tolerance;
      }

      void setup(std::shared_ptr<Matrix> mat) override {
        m_mat = mat;
      }

      void solve(const Vector& rhs, Vector& solution) override {
        CgSolverBase<Number>::solve(rhs, solution);
      }

    private:
      std::shared_ptr<Matrix> m_mat;

      void matvec(VectorStorage<Number>& out, const VectorStorage<Number>& in) override {
        allium_assert(dynamic_cast<const V*>(&in));
        allium_assert(dynamic_cast<V*>(&out));

        m_mat->apply(static_cast<V&>(out),
                     static_cast<const V&>(in));
      }
  };

  #define ALLIUM_CG_DECL(T, N) \
    T class CgSolverBase<N>; \
    T Vector<N> cg_<N>(SparseMatrix<N>, Vector<N>, real_part_t<N>);
  ALLIUM_EXTERN(ALLIUM_CG_DECL)
}

#endif
