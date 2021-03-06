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

#ifndef ALLIUM_LA_LINEAR_SOLVER_HPP
#define ALLIUM_LA_LINEAR_SOLVER_HPP

#include <memory>
#include "vector_storage.hpp"
#include "linear_operator.hpp"

namespace allium {

  /**
   @defgroup linear_solver Linear Solver
   @{
  */

  /**
   @brief Behavior for the initial guess.
   */
  enum class InitialGuess {
    NOT_PROVIDED, ///< Initialize with zero.
    PROVIDED ///< Use the value from the solution vector.
  };

  /**
   @brief Linear solver interface.
   */
  template <typename V>
  class LinearSolver
  {
    public:
      virtual ~LinearSolver() {}
      static_assert(std::is_base_of<VectorStorage<typename V::Number>, V>::value);

      using Matrix = LinearOperator<V>;
      using Vector = V;

      virtual void setup(std::shared_ptr<Matrix> mat) = 0;
      virtual void solve(Vector& solution,
                         const Vector& rhs,
                         InitialGuess initial_guess = InitialGuess::NOT_PROVIDED) = 0;
  };

  /// @}

}

#endif
