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

#ifndef ALLIUM_ODE_IMEX_INTEGRATOR_HPP
#define ALLIUM_ODE_IMEX_INTEGRATOR_HPP

#include <functional>
#include <memory>
#include <allium/la/linear_operator.hpp>
#include <allium/util/numeric.hpp>

namespace allium {

  /// @addtogroup ode
  /// @{

  /**
   Interface for implicit-explicit time-stepping schemes have a linear
   implicit part.

   @f[
      \dot{y} = F_\mathrm{ex}(t, y) + F_\mathrm{im}(t, y)
   @f]

   *Experimental interface*; this interface might change.
   */
  template <typename V>
  class ImexIntegrator {
    public:
      using Vector = V;
      using Number = typename V::Number;

      using ExplicitF = std::function<void(Vector&,
                                           real_part_t<Number>,
                                           const Vector&)>;
      using ImplicitSolve = std::function<void(Vector& y,
                                               real_part_t<Number> t,
                                               Number a,
                                               const Vector& r)>;

      virtual ~ImexIntegrator() {}

      /**
       Set the callback functions for the solver.

       @param[in] f_ex Function that evaluates the explicit part.
       @param[in] solve_impl Function that solves the implicit part.

       The function f_ex should evaluate the right hand side
       @f$ G(t, y) @f$.

       The function solve_impl should solve
       @f[
        y - a f(t, y) = r
       @f]

       */
      virtual void setup(ExplicitF f_ex, ImplicitSolve solve_impl) = 0;
      virtual void initial_values(real_part_t<Number> t0, const Vector& y0) = 0;
  };

  /// @}
}

#endif
