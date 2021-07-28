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

#ifndef ALLIUM_ODE_EXPLICIT_INTEGRATOR_HPP
#define ALLIUM_ODE_EXPLICIT_INTEGRATOR_HPP

#include <allium/util/numeric.hpp>
#include <functional>
#include "integrator.hpp"

namespace allium {

  /**
   @defgroup ode ODE
   @brief Ordinary differential equation solvers.
   @{
   */

  /**
   @brief Base class for explicit time integrators.
   */
  template <typename V>
  class ExplicitIntegrator : public Integrator<V> {
    public:
      using Vector = V;
      using Number = typename Vector::Number;
      using typename Integrator<V>::Real;
      using F = std::function<void(Vector&, Real, const Vector&)>;

      virtual ~ExplicitIntegrator() {}

      virtual void setup(F f) = 0;
      virtual void initial_value(Real t0, const Vector& y0) = 0;
  };

  /// @}
}

#endif
