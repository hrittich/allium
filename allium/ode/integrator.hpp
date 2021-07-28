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

#ifndef ALLIUM_ODE_INTEGRATOR_HPP
#define ALLIUM_ODE_INTEGRATOR_HPP

#include <allium/util/numeric.hpp>

namespace allium {

  /**
   @brief Interface for all ODE integrators.
   */
  template <typename V>
  class Integrator {
    public:
      using Vector = V;
      using Number = typename Vector::Number;
      using Real = real_part_t<Number>;

      virtual ~Integrator() {};

      /**
        @brief The current value of the dependent variable.

        This variable is usually denoted by \f$ y \f$ and contains the
        desired solution.
      */
      virtual const Vector& current_value() const = 0;

      /**
        @brief The current value of the independent variable.

        This variable is usually denoted by \f$ t \f$ for time.
      */
      virtual Real current_argument() const = 0;

      /**
        @brief Integrate the ODE from Integrator::current_argument to t1.

        @param[in] t1 The endpoint of the integration.
      */
      virtual void integrate(Real t1) = 0;

      /**
        @brief Integrate to time t1 and store result in y1.
      */
      [[deprecated]]
      void integrate(Vector& y1, real_part_t<Number> t1) {
        integrate(t1);
        y1.assign(current_value());
      }
  };

}

#endif
