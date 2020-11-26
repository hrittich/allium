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

#include <functional>
#include <memory>
#include <allium/la/linear_operator.hpp>
#include <allium/util/numeric.hpp>

namespace allium {

  /**
   Interface for implicit-explicit time-stepping schemes have a linear
   implicit part.

   *Experimental interface*
   */
  template <typename V>
  class LinearImexIntegrator {
    public:
      using Vector = V;
      using Number = typename V::Number;
      using ExplicitF = std::function<void(Vector&,
                                           real_part_t<Number>,
                                           const Vector&)>;
      using ImplicitF = std::function<void(Vector&,
                                           real_part_t<Number>,
                                           const Vector&)>;

      virtual ~LinearImexIntegrator() {}

      virtual void setup(ExplicitF f_ex, ImplicitF f_impl) = 0;
      virtual void initial_values(real_part_t<Number> t0, const Vector& y0) = 0;
  };

}
