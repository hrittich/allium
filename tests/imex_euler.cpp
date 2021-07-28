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

#include <allium/util/memory.hpp>
#include <allium/la/default.hpp>
#include <allium/ode/imex_euler.hpp>

#include <gtest/gtest.h>

using namespace allium;

TEST(ImexEuler, TestEquation)
{
  using Number = double;
  using Real = double;
  using Vector = DefaultVector<Number>;

  Real alpha = 1.0;
  Real dt = 1e-4;

  ImexEuler<Vector> integrator;

  VectorSpec spec(Comm::world(), 1, 1);
  Vector y0(spec);
  Vector y1(spec);
  Vector y2(spec);
  fill(y0, Number(1.0));

  auto f_ex = [](Vector& out, double t, const Vector& in) {
    set_zero(out);
  };

  auto f_im = [alpha](Vector& out, double t, const Vector& in) {
    out.assign(in);
    out *= alpha;
  };

  // solves y - a * f_im(t, y) = r
  auto f_solve = [alpha](Vector& out,
                         double t,
                         double a,
                         const Vector& r,
                         InitialGuess initial_guess) {
    // out = (1 / (1 - alpha * a)) * r
    out.assign(r);
    out *= (1.0 / (1 - alpha * a));
  };

  integrator.setup(f_ex, f_im, f_solve);
  integrator.initial_value(0, y0);
  integrator.dt(dt);

  integrator.integrate(y1, dt);

  integrator.initial_value(0, y0);
  integrator.dt(dt/2);
  integrator.integrate(y2, dt/2);

  {
    auto y1_loc = local_slice(y1);
    auto y2_loc = local_slice(y2);

    auto err1 = std::abs(y1_loc[0] - exp(dt));
    auto err2 = std::abs(y2_loc[0] - exp(dt/2));

    // compute the consistency order
    auto order = log(err1 / err2) / log(2.0) - 1;
    #if 0
    std::cout
      << order
      << " (e₁ = " << err1 << ", e₂ = " << err2 << ")"
      << std::endl;
    #endif

    EXPECT_NEAR(order, 1, 1e-3);
  }
}

