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

#include <allium/la/default.hpp>
#include <allium/ode/explicit_euler.hpp>
#include <allium/ode/runge_kutta_4.hpp>

#include <gtest/gtest.h>

using namespace allium;

TEST(ExplicitIntegrator, Euler)
{
  using Number = double;
  using Real = double;
  using Vector = DefaultVector<Number>;

  Real alpha = 1.0;
  Real dt = 1e-4;

  ExplicitEuler<Vector> integrator;

  VectorSpec spec(Comm::world(), 1, 1);
  Vector y0(spec);
  Vector y1(spec);
  Vector y2(spec);
  fill(y0, Number(1.0));

  auto f = [alpha](Vector& out, double t, const Vector& in) {
    out.assign(in);
    out *= alpha;
  };

  integrator.setup(f);

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

TEST(ExplicitIntegrator, RungeKutta4)
{
  using Number = double;
  using Real = double;
  using Vector = DefaultVector<Number>;

  Real dt = 1e-2;

  RungeKutta4<Vector> integrator;

  VectorSpec spec(Comm::world(), 1, 1);
  Vector y0(spec);
  Vector y1(spec);
  Vector y2(spec);
  fill(y0, Number(1.0));

  auto f = [](Vector& out, double t, const Vector& in) {
    out.assign(in);
    out *= cos(t);
  };

  integrator.setup(f);

  integrator.initial_value(0, y0);
  integrator.dt(dt);
  integrator.integrate(y1, dt);

  integrator.initial_value(0, y0);
  integrator.dt(dt/2);
  integrator.integrate(y2, dt/2);

  {
    auto y1_loc = local_slice(y1);
    auto y2_loc = local_slice(y2);

    auto err1 = std::abs(y1_loc[0] - exp(sin(dt)));
    auto err2 = std::abs(y2_loc[0] - exp(sin(dt/2)));

    // compute the consistency order
    auto order = log(err1 / err2) / log(2.0) - 1;
    #if 0
    std::cout
      << order
      << " (e₁ = " << err1 << ", e₂ = " << err2 << ")"
      << std::endl;
    #endif

    EXPECT_NEAR(order, 4, 1e-2);
  }
}

TEST(ExplicitIntegrator, RungeKutta4DrivenExp)
{
  using Number = double;
  using Real = double;
  using Vector = DefaultVector<Number>;

  Real dt = 2e-2;
  double alpha = 0.1;

  RungeKutta4<Vector> integrator;

  VectorSpec spec(Comm::world(), 1, 1);
  Vector y0(spec);
  Vector y1(spec);
  Vector y2(spec);
  fill(y0, Number(0.0));

  auto f = [alpha](Vector& out, double t, const Vector& in) {
    Vector aux1(in.spec());
    fill(aux1, sin(t));

    out.assign(in);
    out *= alpha;
    out += aux1;
  };

  auto soln = [alpha](double t) -> double {
    return 1.0 / (1 + alpha*alpha) * (exp(alpha*t) - cos(t) - alpha * sin(t));
  };

  integrator.setup(f);

  integrator.initial_value(0, y0);
  integrator.dt(dt);
  integrator.integrate(y1, dt);

  integrator.initial_value(0, y0);
  integrator.dt(dt/2);
  integrator.integrate(y2, dt/2);

  {
    auto y1_loc = local_slice(y1);
    auto y2_loc = local_slice(y2);

    auto err1 = std::abs(y1_loc[0] - soln(dt));
    auto err2 = std::abs(y2_loc[0] - soln(dt/2));

    // compute the consistency order
    auto order = log(err1 / err2) / log(2.0) - 1;
    #if 1
    std::cout
      << order
      << " (e₁ = " << err1 << ", e₂ = " << err2 << ")"
      << std::endl;
    #endif

    EXPECT_NEAR(order, 4, 1e-1);
  }
}


