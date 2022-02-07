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

#ifndef ALLIUM_ODE_RUNGE_KUTTA_4_HPP
#define ALLIUM_ODE_RUNGE_KUTTA_4_HPP

#include <memory>
#include "explicit_integrator.hpp"

namespace allium {

  /// @ingroup ode
  /// @{

  /**
   @brief The classical explicit Runge-Kutta method of order 4.
   */
  template <typename V>
  class RungeKutta4
    : public ExplicitIntegrator<V>
  {
    public:
      using Vector = V;
      using typename ExplicitIntegrator<V>::Number;
      using typename ExplicitIntegrator<V>::Real;
      using typename ExplicitIntegrator<V>::F;

      RungeKutta4(Real dt = 1e-3)
        : m_dt(dt) {}

      void dt(Real v) { m_dt = v; }

      void setup(F f) override { m_f = f; }
      void initial_value(Real t0, const Vector& y0) override {
        m_t_cur = t0;
        m_y_cur = clone(y0);
      }

      const Vector& current_value() const override { return *m_y_cur; }
      Real current_argument() const override { return m_t_cur; }

      using ExplicitIntegrator<V>::integrate;
      void integrate(Real t1) override;

    private:
      Real m_dt;
      Real m_t_cur;
      std::unique_ptr<Vector> m_y_cur;
      F m_f;
  };

  template <typename V>
  void RungeKutta4<V>::integrate(Real t1) {
    auto aux1 = allocate_like(*m_y_cur);

    auto k1 = allocate_like(*m_y_cur);
    auto k2 = allocate_like(*m_y_cur);
    auto k3 = allocate_like(*m_y_cur);
    auto k4 = allocate_like(*m_y_cur);

    while (m_t_cur < t1) {
      Real t_new = std::min(t1, m_t_cur+m_dt);
      Real h = t_new - m_t_cur;

      Real t_mid = m_t_cur + h/2;

      // k1 = f(t_cur, y1)
      m_f(*k1, m_t_cur, *m_y_cur);

      // k2 = f(t_mid, y1 + 0.5 * h * k1)
      aux1->assign(*m_y_cur);
      aux1->add_scaled(0.5 * h, *k1);
      m_f(*k2, t_mid, *aux1);

      // k3 = f(t_mid, y1 + 0.5 * h * k2)
      aux1->assign(*m_y_cur);
      aux1->add_scaled(0.5 * h, *k2);
      m_f(*k3, t_mid, *aux1);

      // k4 = f(t_new, y1 + h * k3)
      aux1->assign(*m_y_cur);
      aux1->add_scaled(h, *k3);
      m_f(*k4, t_new, *aux1);

      // y1 = y1 + 1/6 (k1 + 2*k2 + 2*k3 + k4)
      m_y_cur->add_scaled(h*1.0/6, *k1);
      m_y_cur->add_scaled(h*2.0/6, *k2);
      m_y_cur->add_scaled(h*2.0/6, *k3);
      m_y_cur->add_scaled(h*1.0/6, *k4);

      m_t_cur = t_new;
    }
  }

  /// @}}}
}

#endif
