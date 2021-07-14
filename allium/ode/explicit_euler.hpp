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

#ifndef ALLIUM_ODE_EXPLICIT_EULER_HPP
#define ALLIUM_ODE_EXPLICIT_EULER_HPP

#include "explicit_integrator.hpp"

namespace allium {

  /**
   @brief The explicit Euler time integrator.
   */
  template <typename V>
  class ExplicitEuler
    : public ExplicitIntegrator<V>
  {
    public:
      using Vector = V;
      using typename ExplicitIntegrator<V>::Number;
      using typename ExplicitIntegrator<V>::Real;
      using typename ExplicitIntegrator<V>::F;

      ExplicitEuler(Real dt = 1e-3)
        : m_dt(dt) {}

      void dt(Real v) { m_dt = v; }

      void setup(F f) override { m_f = f; }
      void initial_values(Real t0, const Vector& y0) override {
        m_t0 = t0;
        m_y0 = allocate_like(y0);
        m_y0->assign(y0);
      }
      void integrate(Vector& y1, Real t1) override;

    private:
      Real m_dt;
      Real m_t0;
      std::unique_ptr<Vector> m_y0;
      F m_f;
  };

  template <typename V>
  void ExplicitEuler<V>::integrate(Vector& y1, Real t1) {
    auto aux1 = allocate_like(*m_y0);

    Real t_old = m_t0;
    y1.assign(*m_y0);

    while (t_old < t1) {
      Real t_new = std::min(t1, t_old+m_dt);
      Real h = t_new - t_old;

      // y1 = y1 + h f(t_old, y1)
      m_f(*aux1, t_old, y1);
      y1.add_scaled(h, *aux1);

      t_old = t_new;
    }
  }

}

#endif
