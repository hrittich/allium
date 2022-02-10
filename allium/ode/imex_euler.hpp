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
#include <allium/la/cg.hpp>
#include "imex_integrator.hpp"

namespace allium {

  /// @ingroup ode
  /// @{

  /**
   @brief Base for the IMEX Euler integrator.
   */
  template <typename N>
  class ImexEulerBase {
    public:
      using Vector = VectorStorage<N>;
      using Number = N;
      using Real = real_part_t<N>;

      virtual ~ImexEulerBase() {}

      void dt(real_part_t<Number> dt) { m_dt = dt; }

    protected:
      void integrate(Real t1);

      real_part_t<Number> m_t_cur;
      std::unique_ptr<Vector> m_y_cur;
    private:
      virtual void apply_f_ex(Vector& out, Real t, const Vector& in) = 0;
      virtual void solve_implicit(VectorStorage<Number>& out,
                          Real t,
                          Number a,
                          const VectorStorage<Number>& r,
                          InitialGuess initial_guess) = 0;

      Real m_dt;
  };

  /**
   @brief IMEX Euler time integration scheme.
   */
  template <typename V>
  class ImexEuler
    : public ImexEulerBase<typename V::Number>,
      public ImexIntegrator<V>
  {
    public:
      using Vector = V;
      using typename ImexIntegrator<V>::Number;
      using typename ImexIntegrator<V>::ExplicitF;
      using typename ImexIntegrator<V>::ImplicitF;
      using typename ImexIntegrator<V>::ImplicitSolve;
      using Real = real_part_t<Number>;
      using ImexEulerBase<Number>::integrate;

      ImexEuler() {}

      void setup(ExplicitF f_ex, ImplicitF f_im, ImplicitSolve solve_im) override {
        m_f_ex = f_ex;
        m_solve_im = solve_im;
      }

      void initial_value(real_part_t<Number> t0, const Vector& y0) override {
        this->m_t_cur = t0;
        this->m_y_cur = clone(y0);
      }

      const Vector& current_value() const override {
        return static_cast<const Vector&>(*this->m_y_cur);
      }

      Real current_argument() const override { return this->m_t_cur; }

      using ImexIntegrator<Vector>::integrate;
      void integrate(real_part_t<Number> t1) override {
        ImexEulerBase<Number>::integrate(t1);
      }

    private:
      ExplicitF m_f_ex;
      ImplicitSolve m_solve_im;

      void apply_f_ex(VectorStorage<Number>& out, Real t, const VectorStorage<Number>& in) override {
        m_f_ex(static_cast<Vector&>(out), t, static_cast<const Vector&>(in));
      }
      void solve_implicit(VectorStorage<Number>& out,
                          Real t,
                          Number a,
                          const VectorStorage<Number>& r,
                          InitialGuess initial_guess) override {
        m_solve_im(static_cast<Vector&>(out),
                   t,
                   a,
                   static_cast<const Vector&>(r),
                   initial_guess);
      }
  };

  /// @}

  #define ALLIUM_IMEX_EULER_DECL(extern, N) \
    extern template class ImexEulerBase<N>;
  ALLIUM_EXTERN_N(ALLIUM_IMEX_EULER_DECL)
}

