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

#ifndef ALLIUM_LA_LINEAR_OPERATOR_HPP
#define ALLIUM_LA_LINEAR_OPERATOR_HPP

#include <memory>
#include <allium/util/cloning_ptr.hpp>

namespace allium {
  template <typename V>
  class LinearOperator {
    public:
      using Vector = V;
      using Number = typename Vector::Number;
      using Real = typename Vector::Real;

      virtual ~LinearOperator() {}

      virtual void apply(Vector& result,
                         const Vector& arg) = 0;
  };

  template <typename V, typename F>
  class FunctorLinearOperator final : public LinearOperator<V> {
    public:
      using typename LinearOperator<V>::Vector;

      FunctorLinearOperator(const F& fun) : m_fun(fun) {};
      FunctorLinearOperator(F&& fun) : m_fun(std::move(fun)) {};

      void apply(Vector& result, const Vector& arg) override {
        m_fun(result, arg);
      }
    private:
      F m_fun;
  };

  template <typename V, typename F>
    FunctorLinearOperator<V, typename std::remove_reference<F>::type>
    make_linear_operator(F&& f) {
      return FunctorLinearOperator<V, F>(std::forward<F>(f));
    }
}

#endif
