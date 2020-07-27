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

#ifndef ALLIUM_LA_CG_HPP
#define ALLIUM_LA_CG_HPP

#include <allium/util/extern.hpp>
#include "vector.hpp"
#include "sparse_matrix.hpp"

namespace allium {
  template <typename N>
    Vector<N> cg_(SparseMatrix<N> mat, Vector<N> rhs, real_part_t<N> tol = 1e-6);

  template <typename Mat, typename... Args>
    Vector<typename Mat::Number> cg(Mat mat, Args&&... args)
    {
      return cg_<typename Mat::Number>(mat, std::forward<Args>(args)...);
    }

  #define ALLIUM_CG_DECL(T, N) \
    T Vector<N> cg_<N>(SparseMatrix<N>, Vector<N>, real_part_t<N>);
  ALLIUM_EXTERN(ALLIUM_CG_DECL)
}

#endif
