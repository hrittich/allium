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

#include "vector.hpp"

#include "eigen_vector.hpp"
#include "petsc_vector.hpp"

namespace allium {

  #if defined(ALLIUM_DEFAULT_BACKEND_EIGEN)
    template <typename N>
      Vector<N> make_vector(VectorSpec spec) {
        return EigenVector<N>(spec);
      }
  #elif defined(ALLIUM_DEFAULT_BACKEND_PETSC)
    template <typename N>
      Vector<N> make_vector_(VectorSpec spec) {
        throw std::logic_error("Not implemented");
      }

    template <>
      Vector<PetscScalar> make_vector_(VectorSpec spec) {
        return PetscVector(spec);
      }

    template <typename N>
      Vector<N> make_vector(VectorSpec spec) {
        return make_vector_<N>(spec);
      }
  #else
    #error "No default linear algebra backend specified."
  #endif
}
