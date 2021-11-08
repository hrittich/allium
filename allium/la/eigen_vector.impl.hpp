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

#include "eigen_vector.hpp"

namespace allium {

  template <typename N>
    EigenVectorStorage<N>::EigenVectorStorage(VectorSpec spec)
      : VectorStorageTrait<EigenVectorStorage<N>, N>(spec),
        vec(spec.global_size())
    {
      if (spec.comm().size() != 1) {
        throw std::logic_error("Objects of type EigenVector cannot be distributed.");
      }
    }

  template <typename N>
    EigenVectorStorage<N>::EigenVectorStorage(const EigenVectorStorage& other)
      : VectorStorageTrait<EigenVectorStorage, N>(other.spec()),
        vec(other.vec)
    {}

  template <typename N>
  auto EigenVectorStorage<N>::operator+=(const EigenVectorStorage<N>& rhs) -> EigenVectorStorage&
  {
    vec += rhs.vec;
    return *this;
  }

  template <typename N>
  auto EigenVectorStorage<N>::operator*=(const N& factor) -> EigenVectorStorage& {
    vec *= factor;
    return *this;
  }

  template <typename N>
  void EigenVectorStorage<N>::add_scaled(N factor, const EigenVectorStorage& other) {
    vec += factor * other.vec;
  }

  template <typename N>
  N EigenVectorStorage<N>::dot(const EigenVectorStorage<N>& rhs) const {
    // eigen has a dot product wich is linear in the SECOND argument
    return rhs.vec.dot(vec);
  }

  template <typename N>
    auto EigenVectorStorage<N>::l2_norm() const -> Real
  {
    return vec.norm();
  }

  template <typename N>
    void EigenVectorStorage<N>::fill(N value)
  {
    vec = BaseVector::Constant(this->spec().global_size(), value);
  }

  template <typename N>
    auto EigenVectorStorage<N>::aquire_data_ptr() -> Number*
  {
    return vec.data();
  }

  template <typename Number>
    void EigenVectorStorage<Number>::release_data_ptr(Number* data)
  {}

}
