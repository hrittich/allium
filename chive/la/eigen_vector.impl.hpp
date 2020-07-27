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

namespace chive {

  template <typename N>
    EigenVectorStorage<N>::EigenVectorStorage(VectorSpec spec)
      : VectorStorageBase<EigenVectorStorage<N>, N>(spec),
        vec(spec.global_size())
    {
      if (spec.comm().size() != 1) {
        throw std::logic_error("Objects of type EigenVector cannot be distributed.");
      }
    }

  template <typename N>
  void EigenVectorStorage<N>::add(const VectorStorage<N>& rhs)
  {
    const EigenVectorStorage* eigen_rhs
      = dynamic_cast<const EigenVectorStorage*>(&rhs);
    if (eigen_rhs != nullptr) {
      vec += eigen_rhs->vec;
    } else {
      throw std::logic_error("Not implemented");
    }
  }

  template <typename Number>
  void EigenVectorStorage<Number>::scale(const Number& factor) {
    vec *= factor;
  }

  template <typename N>
  N EigenVectorStorage<N>::dot(const VectorStorage<N>& rhs) {
    const EigenVectorStorage* eigen_rhs
      = dynamic_cast<const EigenVectorStorage*>(&rhs);
    if (eigen_rhs != nullptr) {
      // eigen has a dot product wich is linear in the FIRST argument
      return eigen_rhs->vec.dot(vec);
    } else {
      throw std::logic_error("Not implemented");
    }
  }

  template <typename Number>
    real_part_t<Number>
    EigenVectorStorage<Number>::l2_norm() const
  {
    return vec.norm();
  }

  template <typename Number>
    typename VectorStorage<Number>::Number* EigenVectorStorage<Number>::aquire_data_ptr()
  {
    return vec.data();
  }

  template <typename Number>
    void EigenVectorStorage<Number>::release_data_ptr(Number* data)
  {}

}
