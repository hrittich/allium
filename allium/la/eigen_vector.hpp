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

#ifndef ALLIUM_LA_EIGEN_VECTOR_HPP
#define ALLIUM_LA_EIGEN_VECTOR_HPP

#include <allium/util/extern.hpp>
#include "vector.hpp"
#include <Eigen/Core>

namespace allium {
  template <typename N>
  class EigenVectorStorage final
    : public VectorStorageTrait<EigenVectorStorage<N>, N>
  {
    public:
      template <typename> friend class LocalSlice;

      using BaseVector = Eigen::Matrix<N, Eigen::Dynamic, 1>;
      using typename VectorStorage<N>::Number;
      using Real = real_part_t<N>;

      explicit EigenVectorStorage(VectorSpec spec);
      EigenVectorStorage(const EigenVectorStorage& other);

      using VectorStorageTrait<EigenVectorStorage, N>::operator+=;
      EigenVectorStorage& operator+=(const EigenVectorStorage<N>& rhs);

      EigenVectorStorage& operator*=(const N& factor) override;

      void add_scaled(N factor, const EigenVectorStorage& other);

      using VectorStorageTrait<EigenVectorStorage, N>::dot;
      N dot(const EigenVectorStorage& rhs) const;
      Real l2_norm() const override;

      BaseVector& native() { return vec; }
      const BaseVector& native() const { return vec; }

    protected:
      Number* aquire_data_ptr() override;
      void release_data_ptr(Number* data) override;

    private:
      BaseVector vec;

      VectorStorage<N>* allocate_like() const& override {
        return new EigenVectorStorage(this->spec());
      }

      Cloneable* clone() const& override {
        return new EigenVectorStorage(*this);
      }
  };

  template <typename N>
  using EigenVector = VectorBase<EigenVectorStorage<N>>;

  #define ALLIUM_EIGEN_VECTOR_DECL(T, N) \
    T class EigenVectorStorage<N>; \
    T class VectorBase<EigenVectorStorage<N>>;
  ALLIUM_EXTERN(ALLIUM_EIGEN_VECTOR_DECL)
}

#endif
