// Copyright 2021 Hannah Rittich
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

#ifndef ALLIUM_CUDA_VECTOR_HPP
#define ALLIUM_CUDA_VECTOR_HPP

#include <allium/config.hpp>
#ifdef ALLIUM_USE_CUDA

#include "vector_storage.hpp"

namespace allium {

  /**
   * Stores an array on the GPU. Memory is automatically managed.
   */
  template <typename T>
  class CudaArray final {
    public:
      CudaArray(size_t element_count=0);

      CudaArray(const CudaArray&) = delete;
      CudaArray& operator= (const CudaArray&) = delete;

      CudaArray(CudaArray&& other);
      CudaArray& operator= (CudaArray&& other);

      ~CudaArray();

      void resize(size_t element_count);

      T* ptr() { return m_ptr; }
      const T* ptr() const { return m_ptr; }
    private:
      T* m_ptr;
  };

  /**
   * Vector which performs all computations on the GPU.
   */
  template <typename N>
  class CudaVector final
    : public VectorStorageTrait<CudaVector<N>, N>
  {
    public:
      template <typename> friend class LocalSlice;

      using typename VectorStorage<N>::Number;
      using Real = real_part_t<N>;

      explicit CudaVector(VectorSpec spec);
      CudaVector(const CudaVector& other);

      ~CudaVector() override;

      using VectorStorageTrait<CudaVector, N>::operator+=;
      CudaVector& operator+=(const CudaVector<N>& rhs);

      CudaVector& operator*=(const N& factor) override;

      void add_scaled(N factor, const CudaVector& other);

      using VectorStorageTrait<CudaVector, N>::dot;
      N dot(const CudaVector& rhs) const;
      Real l2_norm() const override;

    protected:
      Number* aquire_data_ptr() override;
      void release_data_ptr(Number* data) override;

    private:
      CudaArray<N> m_device_data;

      VectorStorage<N>* allocate_like() const& override {
        return new CudaVector(this->spec());
      }

      Cloneable* clone() const& override {
        return new CudaVector(*this);
      }
  };

  #define ALLIUM_CUDA_VECTOR_DECL(extern, N) \
    extern template class CudaArray<N>; \
    extern template class CudaVector<N>;
  ALLIUM_EXTERN_N(ALLIUM_CUDA_VECTOR_DECL)
}

#endif
#endif
