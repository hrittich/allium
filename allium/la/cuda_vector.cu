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

#include "cuda_vector.hpp"
#include "cuda_util.hpp"
#include <allium/util/assert.hpp>

#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <cassert>

namespace allium {

  template <typename T>
  CudaArray<T>::CudaArray(size_t element_count)
    : m_ptr(nullptr)
  {
    cudaError_t err;
    if (element_count > 0) {
      err = cudaMalloc(&m_ptr, element_count * sizeof(T));
      cuda_check_status(err, "allocate cuda array");
    }
  }

  template <typename T>
  CudaArray<T>::CudaArray(CudaArray&& other)
    : m_ptr(other.m_ptr)
  {
    other.m_ptr = nullptr;
  }

  template <typename T>
  auto CudaArray<T>::operator= (CudaArray&& other) -> CudaArray&
  {
    if (m_ptr != nullptr) {
      cudaFree(m_ptr);
    }
    m_ptr = other.m_ptr;
    other.m_ptr = nullptr;

    return *this;
  }

  template <typename T>
  CudaArray<T>::~CudaArray() {
    if (m_ptr != nullptr) {
      cudaFree(m_ptr);
    }
  }

  template <typename T>
  void CudaArray<T>::resize(size_t element_count)
  {
    if (m_ptr != nullptr) {
      cudaFree(m_ptr);
      m_ptr = nullptr;
    }
    if (element_count > 0) {
      cudaMalloc(&m_ptr, element_count * sizeof(T));
    }
  }

  template <typename N, typename ReduceOp, typename MapOp, typename ...Args>
  N cuda_map_reduce(int n, Args ...a)
  {
    using Number = N;
    cudaError_t err;

    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count < 1)
      throw std::runtime_error("No CUDA device found");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    const int grid_size = prop.multiProcessorCount * 2;

    CudaArray<Number> d_partial(grid_size);

    const auto block_size = partial_map_reduce_block_size;
    partial_map_reduce<Number, ReduceOp, MapOp>
                      <<<grid_size, block_size>>>
                      (d_partial.ptr(), n, a...);
    cuda_check_last_status("execute partial_vec_dot");


    CudaArray<Number> d_result(1);
    partial_map_reduce<Number, ReduceOp, cuda_op::Id<N>>
                      <<<1, block_size>>>
                      (d_result.ptr(), grid_size, d_partial.ptr());
    cuda_check_last_status("execute partial_vec_sum");

    Number h_result;
    err = cudaMemcpy(&h_result, d_result.ptr(), sizeof(Number), cudaMemcpyDeviceToHost);
    cuda_check_status(err, "copy result");

    return h_result;
  }

  template <typename N, typename Op, typename ...Args>
  __global__ void map_kernel(Op op, int n, N* a, Args ...args) {
    int i_thread = threadIdx.x + blockDim.x * blockIdx.x;

    if (i_thread < n) {
      a[i_thread] = op(a[i_thread], args[i_thread]...);
    }
  }

  template <typename N, typename Op, typename ...Args>
  void cuda_map(Op op, int n, N* a, Args ...args)
  {
    const auto block_size = partial_map_reduce_block_size;
    int grid_size = (n + block_size - 1) / block_size;

    map_kernel<<<grid_size, block_size>>>(op, n, a, args...);
    cuda_check_last_status("execute map_kernel");
  }

  

// === CudaVector ============================================================

  template <typename N>
  CudaVector<N>::CudaVector(VectorSpec spec)
    : VectorStorageTrait<CudaVector, N>(spec),
      m_device_data(spec.local_size())
  {
    allium_assert(spec.local_size() == spec.global_size(),
                  "Cuda vectors are not distributed");
  }

  template <typename N>
  CudaVector<N>::CudaVector(const CudaVector& other)
    : VectorStorageTrait<CudaVector, N>(other.spec()),
      m_device_data(other.spec().local_size())
  {
    cudaError_t err;
    err = cudaMemcpy(m_device_data.ptr(), other.m_device_data.ptr(),
                     other.spec().local_size() * sizeof(Number),
                     cudaMemcpyDeviceToDevice);
    cuda_check_status(err, "copy constructor");
  }

  template <typename N>
  CudaVector<N>::~CudaVector() {}

  template <typename N>
  auto CudaVector<N>::operator+=(const CudaVector<N>& rhs) -> CudaVector&
  {
    size_t n = this->spec().local_size();
    cuda_map(cuda_op::Sum<N>(), n, m_device_data.ptr(), rhs.m_device_data.ptr());

    return *this;
  }

  template <typename N>
  auto CudaVector<N>::operator*=(const N& factor) -> CudaVector&
  {
    size_t n = this->spec().local_size();
    cuda_map(cuda_op::MulBy<N>(factor), n, m_device_data.ptr());

    return *this;
  }

  template <typename N>
  void CudaVector<N>::add_scaled(N factor, const CudaVector& other)
  {
    size_t n = this->spec().local_size();

    cuda_map(cuda_op::AddScaled<N>(factor), n, m_device_data.ptr(), other.m_device_data.ptr());
  }

  template <typename N>
  N CudaVector<N>::dot(const CudaVector& rhs) const
  {
    size_t n = this->spec().local_size();

    return cuda_map_reduce<Number, cuda_op::Sum<N>, cuda_op::Prod<N>>(n, m_device_data.ptr(), rhs.m_device_data.ptr());
  }

  template <typename N>
  auto CudaVector<N>::l2_norm() const -> Real
  {
    size_t n = this->spec().local_size();

    return sqrt(cuda_map_reduce<Number, cuda_op::Sum<N>, cuda_op::Square<N>>(n, m_device_data.ptr()));
  }

  template <typename N>
  auto CudaVector<N>::aquire_data_ptr() -> Number*
  {
    cudaError_t err;
    size_t n = this->spec().local_size();
    Number* data = new Number[n];

    err = cudaMemcpy(data, m_device_data.ptr(),
                     this->spec().local_size() * sizeof(Number),
                     cudaMemcpyDeviceToHost);
    cuda_check_status(err, "copy from device to host");

    return data;
  }

  template <typename N>
  void CudaVector<N>::release_data_ptr(Number* data)
  {
    cudaError_t err;

    err = cudaMemcpy(m_device_data.ptr(), data,
                     this->spec().local_size() * sizeof(Number),
                     cudaMemcpyHostToDevice);
    delete [] data;

    cuda_check_status(err, "copy from host to device");
  }

  //ALLIUM_NOEXTERN_N(ALLIUM_CUDA_VECTOR_DECL)
  ALLIUM_CUDA_VECTOR_DECL(,float)
  ALLIUM_CUDA_VECTOR_DECL(,double)
}

