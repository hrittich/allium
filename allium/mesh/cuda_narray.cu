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

#include "cuda_narray.hpp"

#include <allium/la/cuda_util.hpp>

namespace allium {

  template <typename N>
  CudaNArray<N, 2>::CudaNArray()
    : m_device_data(nullptr),
      m_shape({0,0})
  {}

  template <typename N>
  CudaNArray<N, 2>::CudaNArray(Point<int, 2> shape)
    : m_device_data(nullptr)
  {
    resize(shape);
  }

  template <typename N>
  CudaNArray<N, 2>::CudaNArray(const CudaNArray& other)
    : m_device_data(nullptr)
  {
    resize(other.m_shape);

    cudaError_t err;
    err = cudaMemcpy2D(
      m_device_data, m_pitch,             // destination + pitch
      other.m_device_data, other.m_pitch, // source + pitch
      m_shape[1]*sizeof(Number),          // row length in bytes
      m_shape[0],                         // #rows
      cudaMemcpyDeviceToDevice);
    cuda_check_status(err, "cudaMemcpy2D on device");
  }

  template <typename N>
  CudaNArray<N, 2>::~CudaNArray()
  {
    resize({0,0});
  }

  template <typename N>
  void CudaNArray<N, 2>::resize(Point<int, 2> shape)
  {
    cudaError_t err;
    if (m_device_data != nullptr) {
      err = cudaFree(m_device_data);
      cuda_check_status(err, "cudaFree");
      m_pitch = 0;
      m_device_data = nullptr;
    }

    m_shape = shape;
    if (m_shape.all_of([](int s){ return s != 0; })) {

      // allocate to shape[1] to be the fast running index
      err = cudaMallocPitch(&m_device_data, &m_pitch,
                            m_shape[1]*sizeof(Number),
                            m_shape[0]);
      cuda_check_status(err, "cudaMallocPitch");
    }

  }

  template <typename N>
  void CudaNArray<N,2>::copy_to(Number* data) const
  {
    cudaError_t err;
    err = cudaMemcpy2D(
      data, m_shape[1]*sizeof(Number),  // destination + pitch
      m_device_data, m_pitch,         // source + pitch
      m_shape[1]*sizeof(Number),        // row length in bytes
      m_shape[0],                       // #rows
      cudaMemcpyDeviceToHost);
    cuda_check_status(err, "cudaMemcpy2D to host");
  }

  template <typename N>
  void CudaNArray<N,2>::copy_to(LocalMesh<N, 2>& other) const {
    allium_assert(m_shape == other.range().shape(), "shapes must match");

    copy_to(other.data());
  }

  template <typename N>
  void CudaNArray<N,2>::copy_from(const Number* data)
  {
    cudaError_t err;
    err = cudaMemcpy2D(
      m_device_data, m_pitch,
      data, m_shape[1]*sizeof(Number),
      m_shape[1]*sizeof(Number),
      m_shape[0],
      cudaMemcpyHostToDevice);
    cuda_check_status(err, "cudaMemcpy2D to device");
  }

  template <typename N>
  void CudaNArray<N,2>::copy_from(const LocalMesh<N, 2>& other)
  {
    allium_assert(m_shape == other.range().shape(), "shapes must match");

    copy_from(other.data());
  }

  template class CudaNArray<double, 2>;
  template class CudaNArray<float, 2>;

}
