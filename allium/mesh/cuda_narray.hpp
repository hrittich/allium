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

#ifndef ALLIUM_MESH_CUDA_NARRAY_HPP
#define ALLIUM_MESH_CUDA_NARRAY_HPP

#include <allium/config.hpp>
#ifdef ALLIUM_USE_CUDA

#include "range.hpp"
#include "local_mesh.hpp"

namespace allium {

  template <typename N, int D>
  class CudaNArray
  {};

  template <typename N>
  class CudaNArray<N, 2> final
  {
    public:
      using Number = N;

      CudaNArray();
      CudaNArray(Point<int, 2> shape);

      CudaNArray(const CudaNArray& other);
      CudaNArray& operator= (const CudaNArray&) = delete;

      ~CudaNArray();

      void resize(Point<int, 2> shape);

      void copy_to(Number* data) const;
      void copy_to(LocalMesh<N, 2>& other) const;

      void copy_from(const Number* data);
      void copy_from(const LocalMesh<N, 2>& other);

      Point<int, 2> shape() const { return m_shape; }

      Number* ptr() { return m_device_data; }
      const Number* ptr() const { return m_device_data; }

      size_t pitch() const { return m_pitch; }
    private:
      Point<int, 2> m_shape;
      Number* m_device_data;
      size_t m_pitch;
  };

}

#endif
#endif
