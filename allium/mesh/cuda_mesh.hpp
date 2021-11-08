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

#ifndef ALLIUM_MESH_CUDA_MESH_HPP
#define ALLIUM_MESH_CUDA_MESH_HPP

#include <allium/config.hpp>
#ifdef ALLIUM_USE_CUDA

#include "cuda_narray.hpp"
#include <allium/la/vector_storage.hpp>

namespace allium {

  template <typename N, int D>
  class CudaMesh
  {};

  template <typename N>
  class CudaMesh<N, 2>
    : public VectorStorageTrait<CudaMesh<N, 2>, N>
  {
    public:
      using typename VectorStorage<N>::Number;
      using Real = real_part_t<N>;

      /** Creates a mesh with the givin number of elements that is "as square
       as possible. */
      CudaMesh(VectorSpec spec);

      explicit CudaMesh(Range<2> range, int ghost_width);
      CudaMesh(const CudaMesh& other);


      ~CudaMesh() override;

      using VectorStorageTrait<CudaMesh, N>::operator+=;
      CudaMesh& operator+=(const CudaMesh& rhs);

      CudaMesh& operator*=(const N& factor) override;

      void add_scaled(N factor, const CudaMesh& other);

      using VectorStorageTrait<CudaMesh, N>::dot;
      N dot(const CudaMesh& rhs) const;
      Real l2_norm() const override;

      void fill(N value) override;

      /** Apply a five point stencil */
      void apply_five_point(CudaMesh& out, std::array<Number, 5> coeff) const;

      void fill_ghost_points(Number value);

      CudaNArray<N, 2>& narray() { return m_device_data; }
      const CudaNArray<N, 2>& narray() const { return m_device_data; }

      int ghost_width() const { return m_ghost_width; }
      Range<2> range() const { return m_range; }

      Range<2> local_range() const { return m_range; }

      Range<2> ghosted_range() const;

      void copy_ghosted_to(LocalMesh<Number, 2> &mesh) const;
      void copy_ghosted_from(const LocalMesh<Number, 2> &mesh);
    protected:
      Number* aquire_data_ptr() override;
      void release_data_ptr(Number* data) override;

    private:
      CudaNArray<N, 2> m_device_data;
      Range<2> m_range;
      int m_ghost_width;

      VectorStorage<N>* allocate_like() const& override {
        return new CudaMesh(m_range, m_ghost_width);
      }

      Cloneable* clone() const& override {
        return new CudaMesh(*this);
      }
  };

}

#endif
#endif
