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

#include "cuda_mesh.hpp"
#include <allium/la/cuda_util.hpp>
#include <allium/la/cuda_vector.hpp>
#include "cuda_mesh_algorithm.hpp"
#include <iostream>
#include <iterator>

namespace allium {

template <typename N>
static MeshDataLayout data_layout(const CudaMesh<N, 2>& mesh) {
  return MeshDataLayout {
    .pitch = mesh.narray().pitch(),
    .start = { mesh.ghost_width(), mesh.ghost_width() },
    .end = { mesh.range().shape()[0] + mesh.ghost_width(),
             mesh.range().shape()[1] + mesh.ghost_width() }
  };
}

static Range<2> default_shape(VectorSpec spec)
{
  allium_assert(spec.local_size() == spec.global_size());

  int divider = static_cast<int>(round(sqrt(spec.local_size())));
  while ((spec.local_size() % divider) != 0) {
    divider--;
  }

  auto const rows = divider;
  auto const cols = static_cast<int>(spec.local_size() / divider);

  return Range<2>(Point<int, 2>::full(0), { rows, cols });
}

template <typename N>
CudaMesh<N, 2>::CudaMesh(VectorSpec spec)
  : CudaMesh(default_shape(spec), 0)
{}

template <typename N>
CudaMesh<N, 2>::CudaMesh(Range<2> range, int ghost_width)
  : VectorStorageTrait<CudaMesh<N, 2>, N>(
      VectorSpec(Comm::world(), range.size(), range.size())),
    m_device_data(range.shape() + Point<int, 2>::full(ghost_width*2)),
    m_range(range),
    m_ghost_width(ghost_width)
{}

template <typename N>
CudaMesh<N, 2>::CudaMesh(const CudaMesh& other)
  : VectorStorageTrait<CudaMesh<N, 2>, N>(other.spec()),
    m_device_data(other.m_device_data),
    m_range(other.m_range),
    m_ghost_width(other.m_ghost_width)
{}

template <typename N>
CudaMesh<N, 2>::~CudaMesh()
{}

template <typename N>
auto CudaMesh<N, 2>::operator+=(const CudaMesh& rhs) -> CudaMesh&
{
  const auto layout = data_layout(*this);
  cuda_mesh_map(m_device_data.ptr(),
                layout,
                cuda_op::Sum<N>(),
                rhs.m_device_data.ptr());

  return *this;
}

template <typename N>
auto CudaMesh<N, 2>::operator*=(const N& factor) -> CudaMesh&
{
  const auto layout = data_layout(*this);
  cuda_mesh_map(m_device_data.ptr(),
                layout,
                cuda_op::MulBy<N>(factor));

  return *this;
}

template <typename N>
void CudaMesh<N, 2>::add_scaled(N factor, const CudaMesh& other)
{
  const auto layout = data_layout(*this);
  cuda_mesh_map(m_device_data.ptr(),
                layout,
                cuda_op::AddScaled<N>(factor),
                other.m_device_data.ptr());
}

template <typename N>
N CudaMesh<N, 2>::dot(const CudaMesh& rhs) const
{
  const auto layout = data_layout(*this);
  return cuda_mesh_map_reduce<N, cuda_op::Sum<N>, cuda_op::Prod<N>>
                             (layout, m_device_data.ptr(), rhs.m_device_data.ptr());
}

template <typename N>
auto CudaMesh<N, 2>::l2_norm() const -> Real
{
  const auto layout = data_layout(*this);
  return sqrt(cuda_mesh_map_reduce<N, cuda_op::Sum<N>, cuda_op::Square<N>>(
                layout, m_device_data.ptr()));
}

template <typename N>
void CudaMesh<N, 2>::apply_five_point(CudaMesh& out, std::array<Number, 5> coeff) const
{
  cuda_mesh_five_point(out.m_device_data.ptr(),
                       data_layout(*this),
                       coeff.data(),
                       m_device_data.ptr());
}

template <typename N>
void CudaMesh<N,2>::fill_ghost_points(Number value)
{
  const Point<int, 2> shape = m_range.shape();

  const auto ghosted_start = Point<int,2>::full(0);
  const auto ghosted_end = shape + Point<int,2>::full(m_ghost_width*2);
  const auto inner_start = Point<int,2>::full(m_ghost_width);
  const auto inner_end = shape + Point<int,2>::full(m_ghost_width);

  MeshDataLayout top = {
    .pitch = m_device_data.pitch(),
    .start = { ghosted_start[0], ghosted_start[1] },
    .end = { ghosted_end[0], inner_start[1] }
  };

  cuda_mesh_fill(m_device_data.ptr(), top, value);

  MeshDataLayout bottom = {
    .pitch = m_device_data.pitch(),
    .start = { ghosted_start[0], inner_end[1] },
    .end = { ghosted_end[0], ghosted_end[1] }
  };

  cuda_mesh_fill(m_device_data.ptr(), bottom, value);

  MeshDataLayout left = {
    .pitch = m_device_data.pitch(),
    .start = { ghosted_start[0], ghosted_start[1] },
    .end = { inner_start[0], ghosted_end[1]}
  };

  cuda_mesh_fill(m_device_data.ptr(), left, value);

  MeshDataLayout right = {
    .pitch = m_device_data.pitch(),
    .start = { inner_end[0], ghosted_start[1] },
    .end = { ghosted_end[0], ghosted_end[1]}
  };

  cuda_mesh_fill(m_device_data.ptr(), right, value);

}

template <typename N>
auto CudaMesh<N, 2>::aquire_data_ptr() -> Number*
{
  LocalMesh<N, 2> host(ghosted_range());

  m_device_data.copy_to(host);

  Number* data = new Number[m_device_data.shape().prod()];
  int i=0;
  for (const auto p : m_range) {
    data[i] = host[p];
    i++;
  }

  return data;
}

template <typename N>
void CudaMesh<N, 2>::release_data_ptr(Number* data)
{
  LocalMesh<N, 2> host(ghosted_range());

  int i=0;
  for (const auto p : m_range) {
    host[p] = data[i];
    i++;
  }
  delete [] data;

  m_device_data.copy_from(host);
}

template <typename N>
Range<2> CudaMesh<N, 2>::ghosted_range() const
{
  return Range<2>(m_range.begin_pos() - Point<int, 2>::full(m_ghost_width),
                  m_range.end_pos() + Point<int, 2>::full(m_ghost_width));
}

template <typename N>
void CudaMesh<N, 2>::copy_ghosted_to(LocalMesh<Number, 2> &mesh) const
{
  allium_assert(mesh.range() == ghosted_range());

  m_device_data.copy_to(mesh);
}


template <typename N>
void CudaMesh<N, 2>::copy_ghosted_from(const LocalMesh<Number, 2> &mesh)
{
  allium_assert(mesh.range() == ghosted_range());

  m_device_data.copy_from(mesh);
}

template class CudaMesh<double, 2>;
template class CudaMesh<float, 2>;

}

