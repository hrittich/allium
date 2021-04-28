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

#ifndef ALLIUM_MESH_CUDA_MESH_ALGORITHM_HPP
#define ALLIUM_MESH_CUDA_MESH_ALGORITHM_HPP

#include <allium/config.hpp>
#ifdef ALLIUM_USE_CUDA

#include <allium/la/cuda_util.hpp>
#include <allium/la/cuda_vector.hpp>
#include "cuda_narray.hpp"

namespace allium {

  struct MeshDataLayout {
    size_t pitch;     /// the pitch in the x-direction
    int start[2];     /// index where the mesh startes
    int end[2];       /// index where the mesh ends
  };

  template <typename N>
  __device__ inline N& mesh_at(N* m, const MeshDataLayout& layout, int i, int j)
  {
    N* row = (N*)((char*)m + i * layout.pitch);
    return row[j];
  }

  template <typename N, typename MapOp, typename ...Args>
  __global__ void mesh_map_kernel(N* out, MeshDataLayout layout, MapOp f, Args ...args)
  {
    using Number = N;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= layout.start[0] && i < layout.end[0]
        && j >= layout.start[1] && j < layout.end[1])
    {
      mesh_at(out, layout, i, j) = f(mesh_at(out, layout, i, j),
                                     mesh_at(args, layout, i, j)...);
    }
  }

  /**
   *
   * Warning: inefficient for large values of layout.start.
   */
  template <typename N, typename MapOp, typename ...Args>
  void cuda_mesh_map(N* out, MeshDataLayout layout, MapOp f, Args ...args)
  {
    const dim3 block_dim = { 32, 32, 1 };

    const dim3 grid_dim { (layout.end[0] + block_dim.x - 1) / block_dim.x,
                          (layout.end[1] + block_dim.y - 1) / block_dim.y,
                          1 };

    mesh_map_kernel<N, MapOp, Args...>
                   <<<grid_dim, block_dim>>>
                   (out,
                    layout,
                    f,
                    args...);
    cuda_check_last_status("mesh_map_kernel");
  }

  namespace mesh_reduction {
    const int BLOCK_ROWS = 32;
    const int BLOCK_COLS = 32;
  }

  template <typename N, typename ReduceOp, typename MapOp, typename ...Args>
  __global__ void
    mesh_map_reduce_kernel(N* out,
                           MeshDataLayout layout,
                           Args... args)
  {
    using namespace mesh_reduction;
    MapOp map_op;
    ReduceOp reduce_op;

    // number of threads per dimension in the entire grid
    const int global_thread_width = gridDim.x * BLOCK_ROWS;
    const int global_thread_height = gridDim.y * BLOCK_COLS;

    // the coordinates of the current thread in the grid
    const int ix = BLOCK_ROWS * blockIdx.x + threadIdx.x;
    const int iy = BLOCK_COLS * blockIdx.y + threadIdx.y;

    // the linearized index of the current thread inside the current block
    const int i_thread = threadIdx.x * BLOCK_COLS + threadIdx.y;

    N acc = 0;
    for (int i = ix+layout.start[0]; i < layout.end[0]; i += global_thread_width) {
      for (int j = iy+layout.start[1]; j < layout.end[1]; j += global_thread_height) {
        acc = reduce_op(acc, map_op(mesh_at(args, layout, i, j)... ));
      }
    }

    __shared__ N partial_acc[BLOCK_ROWS*BLOCK_COLS];
    partial_acc[threadIdx.x * BLOCK_COLS + threadIdx.y] = acc;

    const int block_size = BLOCK_ROWS*BLOCK_COLS;
    for (int shift = block_size / 2; shift > 0; shift /= 2) {
      __syncthreads();
      if (i_thread < shift) {
        partial_acc[i_thread]
          = reduce_op(
              partial_acc[i_thread],
              partial_acc[i_thread + shift]);
      }
    }

    if (i_thread == 0) {
      const int i_out = blockIdx.x * gridDim.y + blockIdx.y;
      out[i_out] = partial_acc[i_thread];
    }
  }

  template <typename N, typename ReduceOp, typename MapOp, typename ...Args>
  N cuda_mesh_map_reduce(MeshDataLayout layout, const N* data1, const Args& ...args)
  {
    using namespace mesh_reduction;
    using Number = N;

    cudaError_t err;

    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count < 1)
      throw std::runtime_error("No CUDA device found");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    const auto grid_side_len = static_cast<unsigned int>(ceil(sqrt(prop.multiProcessorCount * 2)));

    const dim3 block_shape{ BLOCK_ROWS, BLOCK_COLS };
    const dim3 grid_shape{ grid_side_len, grid_side_len };


    const int block_count = grid_shape.x * grid_shape.y;
    CudaArray<N> d_partial(block_count);

    mesh_map_reduce_kernel<N, ReduceOp, MapOp>
                          <<<grid_shape, block_shape>>>
                          (d_partial.ptr(),
                           layout,
                           data1,
                           args...);

    CudaArray<Number> d_result(1);
    partial_map_reduce<Number, cuda_op::Sum<N>, cuda_op::Id<N>>
                      <<<1, partial_map_reduce_block_size>>>
                      (d_result.ptr(), block_count, d_partial.ptr());
    cuda_check_last_status("execute partial_vec_sum");

    Number h_result;
    err = cudaMemcpy(&h_result, d_result.ptr(), sizeof(Number), cudaMemcpyDeviceToHost);
    cuda_check_status(err, "copy result");

    return h_result;
  }

  // naive kernel
  template <typename N>
  __global__ void five_point_dirichlet(N* out, MeshDataLayout layout, N coeff[5], const N* in)
  {
    using Number = N;

    const int i = blockIdx.x * blockDim.x + threadIdx.x + layout.start[0];
    const int j = blockIdx.y * blockDim.y + threadIdx.y + layout.start[1];

    if (i < layout.end[0] && j < layout.end[1])
    {
      Number aux = 0.0;
      aux += mesh_at(in, layout, i-1, j) * coeff[0];
      aux += mesh_at(in, layout, i, j-1) * coeff[1];
      aux += mesh_at(in, layout, i, j) * coeff[2];
      aux += mesh_at(in, layout, i, j+1) * coeff[3];
      aux += mesh_at(in, layout, i+1, j) * coeff[4];
      mesh_at(out, layout, i, j) = aux;
    }
  }

  template <typename N>
  void cuda_mesh_five_point(N* out, MeshDataLayout layout, N coeff[5], const N* in)
  {
    cudaError_t err;

    const dim3 block_dim = { 32, 32, 1};

    const dim3 grid_dim { (layout.end[0] - layout.start[0] + block_dim.x - 1) / block_dim.x,
                          (layout.end[1] - layout.start[0] + block_dim.y - 1) / block_dim.y,
                          1 };

    CudaArray<N> d_coeff(5);
    err = cudaMemcpy(d_coeff.ptr(), coeff, sizeof(N)*5, cudaMemcpyHostToDevice);
    cuda_check_status(err);

    five_point_dirichlet<N>
                        <<<grid_dim, block_dim>>>
                        (out, layout, d_coeff.ptr(), in);
    cuda_check_last_status("five_point");

    err = cudaDeviceSynchronize();
    cuda_check_status(err, "five_point_dirichlet complete");

  }

  template <typename N>
  __global__ void cuda_mesh_fill_kernel(N* out, MeshDataLayout layout, N value)
  {
    const int i = blockIdx.x * blockDim.x + threadIdx.x + layout.start[0];
    const int j = blockIdx.y * blockDim.y + threadIdx.y + layout.start[1];

    if (i < layout.end[0] && j < layout.end[1]) {
      mesh_at(out, layout, i, j) = value;
    }
  }

  template <typename N>
  void cuda_mesh_fill(N* out, MeshDataLayout layout, N value)
  {
    const dim3 block_dim = { 32, 32, 1 };

    const dim3 grid_dim = {
      (layout.end[0] - layout.start[0] + block_dim.x - 1) / block_dim.x,
      (layout.end[1] - layout.start[1] + block_dim.y - 1) / block_dim.y,
      1
    };

    cuda_mesh_fill_kernel<N> <<<grid_dim, block_dim>>> (out, layout, value);
    cuda_check_last_status("cuda_mesh_fill_kernel");
  }

}

#endif
#endif
