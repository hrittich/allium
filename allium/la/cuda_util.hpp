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

#ifndef ALLIUM_LA_CUDA_UTIL_HPP
#define ALLIUM_LA_CUDA_UTIL_HPP

#include <allium/config.hpp>
#ifdef ALLIUM_USE_CUDA

#include <string>
#include <cuda_runtime.h>

namespace allium {
  void cuda_check_status(cudaError_t err, std::string msg = "");
  void cuda_check_last_status(std::string msg = "");

  namespace cuda_op {

    template <typename N>
    struct Inc {
      __device__ N operator() (N a) {
        return a+1;
      }
    };

    template <typename N>
    struct Sum {
      __device__ N operator() (N a, N b) {
        return a+b;
      }
    };

    template <typename N>
    struct Prod {
      __device__ N operator() (N a, N b) {
        return a*b;
      }
    };

    template <typename N>
    struct Id {
      __device__ N operator() (N a) { return a; }
    };

    template <typename N>
    struct Square {
      __device__ N operator() (N a) { return a*a; }
    };

    template <typename N>
    class MulBy {
      public:
        MulBy(N s) : s(s) {}

        __device__ N operator() (N a) {
          return a * s;
        }

      private:
        N s;
    };

    template <typename N>
    class AddScaled {
      public:
        AddScaled(N s) : m_factor(s) {}

        __device__ N operator() (N a, N b) {
          return a + m_factor * b;
        }
      private:
        N m_factor;
    };
  }

  const size_t partial_map_reduce_block_size = 512;

  /**
    Applies a map-reduce operation for each block. The result is stored
    in the output array for each block.
  */
  template <typename N, typename ReduceOp, typename MapOp, typename ...Args>
  __global__ void partial_map_reduce(N* out, size_t n, Args ...a)
  {
    using Number = N;
    MapOp map_op;
    ReduceOp reduce_op;
    const int i_thread = threadIdx.x;
    const int i_block = blockIdx.x;
    const int i_global = threadIdx.x + blockIdx.x * blockDim.x;
    const int grid_size = blockDim.x * gridDim.x;

    const auto block_size = partial_map_reduce_block_size;
    //assert(blockDim.x == block_size); // actually we only need that the block size is a power of 2

    // Each thread applies the map operation to a portion of the input.
    // Furthermore each thread applies the reduction operation to its portion
    // of the data.

    Number thread_result = 0.0;
    for (int i = i_global; i < n; i += grid_size) {
      thread_result = reduce_op(thread_result, map_op(a[i]...));
    }

    __shared__ Number block_result[block_size];
    block_result[i_thread] = thread_result;

    // Apply the reduction operation to obtain one value per block.
    __syncthreads();
    for (int size = block_size/2; size > 0; size /= 2) {
      if (i_thread < size) {
        block_result[i_thread]
          = reduce_op(block_result[i_thread], block_result[i_thread + size]);
      }
      __syncthreads();
    }

    // block_result[0] contains the sum of the block
    if (i_thread == 0) {
      out[i_block] = block_result[0];
    }
  }


}

#endif
#endif
