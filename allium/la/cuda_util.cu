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

#include "cuda_util.hpp"
#include <stdexcept>
#include <sstream>

namespace allium {

  void cuda_check_status(cudaError_t err, std::string msg) {
    if (err != cudaSuccess)
    {
      std::stringstream os;
      os << "CUDA Failure "
         << "(" << cudaGetErrorString(err) << ")";

      if (!msg.empty())
        os << ": " << msg;

      throw std::runtime_error(os.str());
    }
  }

  void cuda_check_last_status(std::string msg) {
    cuda_check_status(cudaGetLastError(), msg);
  }

}
