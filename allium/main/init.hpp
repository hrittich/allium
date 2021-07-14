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

#ifndef ALLIUM_MAIN_INIT_HPP
#define ALLIUM_MAIN_INIT_HPP

#include <allium/ipc/init.hpp>

namespace allium {
  /**
   @brief Initializes the library.

   As long as this object exists the library can be used. Make sure to create
   only one instance of Init.
   */
  class Init {
    public:
      Init(const Init&) = delete;
      Init& operator=(const Init&) = delete;

      Init(int& argc, char** &argv);
      ~Init();

    private:
      IpcInit mpi;
  };
}

#endif
