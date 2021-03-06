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

#ifndef ALLIUM_UTIL_EXCEPT_HPP
#define ALLIUM_UTIL_EXCEPT_HPP

#include <stdexcept>

namespace allium {
  /**
   @brief An exception thrown when trying to use a functionality that is not
     implemented.
   */
  class not_implemented : public std::logic_error {
    public:
      not_implemented() : std::logic_error("Not implemented.") {};
  };
}

#endif
