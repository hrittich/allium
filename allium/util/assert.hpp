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

#ifndef ALLIUM_UTIL_ASSERT_HPP
#define ALLIUM_UTIL_ASSERT_HPP

#include <stdexcept>
#include <sstream>
#include <allium/config.hpp>

namespace allium {
  /**
   @brief An exception raised when an assertion fails.
   */
  class assertion_failed : public std::logic_error {
  public:
    assertion_failed(const std::string& assertion_desc)
      : logic_error(format_msg(assertion_desc))
    {}

  private:
    static std::string format_msg(const std::string& assertion_desc) {
      std::stringstream os;
      os << "Assertion violated: " << assertion_desc << std::endl;
      return os.str();
    }
  };
}

#ifdef ALLIUM_DEBUG
  inline void allium_assert(bool cond, std::string msg = std::string()) {
    if (!cond) {
      throw allium::assertion_failed(msg);
    }
  }

#else
  #define allium_assert(...)
#endif


#endif
