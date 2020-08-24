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

#ifndef ALLIUM_UTIL_CLONEABLE_HPP
#define ALLIUM_UTIL_CLONEABLE_HPP

#include <memory>
#include <sstream>
#include <allium/util/assert.hpp>

namespace allium {

/** A class that allows cloning of objects.
 * Any class that inherits directly or indirectly from Cloneable must
 * override the clone method. Not doing so is considered a bug.
 */
class Cloneable {
  public:
    virtual ~Cloneable();

  private:
    virtual Cloneable* clone() const& = 0;

    template <typename T>
    std::enable_if_t<std::is_base_of<Cloneable, T>::value,
                     std::unique_ptr<T>>
    friend clone(const T& o);
};

template <typename T>
std::enable_if_t<std::is_base_of<Cloneable, T>::value,
                 std::unique_ptr<T>>
clone(const T& o)
{
  Cloneable* ptr = static_cast<const Cloneable&>(o).clone();

  allium_assert(
    dynamic_cast<T*>(ptr) != nullptr,
    std::string("The clone method is not overridden in ")
      + typeid(*ptr).name()
      + ".");

  return std::unique_ptr<T>(static_cast<T*>(ptr));
}

}

#endif
