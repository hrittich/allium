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

#ifndef ALLIUM_MESH_LOCAL_MESH_HPP
#define ALLIUM_MESH_LOCAL_MESH_HPP

#include <vector>
#include "range.hpp"

namespace allium {

template <typename N, int D>
class LocalMesh
{
  public:
    using Number = N;

    LocalMesh(Range<D> range)
      : m_range(range),
        m_entries(range.size())
    {}

    N& operator[] (Point<int, D> pos) {
      return m_entries[m_range.index(pos)];
    }

    const N& operator[] (Point<int, D> pos) const {
      return const_cast<LocalMesh&>(*this)[pos];
    }

  private:
    Range<D> m_range;
    std::vector<double> m_entries;
};

}

#endif
