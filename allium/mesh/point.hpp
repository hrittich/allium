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

#ifndef ALLIUM_MESH_POINT_HPP
#define ALLIUM_MESH_POINT_HPP

#include <allium/util/assert.hpp>
#include <allium/la/vector_trait.hpp>
#include <ostream>
#include <algorithm>
#include <numeric>

namespace allium {

  /**
   A point in D dimensions.

   This class is a short, fixed-length, stack-allocated vector. */
  template <typename N, int D>
  class Point : public VectorTrait<Point<N,D>, N> {
    public:
      Point() = default;

      Point(const std::initializer_list<N> &entries) {
        allium_assert(entries.size() == D, "Correct initializer length.");

        int i = 0;
        for (auto e : entries) {
          m_entries[i] = e;
          ++i;
        }
      }

      static Point full(N initial_value) {
        Point p;
        std::fill(p.m_entries.begin(), p.m_entries.end(), initial_value);
        return p;
      }

      size_t rows() { return D; }

      N& operator[] (int i) {
        return m_entries[i];
      }

      const N& operator[] (int i) const {
        return const_cast<Point<N,D>&>(*this)[i];
      }

      bool operator== (const Point& rhs) const {
        for (int i = 0; i < D; ++i) {
          if (m_entries[i] != rhs[i]) {
            return false;
          }
        }
        return true;
      }

      Point operator+= (const Point& rhs) {
        for (int i = 0; i < D; ++i) {
          m_entries[i] += rhs[i];
        }
        return *this;
      }

      Point operator*= (N factor) {
        for (int i = 0; i < D; ++i) {
          m_entries[i] *= factor;
        }
        return *this;
      }

      Point<N, D+1> joined(N x) {
        Point<N, D+1> result;
        for (int i = 0; i < D; ++i) {
          result[i] = (*this)[i];
        }
        result[D] = x;
        return result;
      }

      template <typename Pred>
      bool all_of(Pred pred) {
        return std::all_of(m_entries.begin(), m_entries.end(), pred);
      }

      N prod() const {
        return std::accumulate(m_entries.begin(), m_entries.end(),
                               N(1),
                               [](N a, N b) { return a*b; });
      }

    private:
      std::array<N, D> m_entries;
  };

  template <typename N, int D>
  std::ostream& operator<< (std::ostream& os, const Point<N, D>& p) {
    bool first = true;
    os << "(";
    for (int i = 0; i < D; ++i) {
      if (first)
        first = false;
      else
        os << ", ";
      os << p[i];
    }
    os << ")";

    return os;
  }

}

#endif
