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

#ifndef ALLIUM_MESH_RANGE_HPP
#define ALLIUM_MESH_RANGE_HPP

#include <iterator>
#include "point.hpp"

namespace allium {
  template <int> class RangeIterator;

  template <int D>
  class Range {
    public:
      Range() = default;

      Range(Point<int, D> begin_pos, Point<int, D> end_pos)
        : m_begin_pos(begin_pos),
          m_end_pos(end_pos)
      {}

      bool in(Point<int, D> p) {
        for (int i = 0; i < D; ++i) {
          if (p[i] < m_begin_pos[i] || p[i] >= m_end_pos[i])
            return false;
        }
        return true;
      }

      int index(Point<int, D> p) {
        int idx = p[0] - m_begin_pos[0];
        for (int i = 1; i < D; ++i) {
          int dim_size = m_end_pos[i] - m_begin_pos[i];
          idx = (p[i] - m_begin_pos[i]) + dim_size * idx;
        }
        return idx;
      }

      Point<int, D> begin_pos() const { return m_begin_pos; }
      Point<int, D> end_pos() const { return m_end_pos; }

      RangeIterator<D> begin() const;
      RangeIterator<D> end() const;

      /**
        The number of points per dimension.
      */
      Point<int, D> shape() const { return m_end_pos - m_begin_pos; }

      /**
        The total number of points.
      */
      size_t size() const {
        size_t s = 1;
        for (int i=0; i < D; ++i) {
          s *= shape()[i];
        }
        return s;
      }
    private:
      Point<int, D> m_begin_pos, m_end_pos;
  };

  /// @cond INTERNAL
  template <int D, int I = D>
  struct next_in_range {
    static void exec(const Range<D>& r, Point<int, D>& p, bool& overflow) {
      ++p[I-1];
      if (p[I-1] >= r.end_pos()[I-1]) {
        p[I-1] = r.begin_pos()[I-1];
        next_in_range<D, I-1>::exec(r, p, overflow);
      }
    }
  };
  template <int D>
  struct next_in_range<D, 0> {
    static void exec(const Range<D>& r, Point<int, D>& p, bool& overflow) {
      overflow = true;
    }
  };
  /// @endcond

  template <int D>
  class RangeIterator {
    public:
      using iterator_category = std::input_iterator_tag;
      using value_type = Point<int, D>;
      //using difference_type Distance
      using pointer   = Point<int, D>*;
      using reference = Point<int, D>&;

      RangeIterator(Point<int, D> start, const Range<D>* range)
        : m_range(range),
          m_value(start),
          m_overflow(false)
      {}

      /* Out of range position. */
      RangeIterator() : m_overflow(true) {}

      RangeIterator(const RangeIterator&) = delete;
      RangeIterator& operator= (const RangeIterator&) = delete;

      RangeIterator(RangeIterator&&) = default;

      bool operator==(const RangeIterator& rhs) const {
        return m_overflow && rhs.m_overflow;
      }

      bool operator!=(const RangeIterator& rhs) const {
        return !(*this == rhs);
      }

      RangeIterator& operator++ () {
        next_in_range<D>::exec(*m_range, m_value, m_overflow);
        return *this;
      }

      Point<int, D> operator* () {
        return m_value;
      }

    private:
      const Range<D>* m_range;
      Point<int, D> m_value;
      bool m_overflow;
  };

  template <int D>
  RangeIterator<D> Range<D>::begin() const {
    return RangeIterator<D>(begin_pos(), this);
  }

  template <int D>
  RangeIterator<D> Range<D>::end() const {
    return RangeIterator<D>();
  }

  template <int D>
  std::ostream& operator<< (std::ostream& os, Range<D> r)
  {
    os << "Range(" << r.begin_pos() << ", " << r.end_pos() << ")";
    return os;
  }

}

#endif
