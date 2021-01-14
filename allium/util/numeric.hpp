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

#ifndef ALLIUM_UTIL_NUMERIC_HPP
#define ALLIUM_UTIL_NUMERIC_HPP

#include <complex>
#include "warnings.hpp"

namespace allium {
  /// @cond INTERNAL
  template <typename T> struct real_part {};
  template <> struct real_part<float> { typedef float type; };
  template <> struct real_part<double> { typedef double type; };
  template <> struct real_part<int> { typedef int type; };
  template <> struct real_part<long> { typedef long type; };
  template <typename T> struct real_part<std::complex<T>> { typedef T type; };


  template <typename N> struct is_complex : public std::false_type {};
  template <typename N> struct is_complex<std::complex<N>> : public std::true_type {};
  /// @endcond

  template <typename T>
  using real_part_t = typename real_part<T>::type;

  /**
   Narrows a number type, which can result in a loss of information or
   accuracy.

   Allows the conversion of double to float, and the conversion of
   complex to real types by dropping the imaginary part.
   */
  template <typename To, typename From, typename Enabled = void>
  struct narrow_number {};

  template <>
  struct narrow_number<float, double> {
    float operator() (double x) { return x; }
  };

  template <typename T>
  struct narrow_number<T, T> {
    T operator() (T x) { return x; }
  };

  template <>
  struct narrow_number<std::complex<float>, std::complex<double>> {
    std::complex<float> operator() (std::complex<double> x) {
      return std::complex<float>(x.real(), x.imag());
    }
  };

  template <typename To, typename From>
  struct narrow_number<To, std::complex<From>,
                       std::enable_if_t<std::is_floating_point<To>::value>>
  {
    To operator() (std::complex<From> x) {
      allium_assert(x.imag() == 0);
      return narrow_number<To, From>()(x.real());
    }
  };

  /**
   Computes `a <= b`, where a comparison between signed and unsigned types is
   safe.
   */
  template <typename T1, typename T2>
  std::enable_if_t<std::is_signed<T1>::value && std::is_unsigned<T2>::value, bool>
  safe_le(T1 a, T2 b) {
    // b is always >= 0, hence a <= b if a < 0, otherwise a comparison is safe
    return (a < 0) || (static_cast<std::make_unsigned_t<T1>>(a) <= b);
  }
  template <typename T1, typename T2>
  std::enable_if_t<std::is_unsigned<T1>::value && std::is_signed<T2>::value, bool>
  safe_le(T1 a, T2 b) {
    // a is always >= 0, hence a <= b requires b >= 0, then a comparison is safe
    return (b >= 0) && (a <= static_cast<std::make_unsigned_t<T2>>(b));
  }
  template <typename T1, typename T2>
  std::enable_if_t<std::is_signed<T1>::value == std::is_signed<T2>::value, bool>
  safe_le(T1 a, T2 b) {
    // same signage, comparison is safe
    return a <= b;
  }

  /**
   Computes `a >= b`, where a comparison between signed and unsigned types is
   safe.
   */
  template <typename T1, typename T2>
  bool safe_ge(T1 a, T2 b) {
    return safe_le(b, a);
  }

  /**
   Computes `a < b`, where a comparison between signed and unsigned types is
   safe.
   */
  template <typename T1, typename T2>
  std::enable_if_t<std::is_signed<T1>::value && std::is_unsigned<T2>::value, bool>
  safe_lt(T1 a, T2 b) {
    // b is always >= 0, hence a < b if a < 0, otherwise a comparison is safe
    return (a < 0) || (static_cast<std::make_unsigned_t<T1>>(a) < b);
  }
  template <typename T1, typename T2>
  std::enable_if_t<std::is_unsigned<T1>::value && std::is_signed<T2>::value, bool>
  safe_lt(T1 a, T2 b) {
    // a is always >= 0, hence a < b requires b >= 0, then a comparison is safe
    return (b >= 0) && (a < static_cast<std::make_unsigned_t<T2>>(b));
  }
  template <typename T1, typename T2>
  std::enable_if_t<std::is_signed<T1>::value == std::is_signed<T2>::value, bool>
  safe_lt(T1 a, T2 b) {
    // same signage, comparison is safe
    return a < b;
  }

  /**
   Computes `a > b`, where a comparison between signed and unsigned types is
   safe.
   */
  template <typename T1, typename T2>
  bool safe_gt(T1 a, T2 b) {
    return safe_lt(b, a);
  }

}

#endif
