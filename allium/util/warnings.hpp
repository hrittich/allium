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

#ifndef ALLIUM_UTIL_WARNINGS_HPP
#define ALLIUM_UTIL_WARNINGS_HPP

#if defined(__GNUC__)
  #define ALLIUM_STORE_WARNING \
    _Pragma("GCC diagnostic push")
  #define ALLIUM_RESTORE_WARNING \
    _Pragma("GCC diagnostic pop")

  #define ALLIUM_NO_NONNULL_WARNING \
    ALLIUM_STORE_WARNING \
    _Pragma("GCC diagnostic ignored \"-Wnonnull-compare\"") \
    _Pragma("GCC diagnostic ignored \"-Waddress\"")

  #define ALLIUM_NO_NARROWING_WARNING \
    ALLIUM_STORE_WARNING \
    _Pragma("GCC diagnostic ignored \"-Wnarrowing\"")

  #define ALLIUM_NO_SIGN_COMPARE_WARNING \
    ALLIUM_STORE_WARNING \
    _Pragma("GCC diagnostic ignored \"-Wsign-compare\"") 

#else
  #define ALLIUM_STORE_WARNING
  #define ALLIUM_RESTORE_WARNING
  #define ALLIUM_NO_NONNULL_WARNING
  #define ALLIUM_NO_NARROWING_WARNING
  #define ALLIUM_NO_SIGN_COMPARE_WARNING
#endif

#endif
