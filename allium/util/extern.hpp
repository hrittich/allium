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

#ifndef ALLIUM_UTIL_EXTERN_HPP
#define ALLIUM_UTIL_EXTERN_HPP

#include "preprocess.hpp"

#define ALLIUM_EXTERN_N(F) \
  ALLIUM_FORALL_N(ALLIUM_ARY2, F,,, extern)
#define ALLIUM_NOEXTERN_N(F) \
  ALLIUM_FORALL_N(ALLIUM_ARY2, F,,,)

#define ALLIUM_EXTERN_D(F) \
  ALLIUM_FORALL_D(ALLIUM_ARY2, F,,, extern)
#define ALLIUM_NOEXTERN_D(F) \
  ALLIUM_FORALL_D(ALLIUM_ARY2, F,,,)

#define ALLIUM_EXTERN_ND(F) \
  ALLIUM_FORALL_N(ALLIUM_FORALL_D, ALLIUM_ARY3, F,, extern)
#define ALLIUM_NOEXTERN_ND(F) \
  ALLIUM_FORALL_N(ALLIUM_FORALL_D, ALLIUM_ARY3, F,,)

#endif
