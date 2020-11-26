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

#ifndef ALLIUM_UTIL_PREPROCESS_HPP
#define ALLIUM_UTIL_PREPROCESS_HPP

#define ALLIUM_FORALL_N(F, A1, A2, A3, A4) \
  F(A1, A2, A3, A4, float) \
  F(A1, A2, A3, A4, double) \
  F(A1, A2, A3, A4, std::complex<float>) \
  F(A1, A2, A3, A4, std::complex<double>)

#define ALLIUM_FORALL_D(F, A1, A2, A3, A4) \
  F(A1, A2, A3, A4, 1) \
  F(A1, A2, A3, A4, 2) \
  F(A1, A2, A3, A4, 3)

#define ALLIUM_ARY1(F, A1, A2, A3, A4) F(A4)
#define ALLIUM_ARY2(F, A1, A2, A3, A4) F(A3, A4)
#define ALLIUM_ARY3(F, A1, A2, A3, A4) F(A2, A3, A4)
#define ALLIUM_ARY4(F, A1, A2, A3, A4) F(A1, A2, A3, A4)

#endif
