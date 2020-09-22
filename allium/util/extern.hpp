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

#define ALLIUM_EXTERN_N_1(DECL, T) \
  DECL(T, float) \
  DECL(T, std::complex<float>) \
  DECL(T, double) \
  DECL(T, std::complex<double>)

#define ALLIUM_EXTERN(DECL)       ALLIUM_EXTERN_N_1(DECL, extern template)
#define ALLIUM_INSTANTIATE(DECL)  ALLIUM_EXTERN_N_1(DECL, template)



#define ALLIUM_EXTERN_D_1(DECL, T) \
  DECL(T, 1) \
  DECL(T, 2) \
  DECL(T, 3)

#define ALLIUM_EXTERN_D(DECL) ALLIUM_EXTERN_D_1(DECL, extern template)
#define ALLIUM_INSTANTIATE_D(DECL) ALLIUM_EXTERN_D_1(DECL, template)



#define ALLIUM_EXTERN_ND_2(DECL, T, N) \
  DECL(T, N, 1) \
  DECL(T, N, 2) \
  DECL(T, N, 3)

#define ALLIUM_EXTERN_ND_1(DECL, T) \
  ALLIUM_EXTERN_ND_2(DECL, T, float) \
  ALLIUM_EXTERN_ND_2(DECL, T, std::complex<float>) \
  ALLIUM_EXTERN_ND_2(DECL, T, double) \
  ALLIUM_EXTERN_ND_2(DECL, T, std::complex<double>)

#define ALLIUM_EXTERN_ND(DECL) ALLIUM_EXTERN_ND_1(DECL, extern template)
#define ALLIUM_INSTANTIATE_ND(DECL) ALLIUM_EXTERN_ND_1(DECL, template)



#endif
