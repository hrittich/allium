#ifndef CHIVE_UTIL_EXTERN_HPP
#define CHIVE_UTIL_EXTERN_HPP

#define CHIVE_INSTANTIATE(DECL) \
  DECL(template, float) \
  DECL(template, std::complex<float>) \
  DECL(template, double) \
  DECL(template, std::complex<double>)

#define CHIVE_EXTERN(DECL) \
  DECL(extern template, float) \
  DECL(extern template, std::complex<float>) \
  DECL(extern template, double) \
  DECL(extern template, std::complex<double>)

#endif
