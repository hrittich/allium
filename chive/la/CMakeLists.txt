add_library(chive_la
  vector.cpp vector.hpp
  local_coo_matrix.hpp
  sparse_matrix.cpp sparse_matrix.hpp
  petsc_vector.cpp petsc_vector.hpp
  petsc_object_ptr.hpp
  petsc_util.hpp)
list(APPEND CHIVE_LIBRARIES chive_la)
set(CHIVE_LIBRARIES ${CHIVE_LIBRARIES} PARENT_SCOPE)
