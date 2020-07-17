add_library(chive_la
  vector_spec.cpp vector_spec.hpp
  vector.cpp vector.impl.hpp vector.hpp
  local_coo_matrix.hpp
  sparse_matrix.cpp sparse_matrix.hpp
  eigen_vector.cpp eigen_vector.impl.hpp eigen_vector.hpp
  petsc_vector.cpp petsc_vector.hpp
  petsc_sparse_matrix.cpp petsc_sparse_matrix.hpp
  petsc_object_ptr.hpp
  petsc_util.hpp
  cg.cpp cg.impl.hpp cg.hpp)

list(APPEND CHIVE_LIBRARIES chive_la)
set(CHIVE_LIBRARIES ${CHIVE_LIBRARIES} PARENT_SCOPE)
