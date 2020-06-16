add_library(chive_mpi
  init.cpp init.hpp
  comm.cpp comm.hpp
  util.cpp util.hpp)
list(APPEND CHIVE_LIBRARIES chive_mpi)
set(CHIVE_LIBRARIES ${CHIVE_LIBRARIES} PARENT_SCOPE)
