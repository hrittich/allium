add_library(chive_main
  init.cpp init.hpp)
list(APPEND CHIVE_LIBRARIES chive_main)
set(CHIVE_LIBRARIES ${CHIVE_LIBRARIES} PARENT_SCOPE)
