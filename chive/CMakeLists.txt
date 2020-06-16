configure_file(config.hpp.in config.hpp)

add_subdirectory(main)
add_subdirectory(mpi)
add_subdirectory(la)

set(CHIVE_LIBRARIES ${CHIVE_LIBRARIES} PARENT_SCOPE)
