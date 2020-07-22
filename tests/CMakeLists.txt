add_executable(test_suite
  main.cpp
  vector.cpp
  sparse_matrix.cpp
  petsc.cpp
  eigen.cpp
  cg.cpp
  gmres.cpp)
target_link_libraries(test_suite ${GTEST_LIBRARIES} ${LIBRARIES} ${CHIVE_LIBRARIES})

