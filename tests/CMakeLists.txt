add_executable(test_suite
  main.cpp
  vector.cpp
  petsc.cpp
  eigen.cpp)
target_link_libraries(test_suite ${GTEST_LIBRARIES} ${LIBRARIES} ${CHIVE_LIBRARIES})

