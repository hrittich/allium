#  Copyright 2020 Hannah Rittich
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

add_executable(poisson poisson.cpp)
target_link_libraries(poisson ${ALLIUM_LIBRARIES} ${LIBRARIES})

add_executable(io io.cpp)
target_link_libraries(io ${ALLIUM_LIBRARIES} ${LIBRARIES}) 

if(ALLIUM_USE_PETSC)
  add_executable(poisson2 poisson2.cpp)
  target_link_libraries(poisson2 ${ALLIUM_LIBRARIES} ${LIBRARIES})

  add_executable(fisher2 fisher2.cpp)
  target_link_libraries(fisher2 ${ALLIUM_LIBRARIES} ${LIBRARIES})

  add_executable(heat_periodic2 heat_periodic2.cpp)
  target_link_libraries(heat_periodic2 ${ALLIUM_LIBRARIES} ${LIBRARIES})
endif()

if(ALLIUM_USE_CUDA)
  add_executable(poisson2_cuda poisson2_cuda.cpp)
  target_link_libraries(poisson2_cuda ${ALLIUM_LIBRARIES} ${LIBRARIES})
endif()

