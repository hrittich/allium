// Copyright 2021 Hannah Rittich
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

#ifndef ALLIUM_MESH_VTK_IO_HPP
#define ALLIUM_MESH_VTK_IO_HPP

#include <allium/config.hpp>
#include <allium/ipc/comm.hpp>
#include "petsc_mesh.hpp"
#include <ostream>

/**
 @defgroup io IO
 @brief Input and output routines.
*/

namespace allium {
 /**
 @addtogroup io
 @{
 */

#ifdef ALLIUM_USE_PETSC
  /**
   Write the mesh values into a VTK file.
   */
  void write_vtk(std::string filename, const PetscMesh<2>& mesh);
#endif

/// @}
}


#endif
