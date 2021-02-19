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

#include "vtk_io.hpp"
#include <fstream>
#include <sstream>

namespace allium {

#ifdef ALLIUM_USE_PETSC

namespace {
  std::string vtk_extent(Range<2> range) {
    // To determine the end of the mesh, VTK wants the last point inside the
    // mesh, while we store the first point *outside* the mesh. Hence, we
    // need to subtract one from each coordinate.
    std::stringstream buf;
      buf
        << range.begin_pos()[0] << " " << range.end_pos()[0]-1 << "  "
        << range.begin_pos()[1] << " " << range.end_pos()[1]-1 << "  "
        << "0 0";

    return buf.str();
  }

  /*
    A simple example file of a 4x4 mesh looks as follows.

      <VTKFile type="ImageData">
        <ImageData WholeExtent="0 3  0 3  0 0" Origin="0 0 0" Spacing="0.1 0.1 0">
        <Piece Extent="0 3  0 3  0 0">
          <PointData Scalars="f">
            <DataArray Name="f" type="Float64" format="ascii">
              0.0 0.1 0.2 0.3
              0.1 0.2 0.3 0.4
              0.1 0.2 0.3 0.4
              0.0 0.1 0.2 0.3
            </DataArray>
          </PointData>
          <CellData>
          </CellData>
        </Piece>
      </ImageData>
      </VTKFile>
  */
  void vtk_write_local(std::string filename,
                       const PetscLocalMesh<double, 2>& mesh,
                       Range<2> range)
  {
    std::fstream os(filename, std::ios_base::out);

    os
      << "<VTKFile type=\"ImageData\">\n"
      << "  <ImageData WholeExtent=\"" << vtk_extent(range) << "\" Origin=\"0 0 0\" Spacing=\"1 1 0\">\n"
      << "  <Piece Extent=\"" << vtk_extent(range) << "\">\n"
      << "    <PointData Scalars=\"mesh\">\n"
      << "      <DataArray Name=\"mesh\" type=\"Float64\" format=\"ascii\">\n";

    // the mesh needs to be written in x-y-z order (x is the fastest running
    // index)
    auto lmesh = local_mesh(mesh);
    for (int iy = range.begin_pos()[1]; iy < range.end_pos()[1]; ++iy) {
      bool first = true;
      for (int ix = range.begin_pos()[0]; ix < range.end_pos()[0]; ++ix) {
        if (!first)
          first = false;
        else
          os << " ";

        os << lmesh(ix, iy);
      }
      os << std::endl;
    }

    os
      << "      </DataArray>\n"
      << "    </PointData>\n"
      << "    <CellData>\n"
      << "    </CellData>\n"
      << "  </Piece>\n"
      << "</ImageData>\n"
      << "</VTKFile>\n";
  }

  std::string remove_vtk_extension(std::string fn)
  {
    if (fn.substr(fn.size()-5) == ".pvti") {
      return fn.substr(0, fn.size()-5);
    } else {
      return fn;
    }
  }

}

/*
  Format description can be found [here](https://vtk.org/Wiki/VTK_XML_Formats)
  and [here](https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf).

  An example of a meta-data file, where the data consists of two pieces.

      <VTKFile type="PImageData">
        <PImageData WholeExtent="0 10  0 10  0 0"
                    GhostLevel="0" Origin="0 0 0" Spacing="0.1 0.1 0.1">
          <PPointData Scalars="f">
            <DataArray Name="f" type="Float64" format="ascii"/>
          </PPointData>
          <Piece Extent="0 10  0 5  0 0" Source="piece1.vti"/>
          <Piece Extent="0 10  5 10  0 0" Source="piece2.vti"/>
        </PImageData>
      </VTKFile>
*/
void write_vtk(std::string filename, const PetscMesh<double, 2>& mesh)
{
  Comm comm = mesh.spec().comm();

  std::fstream os(filename, std::ios_base::out);

  // VTK expects an overlap of the vertices on the domain boundary. Hence,
  // we need to receive the ghost cells.
  PetscLocalMesh<double, 2> ghosted_mesh(mesh.mesh_spec());
  ghosted_mesh.assign(mesh);

  auto global_range = mesh.mesh_spec()->range();

  // to get the overlap, we increase the endpoint (unless the point would be
  // outside of the mesh)
  auto lrange = mesh.mesh_spec()->local_range();

  Point<int, 2> end_pos;
  for (size_t i=0; i < end_pos.rows(); ++i) {
    end_pos[i] = std::min(lrange.end_pos()[i]+1, global_range.end_pos()[i]);
  }
  lrange = Range<2>(lrange.begin_pos(), end_pos);


  auto local_fn = [filename](int rank) {
    std::stringstream fn;
    fn << remove_vtk_extension(filename) << "_" << rank << ".vti";
    return fn.str();
  };

  // write the local data to a file
  vtk_write_local(local_fn(comm.rank()), ghosted_mesh, lrange);

  // write the global meta-data file
  if (comm.rank() == 0) {
      os
        << "<VTKFile type=\"PImageData\">\n"
        << "  <PImageData WholeExtent=\"" << vtk_extent(global_range) << "\"\n"
        << "              GhostLevel=\"0\" Origin=\"0 0 0\" Spacing=\"1 1 0\">\n"
        << "    <PPointData Scalars=\"mesh\">\n"
        << "      <DataArray Name=\"mesh\" type=\"Float64\" format=\"ascii\"/>\n"
        << "    </PPointData>\n";

      for (long i=0; i < comm.size(); ++i) {
        Range<2> rank_range;
        if (i == 0) {
          rank_range = lrange;
        } else {
          comm.recv(rank_range, i, 0);
        }

        os
          << "    <Piece Extent=\"" << vtk_extent(rank_range) << "\" "
                  "Source=\"" << local_fn(i) << "\"/>\n";
      }

      os
        << "  </PImageData>\n"
        << "</VTKFile>\n";
  } else {
    comm.send(lrange, 0, 0);
  }

}
#endif


}
