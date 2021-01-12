// Copyright 2020 Hannah Rittich
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

#include <iostream>
#include <string>
#include <cassert>

#include <allium/main/init.hpp>
#include <allium/ipc/comm.hpp>
#include <allium/la/cg.hpp>
#include <allium/la/default.hpp>

using namespace allium;

int main(int argc, char** argv) {
  // Initialize the library, especially the IPC
  Init init(argc, argv);

  // Read the #dof
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " dimension" << std::endl;
    return EXIT_FAILURE;
  }
  size_t N = std::stoi(argv[1]);

  auto comm = Comm::world();

  // for now assume that we can split the dofs evenly across the processors
  assert(N % comm.size() == 0);
  size_t N_loc = N / comm.size();

  // Create a specification for the vectors we use. This statement specifies
  // to global vector size, the local storage size and the IPC communicator
  // to use.
  VectorSpec spec(comm, N_loc, N);

  std::cout
    << "Rank " << comm.rank()
    << " [" << spec.local_start() << ", " << spec.local_end() << ")"
    << std::endl;

  // Create a sparse matrix, representing a discrete version of the 1D
  // Laplace operator
  LocalCooMatrix<std::complex<double>> lmat;

  //   Every rank sets its local entries
  for (global_size_t i = spec.local_start(); i < spec.local_end(); ++i) {
    if (i > 0) {
      lmat.add(i, i-1, -1.0);
    }
    lmat.add(i, i, 2.0);
    if (i < N-1) {
      lmat.add(i, i+1, -1.0);
    }
  }

  auto mat = std::make_shared<DefaultSparseMatrix<std::complex<double>>>(spec, spec);
  mat->set_entries(lmat);

  // Create the right hand side vector
  DefaultVector<std::complex<double>> v(spec);

  auto v_loc = local_slice(v); // get access to the local portion of the vector
  for (global_size_t i_glob = spec.local_start(); i_glob < spec.local_end(); i_glob++)
  {
    global_size_t i_loc = i_glob - spec.local_start();

    if (i_glob == 0 || i_glob == spec.global_size() - 1) {
      v_loc[i_loc] = 1.0;
    } else {
      v_loc[i_loc] = 0.0;
    }
  }
  v_loc.release(); // commit local changes to vector

  // Solve the linear system using the CG algorithm
  DefaultVector<std::complex<double>> result(spec);
  cg(result, mat, v);

  // Print the result
  for (int i_rank = 0; i_rank < comm.size(); ++i_rank) {
    comm.barrier();

    if (i_rank == comm.rank()) {
      auto loc = local_slice(result);
      for (size_t i_loc = 0; i_loc < spec.local_size(); ++i_loc) {
        std::cout << loc[i_loc] << std::endl;
      }
    }
  }

  return EXIT_SUCCESS;
}

