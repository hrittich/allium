#include <iostream>
#include <string>
#include <cassert>

#include <chive/main/init.hpp>
#include <chive/mpi/comm.hpp>
#include <chive/la/vector.hpp>
#include <chive/la/sparse_matrix.hpp>
#include <chive/la/petsc_sparse_matrix.hpp>
#include <chive/la/cg.hpp>

using namespace chive;

int main(int argc, char** argv) {
  Init init(argc, argv);

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " dimension" << std::endl;
    return EXIT_FAILURE;
  }

  size_t N = std::stoi(argv[1]);

  auto comm = MpiComm::world();

  // for now assume that we can split the dofs evenly across the processors
  assert(N % comm.get_size() == 0);

  size_t N_loc = N / comm.get_size();

  VectorSpec spec(comm, N_loc, N);

  std::cout
    << "Rank " << comm.get_rank()
    << " [" << spec.local_start() << ", " << spec.local_end() << ")"
    << std::endl;

  LocalCooMatrix<PetscScalar> lmat;

  for (global_size_t i = spec.local_start(); i < spec.local_end(); ++i) {
    if (i > 0) {
      lmat.add(i, i-1, -1.0);
    }
    lmat.add(i, i, 2.0);
    if (i < N-1) {
      lmat.add(i, i+1, -1.0);
    }
  }

  PetscSparseMatrix mat(spec, spec);
  mat.set_entries(lmat);

  PetscVector v(spec);
  {
    auto v_loc = local_slice(v);
    for (global_size_t i_glob = spec.local_start(); i_glob < spec.local_end(); i_glob++)
    {
      global_size_t i_loc = i_glob - spec.local_start();

      if (i_glob == 0 || i_glob == spec.global_size() - 1) {
        v_loc[i_loc] = 1.0;
      } else {
        v_loc[i_loc] = 0.0;
      }
    }
  }

  auto result = cg(mat, v);

  for (global_size_t i_rank = 0; i_rank < comm.get_size(); ++i_rank) {
    comm.barrier();

    if (i_rank == comm.get_rank()) {
      auto loc = local_slice(result);
      for (size_t i_loc = 0; i_loc < spec.local_size(); ++i_loc) {
        std::cout << loc[i_loc] << std::endl;
      }
    }
  }

  return EXIT_SUCCESS;
}

