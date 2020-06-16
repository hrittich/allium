#include <iostream>
#include <chive/mpi/init.hpp>

using namespace chive;

int main(int argc, char** argv) {
  MpiInit init(argc, argv);


  return EXIT_SUCCESS;
}

