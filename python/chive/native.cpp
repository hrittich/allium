#include <pybind11/pybind11.h>

#include <mpi.h>
#include <cstdint>
#include <chive/la/vector.hpp>

namespace py = pybind11;

int print_comm(py::object mpi4py_comm)
{
  auto MPI = py::module::import("mpi4py").attr("MPI");

  auto x = MPI.attr("_addressof")(mpi4py_comm).cast<uintptr_t>();
  std::cout << "address: " << x << std::endl;

  MPI_Comm comm = *reinterpret_cast<MPI_Comm*>(x);

  if (comm == MPI_COMM_WORLD) {
    std::cout << "Passed the world." << std::endl;
  } else {
    std::cout << "Received something else." << std::endl;
  }

  return 42;
}

PYBIND11_MODULE(native, m)
{
  m.doc() = "Chive"; // optional module docstring

  py::class_<chive::Vector<double> >(m, "VectorD");

  m.def("print_comm", &print_comm, "Do something with the mpi4py communicator.");
}


