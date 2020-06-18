#include <pybind11/pybind11.h>

#include <mpi.h>
#include <mpi4py/mpi4py.h>
#include <cstdint>
#include <chive/la/vector.hpp>
#include <chive/la/eigen_vector.hpp>

namespace py = pybind11;

struct mpi4py_comm {
  mpi4py_comm() = default;
  mpi4py_comm(MPI_Comm value) : value(value) {}
  operator MPI_Comm () { return value; }

  MPI_Comm value;
};

namespace pybind11 { namespace detail {
  template <> struct type_caster<mpi4py_comm> {
    public:
      PYBIND11_TYPE_CASTER(mpi4py_comm, _("mpi4py_comm"));

      // Python -> C++
      bool load(handle src, bool) {
        /* Extract PyObject from handle */
        PyObject *py_src = src.ptr();

        if (PyObject_TypeCheck(py_src, &PyMPIComm_Type)) {
          value.value = *PyMPIComm_Get(py_src);
        } else {
          return false;
        }

        return !PyErr_Occurred();
      }

      // C++ -> Python
      static handle cast(mpi4py_comm src,
                         return_value_policy /* policy */,
                         handle /* parent */)
      {
        return PyMPIComm_New(src.value);
      }
  };
}} // namespace pybind11::detail

PYBIND11_MODULE(native, m)
{
  if (import_mpi4py() < 0) {
    throw std::runtime_error("Could not load mpi4py.");
  }

  m.doc() = "Chive"; // optional module docstring

  py::class_<chive::MpiComm>(m, "MpiComm")
    .def(py::init<mpi4py_comm>())
    .def_static("world", &chive::MpiComm::world)
    .def("get_rank", &chive::MpiComm::get_rank)
    .def("get_size", &chive::MpiComm::get_size)
    .def("get_handle",
         [](chive::MpiComm self) -> mpi4py_comm
         { return self.get_handle(); });

  py::class_<chive::VectorSpec>(m, "VectorSpec")
    .def(py::init<chive::MpiComm, chive::global_size_t, size_t>())
    .def("get_global_size", &chive::VectorSpec::get_global_size)
    .def("get_local_size", &chive::VectorSpec::get_local_size);

  py::class_<chive::Vector<double>>(m, "VectorD")
    .def(py::init([](chive::VectorSpec spec) {
           return
             chive::Vector<double>(
               std::make_shared<chive::EigenVectorStorage<double>>(spec));
         }));

}


