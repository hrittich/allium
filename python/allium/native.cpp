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

#include <allium/config.hpp>
#include <pybind11/pybind11.h>

#include <mpi.h>

#ifdef ALLIUM_USE_MPI4PY
  #include <mpi4py/mpi4py.h>
#endif // ALLIUM_USE_MPI4PY
#include <cstdint>
#include <allium/la/vector_storage.hpp>
#include <allium/la/eigen_vector.hpp>
#include <allium/la/sparse_matrix.hpp>
#include <allium/la/eigen_sparse_matrix.hpp>

namespace py = pybind11;

#ifdef ALLIUM_USE_MPI4PY
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
#endif // ALLIUM_USE_MPI4PY

PYBIND11_MODULE(native, m)
{
  #ifdef ALLIUM_USE_MPI4PY
  if (import_mpi4py() < 0) {
    throw std::runtime_error("Could not load mpi4py.");
  }
  #endif // ALLIUM_USE_MPI4PY

  m.doc() = "Algorithm Library for Upscaling Mathematics"; // optional module docstring

  py::class_<allium::Comm>(m, "Comm")
  #ifdef ALLIUM_USE_MPI4PY
    .def(py::init<mpi4py_comm>())
    .def("handle",
         [](allium::Comm self) -> mpi4py_comm
         { return self.handle(); })
  #endif
    .def_static("world", &allium::Comm::world)
    .def("rank", &allium::Comm::rank)
    .def("size", &allium::Comm::size);

  py::class_<allium::VectorSpec>(m, "VectorSpec")
    .def(py::init<allium::Comm, allium::global_size_t, size_t>())
    .def("global_size", &allium::VectorSpec::global_size)
    .def("local_size", &allium::VectorSpec::local_size);

}


