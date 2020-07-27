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

#include <pybind11/pybind11.h>

#include <mpi.h>
#include <mpi4py/mpi4py.h>
#include <cstdint>
#include <allium/la/vector.hpp>
#include <allium/la/eigen_vector.hpp>
#include <allium/la/sparse_matrix.hpp>
#include <allium/la/eigen_sparse_matrix.hpp>

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

  m.doc() = "Algorithm Library for Upscaling Mathematics"; // optional module docstring

  py::class_<allium::Comm>(m, "Comm")
    .def(py::init<mpi4py_comm>())
    .def_static("world", &allium::Comm::world)
    .def("rank", &allium::Comm::rank)
    .def("size", &allium::Comm::size)
    .def("handle",
         [](allium::Comm self) -> mpi4py_comm
         { return self.handle(); });

  py::class_<allium::VectorSpec>(m, "VectorSpec")
    .def(py::init<allium::Comm, allium::global_size_t, size_t>())
    .def("global_size", &allium::VectorSpec::global_size)
    .def("local_size", &allium::VectorSpec::local_size);

  py::class_<allium::Vector<double>>(m, "VectorD")
    .def(py::init([] (allium::VectorSpec spec) {
                    return allium::EigenVector<double>(spec);
                  }));

  py::class_<allium::SparseMatrix<double>>(m, "SparseMatrixD")
    .def(py::init([] (allium::VectorSpec row_spec, allium::VectorSpec col_spec) {
                    return
                      allium::SparseMatrix<double>(
                        allium::EigenSparseMatrix<double>(row_spec, col_spec));
                  }));

}


