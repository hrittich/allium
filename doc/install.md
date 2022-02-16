@page install Build and Installation

This section describes how to build and install the library.

## Dependencies

To build Allium, you need to make sure that the required dependencies are
installed. Dependencies fall into three categories. Mandatory dependencies
have to be installed for building Allium. Optional dependencies can be
omitted, but in this case certain functionality might be missing.
Included dependencies are bundled with Allium and do not need to be installed
separately. All dependencies are listed below.

If you do not know how to install the required dependencies on your system,
consider using the build environment Docker image, as described in the
@ref quickstart guide.

- Mandatory
  - [C++-14 Compiler](https://en.wikipedia.org/wiki/C%2B%2B14),
    e.g., [GCC](https://gcc.gnu.org/)
  - [CMake](https://cmake.org)
  - [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface), e.g.,
    [OpenMPI](https://www.open-mpi.org/)
- Optional
  - [PETSc](https://petsc.org/)
  - [Nvidia CUDA](https://developer.nvidia.com/cuda-zone)
  - [GSL](https://www.gnu.org/software/gsl/)
  - [Python 3](https://www.python.org/)
  - [mpi4py](https://bitbucket.org/mpi4py/mpi4py)
  - [Doxygen](https://doxygen.nl)
- Included
  - [googletest](https://github.com/google/googletest)
  - [pybind11](https://github.com/pybind/pybind11)
  - [Eigen](http://eigen.tuxfamily.org/)
  - [Doxygen Awesome](https://jothepro.github.io/doxygen-awesome-css/)

## Build

To build the library, run the following commands when inside the source
directory.

    $ cmake [OPTIONS] .
    $ make

In here, `[OPTIONS]` refers to a list of options which tailors the library
to your needs. The options are listed below. If you are content with the
standard configuration, you can run the command without providing any
options.

- `-DCMAKE_BUILD_TYPE=<TYPE>`  
  The value of `<TYPE>` can be `Debug`, `Release` or `RelWithDebInfo` to
  build the library in debug, release, or release with debugging information
  mode, respectively.

- `-DCMAKE_INSTALL_PREFIX=<PATH>`  
  Determines where the library will be installed (see section below).

- `-DALLIUM_USE_CUDA=(ON|OFF)`  
  Enables or disables the use of CUDA.

- `-DALLIUM_USE_GSL=(ON|OFF)`  
  Enables or disables the use of the GSL.

- `-DALLIUM_USE_MPI4PY=(ON|OFF)`  
  Enables or disables the use of mpi4py.

- `-DALLIUM_USE_PETSC=(ON|OFF)`  
  Enables or disables the use of PETSc.

- `-DALLIUM_USE_PYTHON=(ON|OFF)`  
  Enables or disables the use of Python.

## Running a Demo Application

Once you have successfully built the library, you can run the demo
applications located in the `demo` directory. For example, you can solve a
simple 1D Poisson equation using the following command.

    $ demo/poisson 128

## Installation

Installing the library on your system can be done by running:

    $ sudo make install

For changing the installation destination, consult the description of the
`CMAKE_INSTALL_PREFIX` option above.

## Next Step

- @ref demos
