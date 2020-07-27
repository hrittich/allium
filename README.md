# Chive - Framework for Distributed Numerical Computation

## Developer Quickstart

The easiest way to get started is to run Linux (e.g.,
[Debian](https://debian.org) or [Ubuntu](https://ubuntu.com)) and use a
[Docker](https://www.docker.com/) image.
(For more information about the Docker images, see the [Docker
Section](doc/docker.md) of the documentation.) Executing

    $ docker/scripts/build.sh

in a shell from the main source directory, builds the required images. You
can then run the command

    $ docker/scripts/start.sh full shell

to get shell access to a build environment. The source code is mounted into
the `source` directory of the user's home directory of this environment.
Hence, you can build the framework using the following commands:

    $ cd source
    $ cmake .
    $ make

Then, you can execute the first demo program, by executing

    $ demo/poisson 100

Note that **everything you store outside of the `source` directory will be
lost, when you exit the container**.

## Installation

*@ToDo: Complete the installation instructions*

### Dependencies

- Mandantory
  - [C++-14 Compiler](container-init.sh), e.g., [GCC](https://gcc.gnu.org/)
  - [CMake](https://cmake.org)
  - [Python 3](https://www.python.org/)
  - [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface), e.g.,
    [OpenMPI](https://www.open-mpi.org/)
- Optional
  - [PETSc](https://www.python.org/)
- Included
  - googletest
  - pybind11
  - Eigen



