# Allium - *Al*gorithm *Li*brary for *U*pscaling *M*athematics

![Allium Logo](doc/logo_allium.png)

## Developer Quickstart

The easiest way to get started is to run Linux (e.g.,
[Debian](https://debian.org) or [Ubuntu](https://ubuntu.com)) and use a
[Docker](https://www.docker.com/) image.
(For more information about the Docker images, see the [Docker
Section](doc/docker.md) of the documentation.) Executing

    $ scripts/docker-build.sh

in a shell from the main source directory, builds the required images. You
can then run the command

    $ scripts/docker-start.sh full shell

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

## Full Developer Setup

See [Kubuntu Developer Setup](doc/kubuntu_dev_setup.md).

## Installation

*@ToDo: Complete the installation instructions*

### Dependencies

- Mandantory
  - [C++-14 Compiler](https://en.wikipedia.org/wiki/C%2B%2B14),
    e.g., [GCC](https://gcc.gnu.org/)
  - [CMake](https://cmake.org)
  - [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface), e.g.,
    [OpenMPI](https://www.open-mpi.org/)
- Optional
  - [PETSc](https://www.python.org/)
  - [Python 3](https://www.python.org/)
  - [mpi4py](https://bitbucket.org/mpi4py/mpi4py)
- Included
  - [googletest](https://github.com/google/googletest)
  - [pybind11](https://github.com/pybind/pybind11)
  - [Eigen](http://eigen.tuxfamily.org/)



