@page kubuntu_dev_setup Kubuntu Development Setup

This guide describes how to setup your system to work on the library. This
guide has been tested with [Kubuntu] 20.04. These instructions, however,
should also work on [Debian], [Ubuntu] and [Linux Mint]. This guide does
*not* work on RPM-based systems.

*Warning:* To follow this guide, your machine should have at least 4GB RAM and
2GB swap.

## Install Git

Open Konsole (or another terminal emulator of your choice) and run the
following commands. (The dollar sign marks the beginning of a new command.)
First, install git:

    $ CMD="apt-get install -y git"; sudo $CMD || su -c "$CMD"

## Clone the Repository

To get the source of the library, we use git to clone the source repository.
We start by creating a working directory.

    $ mkdir ~/workspace
    $ cd ~/workspace

Then, clone the repository. In the following command you need to replace
`<REPOSITORY URL>` by the repository URL. In [GitLab], you can find the
repository URL by opening the repository and clicking on the `Clone` button.

    $ git clone <REPOSITORY URL> allium

After executing this command, the sources can be found in the `allium`
subdirectory of your current working directory.

## Install System Dependencies

To use the library, a few dependencies (see [Dependencies]) are needed. The
following command uses the package manager of your system to install those
dependencies.

    $ cd ~/workspace
    $ ./allium/scripts/deb-install-dependencies.sh full

## Install PETSc (Optional, but recommended)

[PETSc] is a library which provides data structures and algorithms for
parallel computing. Since we need a few special settings, PETSc needs to be
compiled from source. To do so, run

    $ cd ~/workspace
    $ ./allium/scripts/install-petsc.sh

## Building Using the CLI

If you want to develop using an IDE, you can skip to the
[Building Using QtCreator](#building-using-qtcreator) section.

To build the library run

    $ cd ~/workspace/allium
    $ cmake .
    $ make

You can then run the test suite by executing

    $ ./tests/test_suite

## Building Using QtCreator

[QtCreator] is a C++ IDE that works well with [CMake], the build system we
use. The following command installes QtCreator:

    $ sudo apt-get install qtcreator

To build the library, open QtCreator.
Then, select Menu -> *File* -> *Open File or Project*. In the dialog open the file
`workspace/allium/CMakeLists.txt`. Afterwards, click the `Configure Project`
button. To build the library select Menu -> *Build* -> *Build All*.

To run the test suite select Menu -> *Build* -> *Open Build and Run Kit
Selector*. In the pop-up window select *test_suite*. Then, run
Menu -> *Build* -> *Run*.

### Troubleshooting

- If you are having problems with the code completion, try turning off the
  Clang code model. To do so, select Menu -> *Help* -> *About Plugins*,
  unselect *ClangCodeModel*, and restart QtCreator.

[Kubuntu]: https://kubuntu.org/
[Debian]: https://debian.org/
[Ubuntu]: https://ubuntu.com/
[Linux Mint]: https://linuxmint.com/
[QtCreator]: https://www.qt.io/product/development-tools
[CMake]: https://cmake.org
[Dependencies]: ../INSTALL.md#dependencies
[PETSc]: https://www.mcs.anl.gov/petsc/
