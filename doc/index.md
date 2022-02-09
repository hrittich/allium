@mainpage Allium

<h2><em>Al</em>gorithm <em>Li</em>brary for <em>U</em>pscaling <em>M</em>athematics</h2>

@image{inline} html logo_allium.png

Allium is a library containing routines for scientific computations,
especially linear algebra routines and solvers for ordinary differential
equations.

**WARNING**: The library is still in *alpha* stage. Expect things to break!

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

- @subpage download
- @subpage install
- @subpage demos

