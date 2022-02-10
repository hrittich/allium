@mainpage Allium

<h2><em>Al</em>gorithm <em>Li</em>brary for <em>U</em>pscaling <em>M</em>athematics</h2>

@image{inline} html logo_allium.png

Allium is a library containing routines for scientific computations,
especially linear algebra routines and solvers for ordinary differential
equations.

**WARNING**: The library is still in *alpha* stage. Expect things to break!

@section quickstart Quickstart

Using the Allium build environment [Docker] image is the easiest way to get
started. For this guide, I am assuming that you have installed [Git] and
Docker and are familiar with their basics.

From a directory of your choice run the following commands (without the dollar
sign) to download the source files of Allium and the build environment Docker
image.

    $ git clone https://github.com/hrittich/allium.git
    $ docker pull ghcr.io/hrittich/allium-full

If you get a permission-denied error, you might have to prepend `sudo` to the
Docker command.

Running

    $ docker run --rm -tiv $PWD:/work ghcr.io/hrittich/allium-full shell

from your choosen directory starts and enters a development container.
Note that **everything you store outside of the `/work` directory will be
lost, when you exit the container**.

When you are inside a development container, you can build Allium and run
your first demo application with the following commands.

    $ cd allium
    $ cmake .
    $ make
    $ demo/poisson 128

[Docker]: https://www.docker.com/
[Git]: https://git-scm.com/

## Further Reading

- @subpage download
- @subpage install
- @subpage demos
