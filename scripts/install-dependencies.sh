#!/bin/bash
export SCRIPTDIR="$(dirname "$0")"

# Install the basic requirements
. "$SCRIPTDIR/install-minimal-dependencies.sh"

sudo apt-get install -y python3-dev python3-mpi4py
sudo apt-get install -y \
  doxygen \
  gfortran \
  ghostscript \
  graphviz \
  libatlas-base-dev \
  libgsl-dev \
  pkg-config \
  texlive-latex-base
