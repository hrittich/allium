#!/bin/bash
export SCRIPTDIR="$(dirname "$0")"

# Install the basic requirements
. "$SCRIPTDIR/install-minimal-dependencies.sh"

sudo apt-get install -y python3-dev python3-mpi4py
sudo apt-get install -y libatlas-base-dev gfortran pkg-config

