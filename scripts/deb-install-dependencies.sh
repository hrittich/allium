#!/bin/bash
export DEBIAN_FRONTEND=noninteractive

function minimal_install {
  if ! which sudo > /dev/null; then
    su -c "apt-get install -y sudo"
  fi

  sudo apt-get install -y wget gcc g++ cmake git
  sudo apt-get install -y libopenmpi-dev
}

function full_install {
  minimal_install
  sudo apt-get install -y python3-dev python3-mpi4py
  sudo apt-get install -y libatlas-base-dev gfortran pkg-config
}

case "$1" in
  minimal) minimal_install;;
  full) full_install;;
  *)
    echo "ERROR: Invalid configuration"
    exit 1
esac

