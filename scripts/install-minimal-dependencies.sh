#!/bin/bash
export DEBIAN_FRONTEND=noninteractive

if ! which sudo > /dev/null; then
  su -c "apt-get install -y sudo"
fi

sudo apt-get install -y wget gcc g++ cmake git
sudo apt-get install -y libopenmpi-dev
