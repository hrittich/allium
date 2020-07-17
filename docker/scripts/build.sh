#!/bin/bash
SCRIPTDIR="$(cd "$(dirname "$0")" && pwd -P)"
DOCKERDIR="$SCRIPTDIR/.."

if (id -nG | grep docker > /dev/null); then
  SUDO=""
else
  SUDO="sudo"
fi

#mkdir -p apt-cache

$SUDO docker build -f "$DOCKERDIR/minimal_dev.dockerfile" -t chive-minimal "$DOCKERDIR"
$SUDO docker build -f "$DOCKERDIR/full_dev.dockerfile" -t chive-full "$DOCKERDIR"

#CID=$($SUDO docker create chive)
#$SUDO docker cp $CID:/var/cache/apt apt-cache
#$SUDO docker rm $CID
