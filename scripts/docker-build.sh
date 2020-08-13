#!/bin/bash
SCRIPTDIR="$(cd "$(dirname "$0")" && pwd -P)"
DOCKERDIR="$SCRIPTDIR/docker"
source "$SCRIPTDIR/docker-vars.sh"

$SUDO docker build -f "$DOCKERDIR/minimal_dev.dockerfile" -t allium-minimal "$SCRIPTDIR" &&
$SUDO docker build -f "$DOCKERDIR/full_dev.dockerfile" -t allium-full "$SCRIPTDIR"

#CID=$($SUDO docker create allium)
#$SUDO docker cp $CID:/var/cache/apt apt-cache
#$SUDO docker rm $CID
