#!/bin/bash
SCRIPTDIR="$(cd "$(dirname "$0")" && pwd -P)"
DOCKERDIR="$SCRIPTDIR/.."
source "$SCRIPTDIR/sudo.sh"
#mkdir -p apt-cache

$SUDO docker build -f "$DOCKERDIR/minimal_dev.dockerfile" -t allium-minimal "$DOCKERDIR"
$SUDO docker build -f "$DOCKERDIR/full_dev.dockerfile" -t allium-full "$DOCKERDIR"

#CID=$($SUDO docker create allium)
#$SUDO docker cp $CID:/var/cache/apt apt-cache
#$SUDO docker rm $CID
