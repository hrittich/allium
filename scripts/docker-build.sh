#!/bin/bash
SCRIPTDIR="$(cd "$(dirname "$0")" && pwd -P)"
DOCKERDIR="$SCRIPTDIR/docker"
source "$SCRIPTDIR/docker-vars.sh"

$DOCKER build $DOCKER_BUILD_FLAGS -f "$DOCKERDIR/minimal_dev.dockerfile" -t allium-minimal "$SCRIPTDIR" &&
$DOCKER build $DOCKER_BUILD_FLAGS -f "$DOCKERDIR/full_dev.dockerfile" -t allium-full "$SCRIPTDIR"
