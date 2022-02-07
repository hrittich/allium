#!/bin/bash
SCRIPTDIR="$(cd "$(dirname "$0")" && pwd -P)"
SOURCEDIR="$(cd "$SCRIPTDIR" && cd ../ && pwd -P)"
source "$SCRIPTDIR/docker-vars.sh"

# Mount the source dir
printf "Mounting source dir: %q\n" "$SOURCEDIR"
ARGS+=" --volume $SOURCEDIR:/work"

TYPE=$1
shift

$DOCKER run --rm -ti $ARGS allium-$TYPE $@
