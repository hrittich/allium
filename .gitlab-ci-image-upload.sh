#!/bin/bash
SCRIPTDIR="$(dirname $0)"
source "$SCRIPTDIR/scripts/docker-vars.sh"

DOCKER=${DOCKER:-docker}
DOCKER_REPOSITORY="gitlab.version.fz-juelich.de:5555/rittich2/allium"
#VERSION="$(date +%Y-%m-%dT%H.%M)"

scripts/docker-build.sh
$SUDO ${DOCKER} image tag allium-minimal "${DOCKER_REPOSITORY}/allium-minimal" &&
$SUDO ${DOCKER} image tag allium-full "${DOCKER_REPOSITORY}/allium-full" &&
$SUDO ${DOCKER} push "${DOCKER_REPOSITORY}/allium-minimal" &&
$SUDO ${DOCKER} push "${DOCKER_REPOSITORY}/allium-full"
