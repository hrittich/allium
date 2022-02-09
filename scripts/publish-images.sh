#!/bin/bash
SCRIPTDIR="$(dirname $0)"
source "$SCRIPTDIR/docker-vars.sh"

DOCKER="${DOCKER:-docker}"
DOCKER_REPOSITORY="ghcr.io/hrittich"
#VERSION="$(date +%Y-%m-%dT%H.%M)"

"$SCRIPTDIR/docker-build.sh"
$SUDO ${DOCKER} image tag allium-minimal "${DOCKER_REPOSITORY}/allium-minimal" &&
$SUDO ${DOCKER} image tag allium-full "${DOCKER_REPOSITORY}/allium-full" &&
$SUDO ${DOCKER} push "${DOCKER_REPOSITORY}/allium-minimal:latest" &&
$SUDO ${DOCKER} push "${DOCKER_REPOSITORY}/allium-full:latest"
