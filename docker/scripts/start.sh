#!/bin/bash
SCRIPTDIR="$(cd "$(dirname "$0")" && pwd -P)"
SOURCEDIR="$(cd "$SCRIPTDIR" && cd ../../ && pwd -P)"

if (id -nG | grep docker > /dev/null); then
  SUDO=""
else
  SUDO="sudo"
fi

# Variables for the user creation
USER_UID=$(id -u)
USER_GID=$(id -g)
ARGS=""
ARGS+=" --env USER_UID=$USER_UID --env USER_GID=$USER_GID"

# Create a virtual home directory
# USER_HOME=$HOME/.docker-home
# mkdir -p $USER_HOME
# ARGS+=" --volume $USER_HOME:/home/developer"

# Mount the document dir
#if [ -z "$XDG_DOCUMENTS_DIR" ]; then
#  XDG_DOCUMENTS_DIR="$HOME/Documents"
#fi
#mkdir -p $XDG_DOCUMENTS_DIR
#ARGS+=" --volume $XDG_DOCUMENTS_DIR:/home/developer/Documents"

# Mount the source dir
printf "Mounting source dir: %q\n" "$SOURCEDIR"
ARGS+=" --volume $SOURCEDIR:/home/developer/source"

TYPE=$1
shift

CID=$($SUDO docker create --rm -ti $ARGS chive-$TYPE $@) &&
$SUDO docker start -i $CID

