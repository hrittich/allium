#!/bin/false
# This file is supposed to be sourced

# The docker executable
DOCKER=${DOCKER:-docker}

# Check if we need root
if ! $DOCKER version &> /dev/null; then
  DOCKER="sudo \"$DOCKER"\"

  # Check if we have access with sudo
  if ! $DOCKER version &> /dev/null; then
    echo "Cannot access Docker"
    exit 1
  fi
fi
