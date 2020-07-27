#!/bin/false
# This file is supposed to be sourced

if [ "$(id -nu)" == "root" ] || (id -nG | grep docker > /dev/null); then
  SUDO=""
else
  SUDO="sudo"
fi

