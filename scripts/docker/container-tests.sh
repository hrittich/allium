#!/bin/bash
set -x

TYPE="${1:-default}"
case "$TYPE" in
  default)
    BUILD_TYPE=Debug
  ;;
  release)
    BUILD_TYPE=Release
  ;;
  *)
    echo "Invalid test type $TYPE"
    exit 1
esac

cd $HOME
git clone source build &&
cd build &&
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" . &&
make -j $(nproc) &&
tests/test_suite
