#!/bin/bash
set -x

cd $HOME
git clone source build &&
cd build &&
cmake . &&
make -j $MAKE_JOBS &&
tests/test_suite
