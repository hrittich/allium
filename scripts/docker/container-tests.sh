#!/bin/bash
set -x

cd $HOME
git clone source build &&
cd build &&
cmake . &&
make -j $(nproc) &&
tests/test_suite
