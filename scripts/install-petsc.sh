#!/bin/bash

PETSC_URL=""

function download {
  (wget --progress=dot:mega --tries=3 http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-3.13.3.tar.gz ||
   wget --progress=dot:mega https://www.mcs.anl.gov/petsc/mirror/release-snapshots/petsc-lite-3.13.3.tar.gz) &&
  echo "dc744895ee6b9c4491ff817bef0d3abd680c5e3c25e601be44240ce65ab4f337  petsc-lite-3.13.3.tar.gz" | sha256sum -c
}

function install {
  tar -axf petsc-lite-3.13.3.tar.gz && \
  (cd petsc-3.13.3 && \
   ./configure --prefix=/usr/local --with-scalar-type=complex --with-fortran-kernels=1 && \
   make && \
   sudo make install) && \
  rm -r petsc-3.13.3
}

CMD="${1:-all}"
case "$CMD" in
  download) download;;
  install) install;;
  all)
    download &&
    install
  ;;
esac

