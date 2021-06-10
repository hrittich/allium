#!/bin/bash

PETSC_URL=""
VERSION=3.15.0
CHECKSUM=ac46db6bfcaaec8cd28335231076815bd5438f401a4a05e33736b4f9ff12e59a

function download {
  (wget --progress=dot:mega --tries=3 https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-${VERSION}.tar.gz ||
   wget --progress=dot:mega https://www.mcs.anl.gov/petsc/mirror/release-snapshots/petsc-${VERSION}.tar.gz) &&
  echo "${CHECKSUM}  petsc-${VERSION}.tar.gz" | sha256sum -c
}

function install {
  tar -axf petsc-${VERSION}.tar.gz && \
  (cd petsc-${VERSION} && \
   ./configure --prefix=/usr/local --with-scalar-type=complex --with-fortran-kernels=1 && \
   make && \
   sudo make install) && \
  rm -r petsc-${VERSION}
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

