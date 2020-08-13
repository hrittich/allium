FROM allium-minimal

RUN $SCRIPTDIR/deb-install-dependencies.sh full

COPY install-petsc.sh $SCRIPTDIR/
WORKDIR /usr/local/src
RUN $SCRIPTDIR/install-petsc.sh download
RUN $SCRIPTDIR/install-petsc.sh install

