FROM allium-minimal

COPY install-dependencies.sh $SCRIPTDIR
RUN $SCRIPTDIR/install-dependencies.sh

COPY install-petsc.sh $SCRIPTDIR/
WORKDIR /usr/local/src
RUN $SCRIPTDIR/install-petsc.sh download
RUN $SCRIPTDIR/install-petsc.sh install

