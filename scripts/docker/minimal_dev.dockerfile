FROM debian:bullseye
ENV SCRIPTDIR=/usr/local/share/container/scripts

# Update APT cache and install basic tools
RUN \
  apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y vim tmux

COPY install-minimal-dependencies.sh $SCRIPTDIR/
RUN $SCRIPTDIR/install-minimal-dependencies.sh

RUN mkdir /work && chown 1000:1000 /work

COPY docker/container-init.sh /usr/local/sbin
RUN chmod u+x /usr/local/sbin/container-init.sh

COPY docker/container-tests.sh /usr/local/bin
RUN chmod a+x /usr/local/bin/container-tests.sh

ENTRYPOINT ["/usr/local/sbin/container-init.sh"]
