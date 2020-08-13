FROM debian:buster
ENV SCRIPTDIR=/usr/local/share/container/scripts

# Update APT cache and install basic tools
RUN \
  apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y vim tmux

COPY deb-install-dependencies.sh $SCRIPTDIR/
RUN $SCRIPTDIR/deb-install-dependencies.sh minimal

RUN mkdir -p /home/developer

COPY docker/container-init.sh /usr/local/sbin
RUN chmod u+x /usr/local/sbin/container-init.sh

COPY docker/container-tests.sh /usr/local/bin
RUN chmod a+x /usr/local/bin/container-tests.sh

ENTRYPOINT ["/usr/local/sbin/container-init.sh"]
