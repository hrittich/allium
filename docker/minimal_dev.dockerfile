FROM debian:buster

RUN \
  apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    vim tmux wget gcc g++ cmake sudo git

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libopenmpi-dev

RUN \
  DEBIAN_FRONTEND=noninteractive apt-get install -y python3-dev python3-mpi4py

RUN mkdir -p /home/developer

COPY container-init.sh /usr/local/sbin
RUN chmod u+x /usr/local/sbin/container-init.sh

COPY container-tests.sh /usr/local/bin
RUN chmod a+x /usr/local/bin/container-tests.sh

ENTRYPOINT ["/usr/local/sbin/container-init.sh"]

