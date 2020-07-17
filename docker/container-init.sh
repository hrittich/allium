#!/bin/bash
USER_UID=${USER_UID:-1000}
USER_GID=${USER_GID:-1000}
MAKE_JOBS=${MAKE_JOBS:-1}

groupadd -g "$USER_GID" developer
useradd -g "$USER_GID" -u "$USER_UID" developer
chown developer:developer /home/developer
sudo -u developer cp -rn /etc/skel/. /home/developer/

cd /home/developer

case "$1" in
  shell)
    exec sudo -HE -u developer bash
  ;;
  tests)
    exec sudo -HE -u developer container-tests.sh
  ;;
  *)
    echo "Needed argument: \"shell\"|\"tests\""
  ;;
esac
