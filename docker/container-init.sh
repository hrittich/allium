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
  sh|bash)
    # This section is needed for GitLab-CI and allows for execution of
    # arbitrary scripts in the container.
    # "$@" passes the arguments exactly as given
    exec "$@"
  ;;
  *)
    echo "Arguments needed : \"shell\"|\"tests\""
    echo "Arguments given  : $@"
  ;;
esac
