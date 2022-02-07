#!/bin/bash
MAKE_JOBS=${MAKE_JOBS:-1}

# Create a user owning the /work directory
USER_UID=$(stat -c '%u' /work)
USER_GID=$(stat -c '%g' /work)
if [ "$USER_UID" -ne 0 ]
then
  groupadd -g "$USER_GID" developer
  useradd -m -g "$USER_GID" -u "$USER_UID" developer
fi

cd /work

case "$1" in
  shell)
    exec sudo -HE -u "#$USER_UID" bash
  ;;
  tests)
    exec sudo -HE -u "#$USER_UID" container-tests.sh "$2"
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
