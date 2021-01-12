#!/bin/bash
SCRIPTDIR="$(dirname "$0")"
set -e
for CONF in minimal full; do
  "$SCRIPTDIR/docker-start.sh" $CONF tests
done
"$SCRIPTDIR/docker-start.sh" full tests release
