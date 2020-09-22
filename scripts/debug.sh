#!/bin/sh
#
#  Copyright 2020 Hannah Rittich
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Poore women's version of a parallel debugger. Requires OpenMPI or
#  Parastation MPI to work.

START_DEBUGER=false
PERSIST=false
RUN=false
QUIT=false
MYRANK=undefined
GDB=gdb
SOURCE=

XTERM=${XTERM:-xterm}

# Use OpenMPI rank if defined
MYRANK=${OMPI_COMM_WORLD_RANK:-$MYRANK}
# Use ParastationMPI rank if defined
MYRANK=${PMI_RANK:-$MYRANK}

#echo "Rank: $MYRANK"
LOAD_STATE_RANK="$MYRANK"

READING=true
while $READING && [ "$#" -gt 0 ]; do
  case "$1" in
    -a|-all)
      # Start debugger for all ranks
      START_DEBUGER=true
      shift
    ;;
    -run)
      RUN=true
      shift
    ;;
    -quit)
      QUIT=true
      shift
    ;;
    -p|-persist)
      PERSIST=true
      shift
    ;;
    -state-from)
      LOAD_STATE_RANK="$2"
      shift 2
    ;;
    -x|-command)
      SOURCE="$2"
      shift 2
    ;;
    -r|-ranks)
      # Comma separated list of ranks for which to start a debugger
      for RANK in $(echo "$2" | tr "," " "); do
        if [ "$MYRANK" -eq "$RANK" ]; then
          START_DEBUGER=true
        fi
      done
      shift 2
    ;;
    -c|-cgdb)
      XTERM_FLAGS="$XTERM_FLAGS -geometry 100x40"
      GDB=cgdb
      shift
    ;;
    --)
      shift
      READING=false
    ;;
    *)
      READING=false
    ;;
  esac
done

if $START_DEBUGER; then
  INIT_FILE=$(tempfile) || exit 1

  STATE_FILE="gdb-$MYRANK.state"
  LSTATE_FILE="gdb-$LOAD_STATE_RANK.state"
    cat << EOF >> $INIT_FILE
define save-state
  save breakpoints $STATE_FILE
  echo Wrote state to $STATE_FILE\n
end
EOF

  if $PERSIST; then
    cat << EOF >> $INIT_FILE
define hook-quit
  save-state
end
EOF
    [ -e "$LSTATE_FILE" ] && echo "source $LSTATE_FILE" >> $INIT_FILE
  fi

  [ -n "$SOURCE" ] && echo "source $SOURCE" >> $INIT_FILE

  echo "echo MPI rank: $MYRANK\\\\n" >> $INIT_FILE

  if $RUN; then
    echo "run" >> $INIT_FILE
  fi

  if $QUIT; then
    echo "quit" >> $INIT_FILE
  fi
  #cat $INIT_FILE

  $XTERM $XTERM_FLAGS -e "$GDB" -x $INIT_FILE --args "$@"
  rm $INIT_FILE
else
  exec $@
fi

