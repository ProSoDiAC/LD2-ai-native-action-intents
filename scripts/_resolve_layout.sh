#!/usr/bin/env bash
# Repository layout: LD2/{scripts,XAIR_Runtime,config,journal}.
# Sets: REPO_ROOT, SCRIPTS, XAIR_ROOT
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -d "$_here/../XAIR_Runtime/xair" ]; then
  echo "ERROR: expected XAIR_Runtime/ next to scripts/" >&2
  exit 1
fi
REPO_ROOT="$(cd "$_here/.." && pwd)"
SCRIPTS="$_here"
XAIR_ROOT="$REPO_ROOT/XAIR_Runtime"
export REPO_ROOT SCRIPTS XAIR_ROOT
unset _here
