#!/bin/sh
# Prints the changelog section for one version from docs/DOCUMENTATION.md.
# Fails if the section is missing or empty, so a version bump without a
# changelog entry can't be released.
#
#   scripts/release-notes.sh 0.1.0
set -eu

version="${1:?usage: release-notes.sh <version>}"
doc="$(dirname "$0")/../docs/DOCUMENTATION.md"

notes="$(awk -v heading="### $version" '
    $0 == heading { found = 1; next }
    found && /^##/ { exit }
    found { print }
' "$doc")"

if [ -z "$(printf '%s' "$notes" | tr -d '[:space:]')" ]; then
    echo "No changelog entry \"### $version\" in docs/DOCUMENTATION.md (section 13)" >&2
    exit 1
fi
printf '%s\n' "$notes"
