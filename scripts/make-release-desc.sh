#!/bin/bash
# Generate the description of a release: the previous release version and
# the change log.
#
# Usage: make-release-desc.sh <version>
#   <version>: current release version (v<maj>.<min>.<pat>, the leading v is optional)
#
# The previous version is the highest plain semver tag (v<maj>.<min>.<pat>)
# strictly below <version>. The change log lists all commits between the
# previous version tag and the release commit, one line per commit.
#
# The release commit is the commit <version> points at when the tag exists,
# HEAD otherwise.
#
# Env (when running in GitHub Actions):
#   GITHUB_OUTPUT: previous_tag, changelog_title and changelog are written here
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $(basename "$0") <version>"
    exit 1
fi
VERSION="$1"

# Accept the version with or without the leading v, reject anything else
if [[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    VERSION="v${VERSION}"
elif [[ ! "${VERSION}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: invalid version '${VERSION}' (expected v<maj>.<min>.<pat>)"
    exit 1
fi

# Make sure all remote tags are available locally (skipped on local runs without origin)
if ! git fetch --tags origin 2>/dev/null; then
    echo "Warning: could not fetch tags from origin (local run?)"
fi

# Release commit: the commit <version> points at when the tag exists, HEAD otherwise.
if ! RELEASE_COMMIT="$(git rev-parse -q --verify "refs/tags/${VERSION}^{commit}" 2>/dev/null)"; then
    RELEASE_COMMIT="$(git rev-parse HEAD)"
fi

echo "Release commit: $(git rev-parse --short "${RELEASE_COMMIT}")"

PREV="$( { git tag --list; echo "${VERSION}"; } \
    | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' \
    | sort -V \
    | awk -v cur="${VERSION}" '$0 == cur { exit } { prev = $0 } END { print prev }')"

if [[ -n "${PREV}" ]]; then
    CHANGELOG="$(git log --oneline "${PREV}..${RELEASE_COMMIT}")"
    CHANGELOG_TITLE="Changelog since ${PREV}"
else
    CHANGELOG="(no previous release tag found)"
    CHANGELOG_TITLE="Changelog"
fi

echo "Previous version: ${PREV:-none}"
echo "${CHANGELOG}"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    {
        echo "previous_tag=${PREV}"
        echo "changelog_title=${CHANGELOG_TITLE}"
        echo "changelog<<CHANGELOG_EOF"
        echo "${CHANGELOG}"
        echo "CHANGELOG_EOF"
    } >> "${GITHUB_OUTPUT}"
fi
