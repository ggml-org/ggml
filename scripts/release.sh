#!/bin/bash
#
# Automated release script for ggml.
#
# Note: Sync from llama.cpp should be done separately via PR process
# prior to running this script.
#
# Usage:
#   ./scripts/release.sh [major|minor|patch] [--dry-run]
#
# Example usage:
# $ ./scripts/release.sh minor --dry-run
#
# This will show what the actions that would be taken to increment the minor
# version of the project.
#
# This script:
# 1. Updates version and removes -dev suffix
# 2. Commits the version bump
# 3. Creates a git tag
# 4. Prepares for next development cycle
#

set -e

if [ ! -f "CMakeLists.txt" ] || [ ! -d "scripts" ]; then
    echo "Error: Must be run from ggml root directory"
    exit 1
fi

# Parse command line arguments
VERSION_TYPE=""
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        major|minor|patch)
            VERSION_TYPE="$arg"
            shift
            ;;
        *)
            echo "Error: Unknown argument '$arg'"
            echo "Usage: $0 [major|minor|patch] [--dry-run]"
            exit 1
            ;;
    esac
done

# Default to patch if no version type specified
VERSION_TYPE="${VERSION_TYPE:-patch}"

if [[ ! "$VERSION_TYPE" =~ ^(major|minor|patch)$ ]]; then
    echo "Error: Version type must be 'major', 'minor', or 'patch'"
    echo "Usage: $0 [major|minor|patch] [--dry-run]"
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    echo "[dry-run] - No changes will be made"
    echo ""
else
    echo "Starting automated release process..."
fi

# Check for uncommitted changes (skip in dry-run)
if [ "$DRY_RUN" = false ] && ! git diff-index --quiet HEAD --; then
    echo "Error: You have uncommitted changes. Please commit or stash them first."
    exit 1
fi

# Ensure we're on master branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "master" ]; then
    if [ "$DRY_RUN" = true ]; then
        echo "[dry run] Warning: Not on master branch (currently on: $CURRENT_BRANCH). Continuing with dry-run..."
        echo ""
    else
        echo "Error: Must be on master branch to create release. Currently on: $CURRENT_BRANCH"
        exit 1
    fi
fi

# Check if we have the latest from master (skip in dry-run)
if [ "$DRY_RUN" = false ]; then
    echo "Checking if local master is up-to-date with remote..."
    git fetch origin master
    LOCAL=$(git rev-parse HEAD)
    REMOTE=$(git rev-parse origin/master)

    if [ "$LOCAL" != "$REMOTE" ]; then
        echo "Error: Your local master branch is not up-to-date with origin/master."
        echo "Please run 'git pull origin master' first."
        exit 1
    fi
    echo "✓ Local master is up-to-date with remote"
    echo ""
elif [ "$CURRENT_BRANCH" = "master" ]; then
    echo "[dry run] Warning: Dry-run mode - not checking if master is up-to-date with remote"
    echo ""
fi

# Extract current version from CMakeLists.txt
echo "Step 1: Reading current version..."
MAJOR=$(grep "set(GGML_VERSION_MAJOR" CMakeLists.txt | sed 's/.*MAJOR \([0-9]*\).*/\1/')
MINOR=$(grep "set(GGML_VERSION_MINOR" CMakeLists.txt | sed 's/.*MINOR \([0-9]*\).*/\1/')
PATCH=$(grep "set(GGML_VERSION_PATCH" CMakeLists.txt | sed 's/.*PATCH \([0-9]*\).*/\1/')

echo "Current version: $MAJOR.$MINOR.$PATCH-dev"

# Calculate new version
case $VERSION_TYPE in
    major)
        NEW_MAJOR=$((MAJOR + 1))
        NEW_MINOR=0
        NEW_PATCH=0
        ;;
    minor)
        NEW_MAJOR=$MAJOR
        NEW_MINOR=$((MINOR + 1))
        NEW_PATCH=0
        ;;
    patch)
        NEW_MAJOR=$MAJOR
        NEW_MINOR=$MINOR
        NEW_PATCH=$((PATCH + 1))
        ;;
esac

NEW_VERSION="$NEW_MAJOR.$NEW_MINOR.$NEW_PATCH"
RC_BRANCH="ggml-rc-v$NEW_VERSION"
echo "New release version: $NEW_VERSION"
echo "Release candidate branch: $RC_BRANCH"
echo ""

# Create release candidate branch
echo "Step 2: Creating release candidate branch..."
if [ "$DRY_RUN" = true ]; then
    echo "  [dry-run] Would create branch: $RC_BRANCH"
else
    git checkout -b "$RC_BRANCH"
    echo "✓ Created and switched to branch: $RC_BRANCH"
fi
echo ""

# Update CMakeLists.txt for release
echo "Step 3: Updating version in CMakeLists.txt..."
if [ "$DRY_RUN" = true ]; then
    echo "  [dry-run] Would update GGML_VERSION_MAJOR to $NEW_MAJOR"
    echo "  [dry-run] Would update GGML_VERSION_MINOR to $NEW_MINOR"
    echo "  [dry-run] Would update GGML_VERSION_PATCH to $NEW_PATCH"
    echo "  [dry-run] Would remove -dev suffix"
else
    sed -i'' -e "s/set(GGML_VERSION_MAJOR [0-9]*)/set(GGML_VERSION_MAJOR $NEW_MAJOR)/" CMakeLists.txt
    sed -i'' -e "s/set(GGML_VERSION_MINOR [0-9]*)/set(GGML_VERSION_MINOR $NEW_MINOR)/" CMakeLists.txt
    sed -i'' -e "s/set(GGML_VERSION_PATCH [0-9]*)/set(GGML_VERSION_PATCH $NEW_PATCH)/" CMakeLists.txt
    sed -i'' -e 's/set(GGML_VERSION_DEV "-dev")/set(GGML_VERSION_DEV "")/' CMakeLists.txt
fi
echo ""

# Commit version bump
echo "Step 4: Committing version bump..."
if [ "$DRY_RUN" = true ]; then
    echo "  [dry-run] Would commit: 'ggml : bump version to $NEW_VERSION'"
else
    git add CMakeLists.txt
    git commit -m "ggml : bump version to $NEW_VERSION"
fi
echo ""

# Create git tag
echo "Step 5: Creating signed git tag..."
if [ "$DRY_RUN" = true ]; then
    echo "  [dry-run] Would create signed tag: v$NEW_VERSION with message 'Release version $NEW_VERSION'"
else
    git tag -s "v$NEW_VERSION" -m "Release version $NEW_VERSION"
    echo "✓ Created signed tag: v$NEW_VERSION"
fi
echo ""

# Prepare for next development cycle
echo "Step 6: Preparing for next development cycle..."
case $VERSION_TYPE in
    major|minor)
        NEXT_DEV_MINOR=$((NEW_MINOR))
        NEXT_DEV_VERSION="$NEW_MAJOR.$NEXT_DEV_MINOR.0-dev"
        if [ "$DRY_RUN" = true ]; then
            echo "  [dry-run] Would update GGML_VERSION_MINOR to $NEXT_DEV_MINOR"
        else
            sed -i'' -e "s/set(GGML_VERSION_MINOR [0-9]*)/set(GGML_VERSION_MINOR $NEXT_DEV_MINOR)/" CMakeLists.txt
        fi
        ;;
    patch)
        NEXT_DEV_PATCH=$((NEW_PATCH))
        NEXT_DEV_VERSION="$NEW_MAJOR.$NEW_MINOR.$NEXT_DEV_PATCH-dev"
        if [ "$DRY_RUN" = true ]; then
            echo "  [dry-run] Would update GGML_VERSION_PATCH to $NEXT_DEV_PATCH"
        else
            sed -i'' -e "s/set(GGML_VERSION_PATCH [0-9]*)/set(GGML_VERSION_PATCH $NEXT_DEV_PATCH)/" CMakeLists.txt
        fi
        ;;
esac

if [ "$DRY_RUN" = true ]; then
    echo "  [dry-run] Would add -dev suffix back"
else
    sed -i'' -e 's/set(GGML_VERSION_DEV "")/set(GGML_VERSION_DEV "-dev")/' CMakeLists.txt
fi
echo ""

# Commit development version
echo "Step 7: Committing development version..."
if [ "$DRY_RUN" = true ]; then
    echo "  [dry-run] Would commit: 'ggml : prepare for development of $NEXT_DEV_VERSION'"
else
    git add CMakeLists.txt
    git commit -m "ggml : prepare for development of $NEXT_DEV_VERSION"
fi

echo ""
if [ "$DRY_RUN" = true ]; then
    echo "[dry-run] Summary (no changes were made):"
    echo "  • Would have created branch: $RC_BRANCH"
    echo "  • Would have created tag: v$NEW_VERSION"
    echo "  • Would have set next development version: $NEXT_DEV_VERSION"
else
    echo "Release process completed!"
    echo "Summary:"
    echo "  • Created branch: $RC_BRANCH"
    echo "  • Created tag: v$NEW_VERSION"
    echo "  • Next development version: $NEXT_DEV_VERSION"
fi
if [ "$DRY_RUN" = false ]; then
    echo "Next steps:"
    echo "  • Review the commits and tag on branch $RC_BRANCH"
    echo "  • Push branch to remote: git push origin $RC_BRANCH"
    echo "  • Create a Pull Request from $RC_BRANCH to master"
    echo "  • After PR is merged, push the tag: git push origin v$NEW_VERSION"
    echo "  • The release will be completed once the tag is pushed"
    echo ""
fi
