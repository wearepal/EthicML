#!/bin/bash

# fail on error
set -e

# confirm the supplied version bump is valid
version_bump=$1

case $version_bump in
  "patch" | "minor" | "major" | "prepatch" | "preminor" | "premajor" | "prerelease")
    echo "valid version bump: $version_bump"
    ;;
  *)
    echo "invalid version bump: \"$version_bump\""
    echo "Usage: bash make_release.sh <version bump>"
    echo ""
    echo "List of valid version bumps: patch, minor, major, prepatch, preminor, premajor, prerelease"
    exit 1
    ;;
esac

if [ -n "$(git status --untracked-files=no --porcelain)" ]; then
  echo "The repository has uncommitted changes."
  echo "This will lead to problems with git checkout."
  exit 2
fi

if [ $(git symbolic-ref --short -q HEAD) != "master" ]; then
  echo "not on master branch"
  exit 3
fi

echo ensure master branch is up-to-date
git pull

echo checkout release branch
git checkout release
echo ensure release branch is up-to-date
git pull
echo merge master into release branch
git merge --no-ff master --no-edit

# bump version
poetry version $version_bump

# commit change
git add pyproject.toml
git commit -m "Bump version"

# create tag and push
new_tag=v$(poetry version -s)
echo New tag: $new_tag
git tag $new_tag
git push origin release $new_tag

# clean previous build and build
echo "clean up old builds"
rm -rf build dist
echo "do new build"
poetry build
echo "publish package"
# to use this, set up an API token with `poetry config pypi-token.pypi <api token>`
poetry publish

# clean up
echo "go back to master branch"
git checkout master
