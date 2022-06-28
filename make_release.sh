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
    echo "Usage:"
    echo ""
    echo "    bash make_release.sh <version bump>"
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

if [ $(git symbolic-ref --short -q HEAD) != "0.x" ]; then
  echo "not on 0.x branch"
  exit 3
fi

echo ""
echo "######################################"
echo "# ensure 0.x branch is up-to-date  #"
echo "######################################"
git pull

echo "#######################################"
echo "#            bump version             #"
echo "#######################################"
poetry version $version_bump
new_version=$(poetry version -s)

echo "#######################################"
echo "#            do new build             #"
echo "#######################################"
poetry build

echo ""
echo "#######################################"
echo "#          publish package            #"
echo "#######################################"
# to use this, set up an API token with
#  `poetry config pypi-token.pypi <api token>`
poetry publish

echo "#######################################"
echo "#         create new branch           #"
echo "#######################################"
branch_name=release-$new_version
git checkout -b $branch_name

echo "#######################################"
echo "#       commit version change         #"
echo "#######################################"
git add pyproject.toml
git commit -m "Bump version"

echo "#######################################"
echo "#          new tag: $new_tag          #"
echo "#######################################"
new_tag=v${new_version}
git tag $new_tag

echo "#######################################"
echo "#      bump prerelease version        #"
echo "#######################################"
poetry version prerelease

echo "#######################################"
echo "#       commit version change         #"
echo "#######################################"
git add pyproject.toml
git commit -m "Bump version to prerelease"

echo "#######################################"
echo "#       commit version change         #"
echo "#######################################"
git push origin $branch_name $new_tag

echo "#######################################"
echo "#     create PR for version bump      #"
echo "#######################################"
gh pr create --fill --base "0.x"

echo "#######################################"
echo "#           create release            #"
echo "#######################################"
gh release create $new_tag --generate-notes

# clean up
echo "#######################################"
echo "#      go back to 0.x branch         #"
echo "#######################################"
git checkout 0.x
