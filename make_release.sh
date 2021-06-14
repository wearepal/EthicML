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

if [ $(git symbolic-ref --short -q HEAD) != "master" ]; then
  echo "not on master branch"
  exit 3
fi

echo ""
echo "######################################"
echo "# ensure master branch is up-to-date #"
echo "######################################"
git pull

echo ""
echo "######################################"
echo "#       checkout release branch      #"
echo "######################################"
git checkout release

echo ""
echo "#######################################"
echo "# ensure release branch is up-to-date #"
echo "#######################################"
git pull

echo ""
echo "#######################################"
echo "#   merge master into release branch  #"
echo "#######################################"
git merge --no-ff master --no-edit

# bump version
poetry version $version_bump

# commit change
git add pyproject.toml
git commit -m "Bump version"

# create tag and push
new_tag=v$(poetry version -s)
echo "#######################################"
echo "#          new tag: $new_tag          #"
echo "#######################################"
git tag $new_tag
git push origin release $new_tag

# clean previous build and build
echo "#######################################"
echo "#        clean up old builds          #"
echo "#######################################"
rm -rf build dist

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

# clean up
echo "#######################################"
echo "#      go back to master branch       #"
echo "#######################################"
git checkout master

echo "#####################################################"
echo "#               all done! now go to                 #"
echo "# https://github.com/predictive-analytics-lab/EthicML/releases/tag/$new_tag"
echo "# and click on \"Edit Tag\" to write release notes  #"
echo "#####################################################"
