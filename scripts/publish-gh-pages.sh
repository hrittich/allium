#!/bin/bash
set -xe

git switch -c gh-pages

cmake -S . -B build
make -C build api-docs

if [ -z "$(git config --get user.name)" ]
then
  git config --global user.email "publish@bots.github.com"
  git config --global user.name "Publish Bot"
fi

git add --force build/doc/
git commit -m 'Publish Pages'
#git remote add origin "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
#git push --force --set-upstream origin gh-pages
git subtree push -P gh-pages origin gh-pages


