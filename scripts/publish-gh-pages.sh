#!/bin/bash
set -xe

commit="$(git log -1 --format="%h")"

cmake -S . -B build
make -C build api-docs

if [ -z "$(git config --get user.name)" ]
then
  git config --global user.email "publish@bots.github.com"
  git config --global user.name "Publish Bot"
fi

git add --force build/doc/html
git commit -m "Documentation for $commit."
git subtree split --branch=gh-pages --prefix=build/doc/html
git push --force origin gh-pages
