#!/bin/bash

set -xe
REPO="$(pwd)"

cmake .
make api-docs

TMPDIR="$(mktemp -d --tmpdir gh-pagesXXXXXXXX)"
cp -r doc/html/ -T "$TMPDIR"

cd "$TMPDIR"
git config --global user.email "publish@bots.github.com"
git config --global user.name "Publish Bot"
git init .
git switch -c gh-pages
git add .
git commit -m 'Publish'
git remote add origin "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git push -u origin gh-pages


