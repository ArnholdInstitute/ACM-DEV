#!/bin/bash

cd webpage
node_modules/.bin/webpack
cd ../
git add client.min.js
git commit
git push