#!/bin/bash

git pull 
git add *.py *.sh ./frozen_graph/*.sh ./frozen_graph/*.py
git commit -m "Update"
git push
