#!/bin/bash

# if repo exists, remove it
if [ -d "2023-europe-space-weather" ]; then
  cd 2023-europe-space-weather
  git pull
  git checkout ground_psi
  cd ..
else
  git clone git@github.com:FrontierDevelopmentLab/2023-europe-space-weather.git --depth 1 --branch ground_psi
fi

# build dockerfile
docker build -t helio-ground -f Dockerfile .
