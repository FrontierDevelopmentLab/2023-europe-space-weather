#!/bin/bash

# if repo exists, remove it
if [ -d "2023-europe-space-weather" ]; then
  rm -rf 2023-europe-space-weather
fi

# clone
git clone git@github.com:FrontierDevelopmentLab/2023-europe-space-weather.git --depth 1 --branch ground_psi

# build dockerfile
docker build -t helio-ground -f Dockerfile .
