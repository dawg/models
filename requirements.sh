#!/usr/bin/env bash

command_exists () {
  type "$1" &> /dev/null
}

pipenv run install --dev

TF_VERSION=1.11.0
if command_exists nvcc; then
  pipenv run pip install tensorflow-gpu==${TF_VERSION}
  pipenv run pip install magenta-gpu
else
  pipenv run pip install tensorflow==${TF_VERSION}
  pipenv run pip install magenta
fi
