#!/usr/bin/env bash

echo "Installing apt-get libsndfile"
sudo apt-get install -y libsndfile1-dev

echo Installing python dependencies...
pipenv install --pre --dev --deploy --skip-lock

echo Running training script...
pipenv run train-transcription