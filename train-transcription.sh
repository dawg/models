#!/usr/bin/env bash

echo Installing python dependencies...
pipenv install --pre --dev --deploy --skip-lock

echo Running training script...
pipenv run train-transcription