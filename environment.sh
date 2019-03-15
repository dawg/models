#!/usr/bin/env bash

command_exists () {
  type "$1" &> /dev/null
}

if [ "$VIRTUAL_ENV" != "" ]; then
  echo "Please do not this script from within a virtual environment!"
  exit 1
fi

OS="`uname`"
case ${OS} in
  'Darwin')
    echo "This script is not currently usable for MacOS!"
    exit 1
    ;;
  *) ;;
esac

echo "Installing apt-get dependencies"
sudo apt-get install -y build-essential libasound2-dev libjack-dev >/dev/null


if ! command_exists pipenv; then
  echo "Installing pipenv"
  pip install --quiet pipenv
fi

echo "Installing python modules"
pipenv install --pre --dev --deploy --skip-lock