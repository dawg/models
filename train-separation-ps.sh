#!/usr/bin/env bash

echo Creating track separation job...
paperspace jobs create --container dawgpeople/models --command "python vusic/separation/scripts/train.py"
echo Done!