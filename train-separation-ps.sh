#!/usr/bin/env bash

# echo Logging into Paperspace
# paperspace login
# echo Done!

echo Creating track separation job...
paperspace jobs create --container dawgpeople/models --command "python vusic/separation/scripts/train.py"
echo Done!
