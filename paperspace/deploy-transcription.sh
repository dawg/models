#!/usr/bin/env bash

# Run training job on paperspace

# echo Logging into Paperspace
# paperspace login
# echo Done!

echo "Creating piano transcription job..."
paperspace jobs create --container "dawgpeople/models:latest" --machineType "P5000" --command "./train-transcription.sh"
echo "Done!"
