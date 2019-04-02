#!/usr/bin/env bash

# Run training job on paperspace

# echo Logging into Paperspace
# paperspace login
# echo Done!

echo "Creating track separation job..."
paperspace jobs create --container "dawgpeople/models:latest" --machineType "GPU+" --command "./train-separation.sh"
echo "Done!"
