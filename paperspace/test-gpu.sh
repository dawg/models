#!/usr/bin/env bash

echo "Creating track separation job..."
paperspace jobs create --container "dawgpeople/models:latest" --machineType "GPU+" --command "nvidia-smi"
echo "Done!"
