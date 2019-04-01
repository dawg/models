#!/usr/bin/env bash


# Script to test out the gpu drivers contained in the docker image
echo "Creating gpu test job..."
paperspace jobs create --container "dawgpeople/models:latest" --machineType "GPU+" --command "nvidia-smi"
echo "Done!"
