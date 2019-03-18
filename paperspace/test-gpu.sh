#!/usr/bin/env bash

echo "Creating track separation job..."
paperspace jobs create --container Test-Container --machineType "GPU+" --command "nvidia-smi"
echo "Done!"
