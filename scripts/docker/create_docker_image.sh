#!/usr/bin/env bash

set -euo pipefail  # Exit on error, unset var, or pipeline failure

# Navigate to the script's directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$SCRIPTPATH"
cd ../../

# Prompt for Docker image tag
read -p "Enter Docker tag (e.g., your_dockerhub_username/your_image_name:latest): " docker_tag

# Detect current user details
USER_ID=$(id -u)
GROUP_ID=$(id -g)
USER_NAME=$(id -un)
USER_HOME=$(eval echo ~${USER_NAME})

echo ""
echo "🛠️  Building Docker image with the following config:"
echo "Docker tag:     ${docker_tag}"
echo "User:           ${USER_NAME}"
echo "UID:            ${USER_ID}"
echo "GID:            ${GROUP_ID}"
echo "Home directory: ${USER_HOME}"
echo ""

# Build Docker image with correct UID, GID, and WORKDIR
docker build . -f Dockerfile \
  --network=host \
  --tag "${docker_tag}" \
  --build-arg USER_ID="${USER_ID}" \
  --build-arg GROUP_ID="${GROUP_ID}" \
  --build-arg USER="${USER_NAME}" \
  --build-arg USER_HOME="${USER_HOME}"

# Push the image
docker push "${docker_tag}"
