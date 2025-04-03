#!/usr/bin/env bash

set -e  # Exit if any command fails

# Navigate to the script's directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$SCRIPTPATH"
cd ../../

# Prompt the user for the Docker tag
read -p "Enter Docker tag (e.g., your_dockerhub_username/your_image_name:latest): " docker_tag

# Confirm input
echo "Building and pushing image: ${docker_tag}"

# Build the Docker image
docker build . -f Dockerfile \
  --network=host \
  --tag "${docker_tag}" \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg USER="username"

# Push the image
docker push "${docker_tag}"
