#!/usr/bin/env bash
JOB_NAME=<YOUR_RUNAI_JOB_NAME>
runai submit $JOB_NAME \
--backoff-limit 0 \
--image "<YOUR_DOCKERHUB_USERNAME>/<YOUR_IMAGE_TAG>" \
--gpu 1 \
--project <YOUR_RUNAI_PROJECT_NAME> \
--large-shm \
--cpu 1 \
-v <PATH_TO_REPO>:/nfs/home/testuser \
--command \
-- bash /nfs/home/testuser/fedyolo/test/test.sh \
--run-as-user
# --command \
# -- bash /nfs/home/testuser/fedyolo/test/test.sh \
# --command \
# -- bash /nfs/home/testuser/scripts/benchmark.sh \
# --interactive -- sleep infinity \
