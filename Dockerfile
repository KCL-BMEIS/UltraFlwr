# need Ulrtalytics 8.3.101 for YOLOv11
FROM ultralytics/ultralytics:8.3.101

ARG USER_ID
ARG GROUP_ID
ARG USER
ARG USER_HOME

RUN echo "Building with user: $USER, user ID: $USER_ID, group ID: $GROUP_ID"

# Create a non-root user
RUN addgroup --gid $GROUP_ID $USER && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER

# Temporarily switch to root for package installation
USER root

# Install dependencies
RUN apt update && \
    apt install -y lsof && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install prettytable==3.16.0 flwr==1.17.0

# Switch back to the non-root user
USER $USER

WORKDIR ${USER_HOME}
