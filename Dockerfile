# need Ulrtalytics 8.3.101 for YOLOv11
FROM ultralytics/ultralytics:8.3.101

ARG USER_ID
ARG GROUP_ID
ARG USER

RUN echo "Building with user: "$USER", user ID: "$USER_ID", group ID: "$GROUP_ID

RUN addgroup --gid $GROUP_ID $USER && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER

# set working directory
WORKDIR /nfs/home/$USER

# Install dependencies
RUN apt update && \
    apt install -y lsof && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install prettytable==3.16.0 flwr==1.17.0
