# Local Simulation with Docker

## Clone the repo
First, clone the repo to local machine. Navigate to the root of the repo.

## Running Docker

Assuming the machine has a Nvidia GPU, first, [Nvidia Container Runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is needed.

Then, spin up a docker container using, update the path accordingly:

```bash
sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v <PATH_CONTAINING_ULTRAFLWR>/UltraFlwr:/UltraFlwr nvcr.io/nvidia/pytorch:24.05-py3
```

In the docker container, `cd` to `\UltraFlwr`.

Install Dependencies:

```bash
apt update
apt install lsof
pip install prettytable==3.12.0 flwr==1.14.0 ultralytics==8.3.53 opencv-python==4.8.0.74
```

**In the `setup.py`, remove the contents in `install_requires`.**

Check `FedYOLO/config.py` to see the default configurations, especially regarding `BASE`, `DATASET_NAME`, and `NUM_CLIENTS`.

If you are running docker following the command provided, `BASE` needs to be empty.

Lastly, install FedYOLO with `pip install -e .`

## Prepare Datasets

1. `cd` into the datasets folder: `cd datasets`
2. Make a directory for a specific dataset: `mkdir pills`
3. `cd` into the dataset folder: `cd pills`
4. Get data-set from Roboflow
5. Create a directory for the client specific datasets: `mkdir partitions`
6. Create the partitions
7. Go to the base of the clone: `cd ../../`
   - Create the splits: `python FedYOLO/data_partitioner/fed_split.py` 
   - To choose the dataset, change the `DATASET_NAME` parameter in the `FedYOLO/config.py` file


### To Build Custom Dataset

Follow the style of roboflow downloads as mentioned in above steps.

![sample_dataset](../assets/sample_dataset.png)

## Training

For one-off: `./scripts/run.sh`

For multiple Runs, reference: `bash scripts/federated_train_and_test/benchmark.sh`

For normal YOLO training on entire server dataset and client data partitions: `bash scripts/central_train_and_test/run_local_train_and_test.sh`