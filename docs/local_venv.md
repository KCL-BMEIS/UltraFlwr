# Local Simulation with Python venv

### Python venv

Check `FedYOLO/config.py` to see the default configurations 

1. Make a custom environment: `python -m venv ultravenv`
2. Activate the custom environment: `source ultravenv/bin/activate`
3. Clone the repository
4. `cd` into the repository: `cd UltraFlwr`   
5. pip install the requirements: `pip install -e .`

### Prepare Datasets

6. `cd` into the datasets folder: `cd datasets`
7. Make a directory for a specific dataset: `mkdir pills`
8. `cd` into the dataset folder: `cd pills`
9. Get data-set from Roboflow
10. Create a directory for the client specific datasets: `mkdir partitions`
11. Create the partitions
    - Go to the base of the clone: `cd ../../`
    - Create the splits: `python FedYOLO/data_partitioner/fed_split.py` 
      - To choose the dataset, change the `DATASET_NAME` parameter in the `FedYOLO/config.py` file


#### To Build Custom Dataset

Follow the style of roboflow downloads as mentioned in above steps.

![sample_dataset](../assets/sample_dataset.png)

### Training

12. For one-off: `./scripts/federated_train_and_test/run.sh`
    - For multiple Runs, reference: `bash scripts/federated_train_and_test/benchmark.sh`
    - For normal YOLO training on entire server dataset and client data partitions: `bash scripts/central_train_and_test/run_local_train_and_test.sh`