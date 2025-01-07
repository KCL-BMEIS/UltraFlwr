from ultralytics import YOLO
from extract_final_save_from_client import extract_results_path

from FedYOLO.config import HOME, SERVER_CONFIG

import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='baseline')
parser.add_argument('--strategy_name', type=str, default='FedAvg')
parser.add_argument('--client_num', type=int, default=1)
parser.add_argument('--scoring_style', type=str, default="client-client")

args = parser.parse_args()

dataset_name = args.dataset_name
strategy_name = args.strategy_name
client_num = args.client_num
scoring_style = args.scoring_style
num_rounds = SERVER_CONFIG['rounds']

def get_classwise_results_table(results):
    class_wise_results = {
        'precision': {results.names[idx]: results.results_dict['metrics/precision(B)'] for idx in results.names},
        'recall': {results.names[idx]: results.results_dict['metrics/recall(B)'] for idx in results.names},
        'mAP50': {results.names[idx]: results.results_dict['metrics/mAP50(B)'] for idx in results.names},
        'mAP50-95': {results.names[idx]: value for idx, value in enumerate(results.maps)}
    }
    table = pd.DataFrame(class_wise_results)
    table.index.name = 'class'
    return table

def client_client_metrics(client_number, dataset_name, strategy_name):

    logs_path = f"{HOME}/logs/client_{client_number}_log_{dataset_name}_{strategy_name}.txt"
    weights_path = extract_results_path(logs_path)
    weights = f"{HOME}/{weights_path}/weights/best.pt"
    model = YOLO(weights)
    results = model.val(data=f'{HOME}/datasets/{dataset_name}/partitions/client_{client_number}/data.yaml', verbose=True)
    table = get_classwise_results_table(results)
    table.to_csv(f"{HOME}/results/client_{client_number}_results_{dataset_name}_{strategy_name}.csv", index=True, index_label='class')

def client_server_metrics(client_number, dataset_name, strategy_name):

    logs_path = f"{HOME}/logs/client_{client_number}_log_{dataset_name}_{strategy_name}.txt"
    weights_path = extract_results_path(logs_path)
    weights = f"{HOME}/{weights_path}/weights/best.pt"
    model = YOLO(weights)
    results = model.val(data=f'{HOME}/datasets/{dataset_name}/data.yaml', verbose=True)
    table = get_classwise_results_table(results)
    table.to_csv(f"{HOME}/results/client_{client_number}_results_{dataset_name}_{strategy_name}_server.csv", index=True, index_label='class')

def server_client_metrics(client_number, dataset_name, strategy_name, num_rounds):

    weights_path = f"{HOME}/weights/model_round_{num_rounds}_{dataset_name}_Strategy_{strategy_name}.pt"
    server_model = YOLO(weights_path)
    normal_model = YOLO()

    # if strategy_name has 'head' in it, then we need to load the detection weights only
    if 'head' in strategy_name.lower():
        detection_weights = {k: v for k, v in server_model.model.state_dict().items() if k.startswith('model.detect')}
        normal_model.model.load_state_dict({**normal_model.model.state_dict(), **detection_weights}, strict=False)   
        server_model = normal_model 
    
    results = server_model.val(data=f'{HOME}/datasets/{dataset_name}/partitions/client_{client_number}/data.yaml', verbose=True)
    table = get_classwise_results_table(results)
    table.to_csv(f"{HOME}/results/server_client_{client_number}_results_{dataset_name}_{strategy_name}.csv", index=True, index_label='class')

def server_server_metrics(dataset_name, strategy_name, num_rounds):

    weights_path = f"{HOME}/weights/model_round_{num_rounds}_{dataset_name}_Strategy_{strategy_name}.pt"
    server_model = YOLO(weights_path)
    normal_model = YOLO()

    if 'head' in strategy_name.lower():
        detection_weights = {k: v for k, v in server_model.model.state_dict().items() if k.startswith('model.detect')}
        normal_model.model.load_state_dict({**normal_model.model.state_dict(), **detection_weights}, strict=False)   
        server_model = normal_model 
    
    results = server_model.val(data=f'{HOME}/datasets/{dataset_name}/data.yaml', verbose=True)
    table = get_classwise_results_table(results)
    table.to_csv(f"{HOME}/results/server_results_{dataset_name}_{strategy_name}.csv", index=True, index_label='class')

if scoring_style == "client-client":
    client_metrics_table = client_client_metrics(client_num, dataset_name, strategy_name)
elif scoring_style == "client-server":
    client_metrics_table = client_server_metrics(client_num, dataset_name, strategy_name)
elif scoring_style == "server-client":
    client_metrics_table = server_client_metrics(client_num, dataset_name, strategy_name, num_rounds)
elif scoring_style == "server-server":
    client_metrics_table = server_server_metrics(dataset_name, strategy_name, num_rounds)
else:
    raise ValueError(f"Invalid scoring_style: {scoring_style}")