from ultralytics import YOLO
from extract_final_save_from_client import extract_results_path
import os # Import os module
import sys
from tabulate import tabulate

# Import CLIENT_CONFIG
from FedYOLO.config import HOME, SERVER_CONFIG, CLIENT_CONFIG

import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='baseline')
parser.add_argument('--strategy_name', type=str, default='FedAvg')
parser.add_argument('--client_num', type=int, default=1)
parser.add_argument('--scoring_style', type=str, default="client-client")
parser.add_argument('--print_env_only', type=str, default='false')
parser.add_argument('--print_matrix_only', type=str, default='false')
parser.add_argument('--print_env', type=str, default='true')
parser.add_argument('--print_summary', type=str, default='true')
args = parser.parse_args()

dataset_name = args.dataset_name
strategy_name = args.strategy_name
client_num = args.client_num
scoring_style = args.scoring_style
num_rounds = SERVER_CONFIG['rounds']

# Convert string boolean arguments to actual booleans
print_env_only = args.print_env_only.lower() == 'true'
print_matrix_only = args.print_matrix_only.lower() == 'true'
print_env = args.print_env.lower() == 'true'
print_summary = args.print_summary.lower() == 'true'

# Create a test execution tracker
test_execution_summary = []
all_possible_tests = []

def print_header(message):
    """Print a formatted header."""
    print("\n" + "="*68)
    print(message)
    print("="*68)

def get_all_dataset_names():
    """Extract all dataset names from the client config."""
    datasets = set([client_cfg.get('dataset_name', 'unknown') for client_cfg in CLIENT_CONFIG.values()])
    return sorted(list(datasets))

def get_classwise_results_table(results):
    # Access precision, recall, and mAP values directly as arrays
    precision_values = results.box.p  # List/array of precision values for each class
    recall_values = results.box.r  # List/array of recall values for each class
    ap50_values = results.box.ap50  # Array of AP50 values for each class
    ap50_95_values = results.box.ap  # Array of AP50-95 values for each class

    # Ensure alignment between metrics and class names
    num_classes = min(len(results.names), len(precision_values))

    # Construct class-wise results table
    class_wise_results = {
        'precision': {results.names[idx]: precision_values[idx] for idx in range(num_classes)},
        'recall': {results.names[idx]: recall_values[idx] for idx in range(num_classes)},
        'mAP50': {results.names[idx]: ap50_values[idx] for idx in range(num_classes)},
        'mAP50-95': {results.names[idx]: ap50_95_values[idx] for idx in range(num_classes)}
    }

    # Calculate mean results (overall "all" row)
    mp, mr, map50, map5095 = results.box.mean_results()
    class_wise_results['precision']['all'] = mp
    class_wise_results['recall']['all'] = mr
    class_wise_results['mAP50']['all'] = map50
    class_wise_results['mAP50-95']['all'] = map5095

    # Convert to DataFrame
    table = pd.DataFrame(class_wise_results)
    table.index.name = 'class'

    return table


def is_multi_task_environment():
    """Check if clients are using different tasks."""
    tasks = set()
    for client_id, config in CLIENT_CONFIG.items():
        if 'task' in config:
            tasks.add(config['task'])
    
    return len(tasks) > 1


def is_partial_aggregation_strategy():
    """Check if the strategy uses partial aggregation."""
    # Strategies that use partial aggregation have it in their name
    partial_strategies = ['FedPAv', 'FedNeck', 'FedBackbone', 'FedHead', 'Partial', 'partial']
    return any(p in strategy_name for p in partial_strategies)


def client_client_metrics(client_number, dataset_name, strategy_name):
    test_name = f"Client {client_number} on own data"
    result = "Not Run"
    # Get client-specific configuration
    client_cfg = CLIENT_CONFIG[client_number]
    client_dataset_name = client_cfg['dataset_name']
    client_task = client_cfg.get('task', 'detect')  # Default to 'detect' if not specified
    
    # Construct the log path using the client's specific dataset name
    logs_path = f"{HOME}/logs/client_{client_number}_log_{client_dataset_name}_{strategy_name}.txt"
    try:
        weights_path = extract_results_path(logs_path)
        weights = f"{HOME}/{weights_path}/weights/best.pt"
        if not os.path.exists(weights):
             print(f"Weight file not found: {weights}. Skipping client-client test for client {client_number}, strategy {strategy_name}.")
             result = "Failed (Missing Weights)"
             test_execution_summary.append([test_name, "Client-Client", strategy_name, client_dataset_name, result])
             return None
        model = YOLO(weights)
        # Use the client's specific data path for validation
        results = model.val(data=f'{HOME}/datasets/{client_dataset_name}/partitions/client_{client_number}/data.yaml', split="test", verbose=True)
        table = get_classwise_results_table(results)
        # Save results with client-specific dataset name in the filename
        output_path = f"{HOME}/results/client_{client_number}_results_{client_dataset_name}_{strategy_name}.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure directory exists
        table.to_csv(output_path, index=True, index_label='class')
        print(f"Client-client results saved to {output_path}")
        result = "Success"
    except FileNotFoundError:
        print(f"Log file not found: {logs_path}. Skipping client-client test for client {client_number}, strategy {strategy_name}.")
        result = "Failed (Missing Logs)"
    except Exception as e:
        print(f"An error occurred during client-client test for client {client_number}, strategy {strategy_name}: {e}")
        result = f"Failed ({type(e).__name__})"
    
    test_execution_summary.append([test_name, "Client-Client", strategy_name, client_dataset_name, result])
    return None


def client_server_metrics(client_number, dataset_name, strategy_name):
    test_name = f"Client {client_number} on server data"
    result = "Not Run"
    # Skip if we're in a multi-task environment or using partial aggregation
    if is_multi_task_environment() or is_partial_aggregation_strategy():
        print(f"Skipping client-server test for client {client_number}: multi-task environment or partial aggregation strategy detected.")
        result = "Skipped (Multi-task/Partial Aggregation)"
        test_execution_summary.append([test_name, "Client-Server", strategy_name, dataset_name, result])
        return None
        
    # Get client-specific configuration for log path and task
    client_cfg = CLIENT_CONFIG[client_number]
    client_dataset_name = client_cfg['dataset_name']
    client_task = client_cfg.get('task', 'detect')  # Default to 'detect' if not specified

    # Task Mismatch Check
    if client_task == 'segment' and dataset_name == 'baseline':
        print(f"Skipping client-server test for client {client_number} (segment task) on dataset '{dataset_name}' (detect task). Incompatible.")
        result = "Skipped (Task Mismatch)"
        test_execution_summary.append([test_name, "Client-Server", strategy_name, dataset_name, result])
        return None
    
    # Construct the log path using the client's specific dataset name
    logs_path = f"{HOME}/logs/client_{client_number}_log_{client_dataset_name}_{strategy_name}.txt"
    try:
        weights_path = extract_results_path(logs_path)
        weights = f"{HOME}/{weights_path}/weights/best.pt"
        if not os.path.exists(weights):
             print(f"Weight file not found: {weights}. Skipping client-server test for client {client_number}, strategy {strategy_name}.")
             result = "Failed (Missing Weights)"
             test_execution_summary.append([test_name, "Client-Server", strategy_name, dataset_name, result])
             return None
        model = YOLO(weights)
        # Use the main dataset's test set (passed as dataset_name) for validation
        results = model.val(data=f'{HOME}/datasets/{dataset_name}/data.yaml', split="test", verbose=True)
        table = get_classwise_results_table(results)
        # Save results with client-specific dataset name and "_server" suffix in the filename
        output_path = f"{HOME}/results/client_{client_number}_results_{client_dataset_name}_{strategy_name}_server.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure directory exists
        table.to_csv(output_path, index=True, index_label='class')
        print(f"Client-server results saved to {output_path}")
        result = "Success"
    except FileNotFoundError:
        print(f"Log file not found: {logs_path}. Skipping client-server test for client {client_number}, strategy {strategy_name}.")
        result = "Failed (Missing Logs)"
    except IndexError as e:
         print(f"IndexError during client-server validation for client {client_number}, strategy {strategy_name} on dataset {dataset_name}: {e}. Likely task mismatch.")
         result = "Failed (Index Error - Task Mismatch)"
    except Exception as e:
        print(f"An error occurred during client-server test for client {client_number}, strategy {strategy_name}: {e}")
        result = f"Failed ({type(e).__name__})"
    
    test_execution_summary.append([test_name, "Client-Server", strategy_name, dataset_name, result])
    return None


def server_client_metrics(client_number, dataset_name, strategy_name, num_rounds):
    test_name = f"Server on Client {client_number} data"
    result = "Not Run"
    # Skip if we're in a multi-task environment or using partial aggregation
    if is_multi_task_environment() or is_partial_aggregation_strategy():
        print(f"Skipping server-client test for client {client_number}: multi-task environment or partial aggregation strategy detected.")
        result = "Skipped (Multi-task/Partial Aggregation)"
        client_cfg = CLIENT_CONFIG[client_number]
        client_dataset_name = client_cfg.get('dataset_name', 'unknown')
        test_execution_summary.append([test_name, "Server-Client", strategy_name, client_dataset_name, result])
        return None
        
    # Get client-specific configuration for data path
    client_cfg = CLIENT_CONFIG[client_number]
    client_dataset_name = client_cfg['dataset_name']
    client_data_path = client_cfg['data_path'] # Use the full path from config
    client_task = client_cfg.get('task', 'detect')  # Default to 'detect' if not specified

    # Task Mismatch Check - add the same check as in client_server_metrics
    if client_task == 'segment' and dataset_name == 'baseline':
        print(f"Skipping server-client test for client {client_number} (segment task) on dataset '{dataset_name}' (detect task). Incompatible.")
        result = "Skipped (Task Mismatch)"
        test_execution_summary.append([test_name, "Server-Client", strategy_name, client_dataset_name, result])
        return None

    # Server weights path uses the base dataset_name
    weights_path = f"{HOME}/weights/model_round_{num_rounds}_{dataset_name}_Strategy_{strategy_name}.pt"

    if not os.path.exists(weights_path):
        print(f"Server weight file not found: {weights_path}. Skipping server-client test for client {client_number}, strategy {strategy_name}.")
        result = "Failed (Missing Server Weights)"
        test_execution_summary.append([test_name, "Server-Client", strategy_name, client_dataset_name, result])
        return None

    try:
        server_model = YOLO(weights_path)
        # Use the client's specific data path for validation
        results = server_model.val(data=client_data_path, split="test", verbose=True)
        table = get_classwise_results_table(results)
        # Save results with client-specific dataset name in the filename
        output_path = f"{HOME}/results/server_client_{client_number}_results_{client_dataset_name}_{strategy_name}.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure directory exists
        table.to_csv(output_path, index=True, index_label='class')
        print(f"Server-client results saved to {output_path}")
        result = "Success"
    except IndexError as e:
         print(f"IndexError during server-client validation for client {client_number}, strategy {strategy_name}: {e}. Likely task mismatch between server model and client data.")
         result = "Failed (Index Error - Task Mismatch)"
    except Exception as e:
        print(f"An error occurred during server-client test for client {client_number}, strategy {strategy_name}: {e}")
        result = f"Failed ({type(e).__name__})"
    
    test_execution_summary.append([test_name, "Server-Client", strategy_name, client_dataset_name, result])
    return None


def server_server_metrics(dataset_name, strategy_name, num_rounds):
    test_name = f"Server on server data"
    result = "Not Run"
    # Skip if we're in a multi-task environment or using partial aggregation
    if is_multi_task_environment() or is_partial_aggregation_strategy():
        print(f"Skipping server-server test: multi-task environment or partial aggregation strategy detected.")
        result = "Skipped (Multi-task/Partial Aggregation)"
        test_execution_summary.append([test_name, "Server-Server", strategy_name, dataset_name, result])
        return None
        
    # Server weights path uses the base dataset_name
    weights_path = f"{HOME}/weights/model_round_{num_rounds}_{dataset_name}_Strategy_{strategy_name}.pt"

    if not os.path.exists(weights_path):
        print(f"Server weight file not found: {weights_path}. Skipping server-server test for strategy {strategy_name}.")
        result = "Failed (Missing Server Weights)"
        test_execution_summary.append([test_name, "Server-Server", strategy_name, dataset_name, result])
        return None

    try:
        server_model = YOLO(weights_path)
        # Use the main dataset's test set (passed as dataset_name) for validation
        results = server_model.val(data=f'{HOME}/datasets/{dataset_name}/data.yaml', split="test", verbose=True)
        table = get_classwise_results_table(results)
        # Save results with base dataset name
        output_path = f"{HOME}/results/server_results_{dataset_name}_{strategy_name}.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure directory exists
        table.to_csv(output_path, index=True, index_label='class')
        print(f"Server-server results saved to {output_path}")
        result = "Success"
    except IndexError as e:
         print(f"IndexError during server-server validation for strategy {strategy_name} on dataset {dataset_name}: {e}. Likely task mismatch.")
         result = "Failed (Index Error - Task Mismatch)"
    except Exception as e:
        print(f"An error occurred during server-server test for strategy {strategy_name}: {e}")
        result = f"Failed ({type(e).__name__})"
    
    test_execution_summary.append([test_name, "Server-Server", strategy_name, dataset_name, result])
    return None


def print_environment_summary():
    """Print a summary of the environment at the start."""
    all_datasets = get_all_dataset_names()
    multi_task = is_multi_task_environment()
    partial_agg = is_partial_aggregation_strategy()
    
    # Create summary table
    env_summary = [
        ["Base Dataset", dataset_name],
        ["Strategy", strategy_name],
        ["Total Rounds", num_rounds],
        ["Multi-Task Environment", "Yes" if multi_task else "No"],
        ["Partial Aggregation", "Yes" if partial_agg else "No"],
        ["Available Datasets", ", ".join(all_datasets)],
        ["Number of Clients", len(CLIENT_CONFIG)],
        ["Server Tests Enabled", "No" if (multi_task or partial_agg) else "Yes"]
    ]
    
    print_header("Test Environment Summary")
    print(tabulate(env_summary, tablefmt="grid"))
    
    # Client details
    client_details = []
    for client_id, config in CLIENT_CONFIG.items():
        task = config.get('task', 'detect')
        dataset = config.get('dataset_name', 'unknown')
        client_details.append([client_id, dataset, task])
    
    print("\nClient Configuration:")
    print(tabulate(client_details, headers=["Client ID", "Dataset", "Task"], tablefmt="grid"))


def build_test_matrix():
    """Build a matrix of all possible tests that could be executed."""
    # If we're in matrix-only mode, load all strategies from the result files
    if print_matrix_only:
        all_strategies = set()
        results_dir = f"{HOME}/test_results"
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                if filename.startswith("test_results_"):
                    parts = filename.split('_')
                    if len(parts) > 3:
                        all_strategies.add(parts[3])  # Extract strategy name
    else:
        all_strategies = [strategy_name]
    
    # Clear the existing list
    all_possible_tests.clear()
    
    # Build tests for each strategy
    for strategy in all_strategies:
        # Client-Client tests
        for client_id in CLIENT_CONFIG.keys():
            dataset_name = CLIENT_CONFIG[client_id].get('dataset_name', 'unknown')
            all_possible_tests.append([f"Client {client_id} on own data", "Client-Client", strategy, dataset_name, "Not Run"])
        
        # Client-Server tests
        for client_id in CLIENT_CONFIG.keys():
            all_possible_tests.append([f"Client {client_id} on server data", "Client-Server", strategy, dataset_name, "Not Run"])
        
        # Server-Client tests
        for client_id in CLIENT_CONFIG.keys():
            client_dataset = CLIENT_CONFIG[client_id].get('dataset_name', 'unknown')
            all_possible_tests.append([f"Server on Client {client_id} data", "Server-Client", strategy, client_dataset, "Not Run"])
        
        # Server-Server test
        all_possible_tests.append([f"Server on server data", "Server-Server", strategy, dataset_name, "Not Run"])
    

def print_test_matrix():
    """Print a comprehensive test matrix showing all possible tests and their execution status."""
    print_header("Test Matrix Summary")
    
    # Load all results if in matrix-only mode
    if print_matrix_only:
        global test_execution_summary
        test_execution_summary = load_all_test_results()
    
    # Group tests by experiment type
    client_client_tests = []
    client_server_tests = []
    server_client_tests = []
    server_server_tests = []
    
    # Identify executed tests
    executed_tests = {(t[0], t[1]) for t in test_execution_summary}
    
    # Organize all possible tests by type and mark executed ones
    for test in all_possible_tests:
        test_name, test_type, strategy, dataset, _ = test
        
        # Check if this test was executed and get its result
        status = "Not Run"
        for executed_test in test_execution_summary:
            if executed_test[0] == test_name and executed_test[1] == test_type:
                status = executed_test[4]  # Get the result
                break
        
        # Group by test type
        if test_type == "Client-Client":
            client_client_tests.append([test_name, status])
        elif test_type == "Client-Server":
            client_server_tests.append([test_name, status])
        elif test_type == "Server-Client":
            server_client_tests.append([test_name, status])
        elif test_type == "Server-Server":
            server_server_tests.append([test_name, status])
    
    # Create and print the summary table
    print("\nExperiment Type: Client-Client")
    print(tabulate(client_client_tests, headers=["Specific Experiment", "Status"], tablefmt="grid"))
    
    print("\nExperiment Type: Client-Server")
    print(tabulate(client_server_tests, headers=["Specific Experiment", "Status"], tablefmt="grid"))
    
    print("\nExperiment Type: Server-Client")
    print(tabulate(server_client_tests, headers=["Specific Experiment", "Status"], tablefmt="grid"))
    
    print("\nExperiment Type: Server-Server")
    print(tabulate(server_server_tests, headers=["Specific Experiment", "Status"], tablefmt="grid"))


def print_test_summary():
    """Print the summary of the executed test."""
    print_header("Test Execution Details")
    if test_execution_summary:
        for test in test_execution_summary:
            if test[4] == "Success":
                print(f"✅ {test[1]}: {test[0]} using {test[2]} on {test[3]} - {test[4]}")
            elif "Failed" in test[4]:
                print(f"❌ {test[1]}: {test[0]} using {test[2]} on {test[3]} - {test[4]}")
            else:
                print(f"⚠️ {test[1]}: {test[0]} using {test[2]} on {test[3]} - {test[4]}")
    else:
        print("No tests were executed.")


import json
import os

def save_test_results():
    """Save current test execution results to a file."""
    results_dir = f"{HOME}/test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Use a unique filename based on test parameters
    filename = f"{results_dir}/test_results_{dataset_name}_{strategy_name}_{scoring_style}"
    if scoring_style in ["client-client", "client-server", "server-client"]:
        filename += f"_client{client_num}"
    filename += ".json"
    
    # Save the test execution summary
    with open(filename, 'w') as f:
        json.dump(test_execution_summary, f)
    
def load_all_test_results():
    """Load all test results from files and combine them."""
    results_dir = f"{HOME}/test_results"
    if not os.path.exists(results_dir):
        return []
        
    all_results = []
    for filename in os.listdir(results_dir):
        if filename.startswith("test_results_") and filename.endswith(".json"):
            try:
                with open(f"{results_dir}/{filename}", 'r') as f:
                    results = json.load(f)
                    all_results.extend(results)
            except Exception as e:
                print(f"Error loading results from {filename}: {e}")
    
    return all_results

def execute_test_by_scoring_style(scoring_style, client_num, dataset_name, strategy_name, num_rounds, skip_server_tests):
    """
    Execute the appropriate test based on the specified scoring style.
    
    Args:
        scoring_style: Type of test to run (client-client, client-server, server-client, server-server).
        client_num: Client number to test.
        dataset_name: Name of the dataset being tested.
        strategy_name: Name of the federated learning strategy.
        num_rounds: Number of training rounds.
        skip_server_tests: Whether to skip server-based tests.
        
    Returns:
        None
    
    Note: 
        Results are added to the global test_execution_summary list.
    """
    if scoring_style == "client-client":
        print_header(f"Running client-dependent test: client_num={client_num}, scoring_style=client-client")
        client_client_metrics(client_num, dataset_name, strategy_name)
    elif scoring_style == "client-server":
        print_header(f"Running client-dependent test: client_num={client_num}, scoring_style=client-server")
        if skip_server_tests:
            print(f"Skipping client-server test: multi-task environment or partial aggregation strategy detected.")
            test_execution_summary.append([
                f"Client {client_num} on server data", 
                "Client-Server",
                strategy_name, 
                dataset_name, 
                "Skipped (Multi-task/Partial Aggregation)"
            ])
        else:
            client_server_metrics(client_num, dataset_name, strategy_name)
    elif scoring_style == "server-client":
        print_header(f"Running server-client test: client_num={client_num}")
        if skip_server_tests:
            print(f"Skipping server-client test for STRATEGY={strategy_name} (multi-task or partial aggregation)")
            client_cfg = CLIENT_CONFIG[client_num]
            client_dataset_name = client_cfg.get('dataset_name', 'unknown')
            test_execution_summary.append([
                f"Server on Client {client_num} data", 
                "Server-Client",
                strategy_name, 
                client_dataset_name, 
                "Skipped (Multi-task/Partial Aggregation)"
            ])
        else:
            server_client_metrics(client_num, dataset_name, strategy_name, num_rounds)
    elif scoring_style == "server-server":
        print_header(f"Running server-server test")
        if skip_server_tests:
            print(f"Skipping server-server test for STRATEGY={strategy_name} (multi-task or partial aggregation)")
            test_execution_summary.append([
                "Server on server data", 
                "Server-Server",
                strategy_name, 
                dataset_name, 
                "Skipped (Multi-task/Partial Aggregation)"
            ])
        else:
            server_server_metrics(dataset_name, strategy_name, num_rounds)
    else:
        if skip_server_tests:
            print(f"Skipping {scoring_style} test: multi-task environment or partial aggregation strategy detected.")
        else:
            raise ValueError(f"Invalid scoring_style: {scoring_style}")

def mark_skipped_tests(skip_server_tests, current_scoring_style, dataset_name, strategy_name):
    """
    Mark tests as skipped in the test_execution_summary when using multi-task or partial aggregation.
    
    Args:
        skip_server_tests: Boolean indicating whether to skip server-based tests.
        current_scoring_style: The current test scoring style being executed.
        dataset_name: Name of the dataset being tested.
        strategy_name: Name of the federated learning strategy.
        
    Returns:
        None
        
    Note:
        Modifies the global test_execution_summary list.
    """
    if not skip_server_tests:
        return
        
    # Mark server-client tests as skipped
    if current_scoring_style != "server-client":
        for client_id in CLIENT_CONFIG.keys():
            client_cfg = CLIENT_CONFIG[client_id]
            client_dataset_name = client_cfg.get('dataset_name', 'unknown')
            entry = [
                f"Server on Client {client_id} data", 
                "Server-Client",
                strategy_name, 
                client_dataset_name, 
                "Skipped (Multi-task/Partial Aggregation)"
            ]
            # Only add if not already in the summary
            if not any(t[0] == entry[0] and t[1] == entry[1] for t in test_execution_summary):
                test_execution_summary.append(entry)

    # Mark server-server test as skipped
    if current_scoring_style != "server-server":
        entry = [
            "Server on server data", 
            "Server-Server",
            strategy_name, 
            dataset_name, 
            "Skipped (Multi-task/Partial Aggregation)"
        ]
        if not any(t[0] == entry[0] and t[1] == entry[1] for t in test_execution_summary):
            test_execution_summary.append(entry)

# Build the test matrix before running any tests
build_test_matrix()

# Determine if we should run based on the environment
multi_task = is_multi_task_environment()
partial_agg = is_partial_aggregation_strategy()
skip_server_tests = multi_task or partial_agg

# Special modes for printing just environment or matrix
if print_env_only:
    print_environment_summary()
    sys.exit(0)

if print_matrix_only:
    print_test_matrix()
    sys.exit(0)

# Regular test execution
if print_env:
    print_environment_summary()

if skip_server_tests:
    print_header(f"Skipping server-based tests for STRATEGY={strategy_name} (multi-task or partial aggregation)")

# Execute the appropriate test based on scoring style
execute_test_by_scoring_style(scoring_style, client_num, dataset_name, strategy_name, num_rounds, skip_server_tests)

# Mark tests as skipped in multi-task/partial aggregation environments
mark_skipped_tests(skip_server_tests, scoring_style, dataset_name, strategy_name)
    
# Save the test results to file for future summaries
save_test_results()

# Print brief execution summary at the end of each individual test run
if print_summary:
    print_test_summary()