#!/usr/bin/env python3
"""
Script to organize and extract the validation results from YOLO training logs.

This script processes log files from multiple seeds and extracts the final validation 
metrics (all "all" rows that appear after "Validating") from each log file.

The script outputs results in both CSV format and a formatted table for easy comparison
across different seeds and clients, including precision, recall, mAP@0.5, and mAP@0.5:0.95.
"""

import os
import re
import csv
import pandas as pd
from pathlib import Path

def extract_validation_results(log_file_path):
    """
    Extract the last validation result from a YOLO training log file.
    
    This function looks for validation results that appear after "Validating" 
    and extracts only the final metrics row starting with "all".
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        dict: Dictionary containing the extracted metrics or None if not found
    """
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
        
        # Find all "Validating" sections and extract the "all" rows that follow
        validating_pattern = r'Validating.*?\.pt\.\.\.'
        validating_matches = list(re.finditer(validating_pattern, content, re.DOTALL))
        
        if not validating_matches:
            print(f"No validation sections found in {log_file_path}")
            return None
        
        # Get the content after the last "Validating" section
        last_validating_pos = validating_matches[-1].end()
        remaining_content = content[last_validating_pos:]
        
        # Find all "all" rows in the remaining content
        all_pattern = r'^\s*all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
        all_matches = list(re.finditer(all_pattern, remaining_content, re.MULTILINE))
        
        if not all_matches:
            print(f"No 'all' rows found after last validation in {log_file_path}")
            return None
        
        # Get only the last "all" row (most important validation result)
        last_match = all_matches[-1]
        
        result = {
            'images': int(last_match.group(1)),
            'instances': int(last_match.group(2)),
            'precision': float(last_match.group(3)),
            'recall': float(last_match.group(4)),
            'map50': float(last_match.group(5)),
            'map50_95': float(last_match.group(6))
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing {log_file_path}: {e}")
        return None

def process_log_directories(base_dir):
    """
    Process all log directories and extract results from all log files.
    
    Args:
        base_dir (str): Base directory containing the log directories
        
    Returns:
        list: List of dictionaries containing all results
    """
    results = []
    
    # Pattern for log directories
    log_dir_pattern = "logs_local_train_polypGen_f_seed_*"
    log_dirs = sorted(Path(base_dir).glob(log_dir_pattern))
    
    for log_dir in log_dirs:
        # Extract seed number from directory name
        seed_match = re.search(r'seed_(\d+)', log_dir.name)
        if not seed_match:
            continue
        seed = int(seed_match.group(1))
        
        # Process all .log files in the directory
        log_files = sorted(log_dir.glob("*.log"))
        
        for log_file in log_files:
            # Extract client/dataset info from filename
            filename = log_file.stem  # Remove .log extension
            
            # Parse different filename patterns
            if 'client_' in filename:
                client_match = re.search(r'client_(\d+)', filename)
                client_id = int(client_match.group(1)) if client_match else None
                dataset_type = f"client_{client_id}"
            elif 'full_dataset' in filename:
                client_id = None
                dataset_type = "full_dataset"
            else:
                client_id = None
                dataset_type = "unknown"
            
            # Extract metrics - now returns a single result
            metrics = extract_validation_results(str(log_file))
            
            if metrics:
                result = {
                    'seed': seed,
                    'dataset_type': dataset_type,
                    'client_id': client_id,
                    'log_file': str(log_file.relative_to(base_dir)),
                    **metrics
                }
                results.append(result)
                print(f"✓ Processed: {log_file.name} (Seed {seed})")
            else:
                print(f"✗ Failed to extract metrics from: {log_file.name}")
    
    return results

def save_results(results, base_dir):
    """
    Save results to both CSV and formatted text files.
    
    Args:
        results (list): List of result dictionaries
        base_dir (str): Base directory for saving files
    """
    if not results:
        print("No results to save!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by seed, then by dataset_type, then by client_id
    df = df.sort_values(['seed', 'dataset_type', 'client_id'], na_position='last')
    
    # Save to CSV
    csv_path = Path(base_dir) / "training_results_summary.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n📊 Results saved to: {csv_path}")
    
    # Create formatted text output
    txt_path = Path(base_dir) / "training_results_summary.txt"
    with open(txt_path, 'w') as f:
        f.write("YOLO Training Results Summary\n")
        f.write("=" * 80 + "\n\n")
        
        # Group by seed
        for seed in sorted(df['seed'].unique()):
            seed_data = df[df['seed'] == seed]
            f.write(f"SEED {seed}\n")
            f.write("-" * 40 + "\n")
            
            # Format as table
            f.write(f"{'Dataset':<15} {'Images':<8} {'Instances':<10} {'Precision':<10} {'Recall':<8} {'mAP@0.5':<8} {'mAP@0.5:0.95':<12}\n")
            f.write("-" * 85 + "\n")
            
            for _, row in seed_data.iterrows():
                f.write(f"{row['dataset_type']:<15} {row['images']:<8} {row['instances']:<10} "
                       f"{row['precision']:<10.4f} {row['recall']:<8.4f} {row['map50']:<8.4f} {row['map50_95']:<12.4f}\n")
            
            f.write("\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 40 + "\n")
        
        # Group by dataset_type for summary
        summary_stats = df.groupby('dataset_type')[['precision', 'recall', 'map50', 'map50_95']].agg(['mean', 'std'])
        
        for dataset_type in summary_stats.index:
            f.write(f"\n{dataset_type.upper()}\n")
            f.write("-" * 20 + "\n")
            stats = summary_stats.loc[dataset_type]
            
            for metric in ['precision', 'recall', 'map50', 'map50_95']:
                mean_val = stats[(metric, 'mean')]
                std_val = stats[(metric, 'std')]
                f.write(f"{metric:<12}: {mean_val:.4f} ± {std_val:.4f}\n")
    
    print(f"📝 Formatted results saved to: {txt_path}")
    
    # Display summary in console
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    
    for dataset_type in sorted(df['dataset_type'].unique()):
        subset = df[df['dataset_type'] == dataset_type]
        print(f"\n{dataset_type.upper()}:")
        print(f"  Precision:    {subset['precision'].mean():.4f} ± {subset['precision'].std():.4f}")
        print(f"  Recall:       {subset['recall'].mean():.4f} ± {subset['recall'].std():.4f}")
        print(f"  mAP@0.5:      {subset['map50'].mean():.4f} ± {subset['map50'].std():.4f}")
        print(f"  mAP@0.5:0.95: {subset['map50_95'].mean():.4f} ± {subset['map50_95'].std():.4f}")

def main():
    """Main function to run the log organization script."""
    # Base directory containing the log directories
    base_dir = "/nfs/home/yli/new_UltraFlwr_polypgen/p_012_fedavg_fedheadavg/UltraFlwr"
    
    print("🔍 Starting log file analysis...")
    print(f"📂 Base directory: {base_dir}")
    
    # Process all log files
    results = process_log_directories(base_dir)
    
    if results:
        print(f"\n✅ Successfully processed {len(results)} log files")
        save_results(results, base_dir)
    else:
        print("\n❌ No results found. Please check the log file paths and formats.")

if __name__ == "__main__":
    main()
