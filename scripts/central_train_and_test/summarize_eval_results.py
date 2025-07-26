#!/usr/bin/env python3
"""
Script to summarize evaluation results from logs_local_eval_polypGen_f directory.
Extracts and organizes the "all" metrics from YOLO evaluation log files.
"""

import os
import re
import pandas as pd
from pathlib import Path

def parse_log_file(filepath):
    """Parse a single log file and extract the 'all' metrics."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Look for the line containing "all" metrics
        # Format: "                   all        115        116      0.809      0.767      0.795      0.648"
        pattern = r'\s+all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
        match = re.search(pattern, content)
        
        if match:
            return {
                'images': int(match.group(1)),
                'labels': int(match.group(2)),
                'precision': float(match.group(3)),
                'recall': float(match.group(4)),
                'mAP_50': float(match.group(5)),
                'mAP_50_95': float(match.group(6))
            }
        else:
            print(f"Warning: Could not parse metrics from {filepath}")
            return None
            
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    # Directory containing the log files
    log_dir = Path("logs_local_eval_polypGen_f")
    
    if not log_dir.exists():
        print(f"Error: Directory {log_dir} does not exist")
        return
    
    # Parse all log files
    results = []
    
    for log_file in sorted(log_dir.glob("*.log")):
        filename = log_file.name
        
        # Extract seed and evaluation target from filename
        # Format: seed_X_on_Y.log
        match = re.match(r'seed_(\d+)_on_(.+)\.log', filename)
        if match:
            seed = int(match.group(1))
            target = match.group(2)
            
            metrics = parse_log_file(log_file)
            if metrics:
                result = {
                    'filename': filename,
                    'seed': seed,
                    'target': target,
                    **metrics
                }
                results.append(result)
    
    if not results:
        print("No valid results found")
        return
    
    # Convert to DataFrame for better organization
    df = pd.DataFrame(results)
    
    # Sort by seed and target
    df = df.sort_values(['seed', 'target'])
    
    print("="*80)
    print("YOLO Evaluation Results Summary")
    print("="*80)
    print()
    
    # Display full results table
    print("Complete Results:")
    print("-" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(df.to_string(index=False))
    print()
    
    # Group by seed and calculate statistics
    print("Results by Seed:")
    print("-" * 40)
    for seed in sorted(df['seed'].unique()):
        seed_data = df[df['seed'] == seed]
        print(f"\nSeed {seed}:")
        print(seed_data[['target', 'images', 'labels', 'precision', 'recall', 'mAP_50', 'mAP_50_95']].to_string(index=False))
    
    print()
    
    # Group by target (client_0, client_1, client_2, full_dataset)
    print("Results by Target:")
    print("-" * 40)
    for target in sorted(df['target'].unique()):
        target_data = df[df['target'] == target]
        print(f"\nTarget: {target}")
        print(f"Average across seeds:")
        means = target_data[['images', 'labels', 'precision', 'recall', 'mAP_50', 'mAP_50_95']].mean()
        stds = target_data[['precision', 'recall', 'mAP_50', 'mAP_50_95']].std()
        
        print(f"  Images: {means['images']:.0f}")
        print(f"  Labels: {means['labels']:.0f}")
        print(f"  Precision: {means['precision']:.3f} ± {stds['precision']:.3f}")
        print(f"  Recall: {means['recall']:.3f} ± {stds['recall']:.3f}")
        print(f"  mAP@0.5: {means['mAP_50']:.3f} ± {stds['mAP_50']:.3f}")
        print(f"  mAP@0.5:0.95: {means['mAP_50_95']:.3f} ± {stds['mAP_50_95']:.3f}")
    
    print()
    
    # Overall summary
    print("Overall Summary Across All Seeds and Targets:")
    print("-" * 50)
    overall_means = df[['precision', 'recall', 'mAP_50', 'mAP_50_95']].mean()
    overall_stds = df[['precision', 'recall', 'mAP_50', 'mAP_50_95']].std()
    
    print(f"Precision: {overall_means['precision']:.3f} ± {overall_stds['precision']:.3f}")
    print(f"Recall: {overall_means['recall']:.3f} ± {overall_stds['recall']:.3f}")
    print(f"mAP@0.5: {overall_means['mAP_50']:.3f} ± {overall_stds['mAP_50']:.3f}")
    print(f"mAP@0.5:0.95: {overall_means['mAP_50_95']:.3f} ± {overall_stds['mAP_50_95']:.3f}")
    
    # Save to CSV
    output_file = "evaluation_results_summary.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Save summary statistics
    summary_file = "evaluation_summary_stats.csv"
    summary_stats = []
    
    for target in sorted(df['target'].unique()):
        target_data = df[df['target'] == target]
        means = target_data[['images', 'labels', 'precision', 'recall', 'mAP_50', 'mAP_50_95']].mean()
        stds = target_data[['precision', 'recall', 'mAP_50', 'mAP_50_95']].std()
        
        summary_stats.append({
            'target': target,
            'images_avg': means['images'],
            'labels_avg': means['labels'],
            'precision_mean': means['precision'],
            'precision_std': stds['precision'],
            'recall_mean': means['recall'],
            'recall_std': stds['recall'],
            'mAP_50_mean': means['mAP_50'],
            'mAP_50_std': stds['mAP_50'],
            'mAP_50_95_mean': means['mAP_50_95'],
            'mAP_50_95_std': stds['mAP_50_95']
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to: {summary_file}")

if __name__ == "__main__":
    main()
