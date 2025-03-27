#!/usr/bin/env python
"""
Simple comparison script to directly compare results from baseline and enhanced models.
This script specifically works with the directory structure as it exists.
"""

import os
import pandas as pd
import numpy as np
from tabulate import tabulate
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Use relative paths - try different possible locations based on where the script is run from
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define possible locations for baseline results
possible_baseline_dirs = [
    os.path.join(project_root, "baseline_results"),
    "baseline_results",
    "../baseline_results",
    "../../baseline_results"
]

# Define possible locations for enhanced results
possible_enhanced_dirs = [
    os.path.join(project_root, "results"),
    "results",
    "../results",
    "../../results"
]

# Find the first existing baseline and enhanced directories
BASELINE_DIR = next((d for d in possible_baseline_dirs if os.path.exists(d)), 
                   possible_baseline_dirs[0])
ENHANCED_DIR = next((d for d in possible_enhanced_dirs if os.path.exists(d)), 
                   possible_enhanced_dirs[0])

logger.info(f"Using baseline results from: {BASELINE_DIR}")
logger.info(f"Using enhanced results from: {ENHANCED_DIR}")

# Create output directory for analysis results
ANALYSIS_DIR = os.path.join(ENHANCED_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)
logger.info(f"Analysis results will be saved to: {ANALYSIS_DIR}")

# Define file paths based on found directories
BASELINE_FILES = {
    "pytorch": os.path.join(BASELINE_DIR, "pytorch_NB.csv"),
    "tensorflow": os.path.join(BASELINE_DIR, "tensorflow_NB.csv"),
    "keras": os.path.join(BASELINE_DIR, "keras_NB.csv"),
    "incubator-mxnet": os.path.join(BASELINE_DIR, "incubator-mxnet_NB.csv"),
    "caffe": os.path.join(BASELINE_DIR, "caffe_NB.csv")
}

ENHANCED_FILES = {
    "pytorch": os.path.join(ENHANCED_DIR, "pytorch_SVM.csv"),
    "tensorflow": os.path.join(ENHANCED_DIR, "tensorflow_SVM.csv"),
    "keras": os.path.join(ENHANCED_DIR, "keras_SVM.csv"),
    "incubator-mxnet": os.path.join(ENHANCED_DIR, "incubator-mxnet_SVM.csv"),
    "caffe": os.path.join(ENHANCED_DIR, "caffe_SVM.csv")
}

# Metrics to compare
METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUC"]

def load_results_direct(file_path):
    """Load results directly from a file path."""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Take the first row if there are multiple rows
            return df.iloc[0].to_dict()
        else:
            logger.warning(f"File not found - {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

def main():
    """Main function to run the comparison."""
    logger.info("Loading results directly from files...")
    
    # Prepare data for table
    comparison_data = []
    
    for dataset in BASELINE_FILES.keys():
        baseline_file = BASELINE_FILES[dataset]
        enhanced_file = ENHANCED_FILES[dataset]
        
        baseline_results = load_results_direct(baseline_file)
        enhanced_results = load_results_direct(enhanced_file)
        
        if baseline_results and enhanced_results:
            row = [dataset]
            
            for metric in METRICS:
                if metric in baseline_results and metric in enhanced_results:
                    baseline_value = baseline_results[metric]
                    enhanced_value = enhanced_results[metric]
                    improvement = enhanced_value - baseline_value
                    percentage = (improvement / baseline_value) * 100 if baseline_value != 0 else 0
                    
                    row.append(f"{baseline_value:.4f}")
                    row.append(f"{enhanced_value:.4f}")
                    row.append(f"{improvement:.4f}")
                    row.append(f"{percentage:.2f}%")
            
            comparison_data.append(row)
    
    # Generate headers for table
    headers = ["Dataset"]
    for metric in METRICS:
        headers.extend([
            f"{metric} (NB)", 
            f"{metric} (SVM)", 
            f"{metric} Diff", 
            f"{metric} Improvement %"
        ])
    
    # Print comparison table
    logger.info("\n=== PERFORMANCE COMPARISON: BASELINE (NB) vs ENHANCED (SVM) ===\n")
    table = tabulate(comparison_data, headers=headers, tablefmt="grid")
    logger.info(f"\n{table}")
    
    # Save comparison table to file
    with open(os.path.join(ANALYSIS_DIR, "comparison_table.txt"), "w") as f:
        f.write("=== PERFORMANCE COMPARISON: BASELINE (NB) vs ENHANCED (SVM) ===\n\n")
        f.write(table)
    
    # Calculate average improvements across datasets
    avg_improvements = {}
    for metric in METRICS:
        values = []
        
        for dataset in BASELINE_FILES.keys():
            baseline_file = BASELINE_FILES[dataset]
            enhanced_file = ENHANCED_FILES[dataset]
            
            baseline_results = load_results_direct(baseline_file)
            enhanced_results = load_results_direct(enhanced_file)
            
            if baseline_results and enhanced_results and metric in baseline_results and metric in enhanced_results:
                baseline_value = baseline_results[metric]
                enhanced_value = enhanced_results[metric]
                improvement = enhanced_value - baseline_value
                percentage = (improvement / baseline_value) * 100 if baseline_value != 0 else 0
                values.append(percentage)
        
        if values:
            avg_improvements[metric] = np.mean(values)
    
    # Print summary of average improvements
    logger.info("\n=== AVERAGE PERCENTAGE IMPROVEMENTS ACROSS ALL DATASETS ===\n")
    
    avg_data = []
    for metric, improvement in avg_improvements.items():
        avg_data.append([metric, f"{improvement:.2f}%"])
    
    avg_table = tabulate(avg_data, headers=["Metric", "Average Improvement %"], tablefmt="grid")
    logger.info(f"\n{avg_table}")
    
    # Save average improvements to file
    with open(os.path.join(ANALYSIS_DIR, "average_improvements.txt"), "w") as f:
        f.write("=== AVERAGE PERCENTAGE IMPROVEMENTS ACROSS ALL DATASETS ===\n\n")
        f.write(avg_table)
    
    # Identify which metrics consistently improved
    improved_metrics = [metric for metric, avg in avg_improvements.items() if avg > 0]
    
    conclusion = f"\nMetrics with consistent improvement: {', '.join(improved_metrics)}"
    logger.info(conclusion)
    
    # Overall conclusion
    if len(improved_metrics) > len(METRICS) / 2:
        final_conclusion = "\nCONCLUSION: The enhanced SVM model consistently outperforms the baseline Naive Bayes model."
    else:
        final_conclusion = "\nCONCLUSION: The enhanced SVM model shows mixed results compared to the baseline Naive Bayes model."
    
    logger.info(final_conclusion)
    
    # Save conclusions to file
    with open(os.path.join(ANALYSIS_DIR, "conclusions.txt"), "w") as f:
        f.write(conclusion + "\n")
        f.write(final_conclusion + "\n")
    
    logger.info(f"\nComparison analysis completed. Results saved to {ANALYSIS_DIR}")

if __name__ == "__main__":
    main() 