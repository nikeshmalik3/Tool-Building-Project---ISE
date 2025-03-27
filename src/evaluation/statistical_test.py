"""
Statistical testing script to evaluate the significance of improvements 
in the Enhanced Bug Report Classifier over the baseline model.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import glob
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Statistical Testing for Bug Report Classification')
    
    parser.add_argument('--baseline-dir', type=str, default='baseline_results',
                        help='Directory containing baseline results (default: baseline_results)')
    
    parser.add_argument('--enhanced-dir', type=str, default='results',
                        help='Directory containing enhanced results (default: results)')
    
    parser.add_argument('--output', type=str, default='statistical_analysis',
                        help='Output directory for statistical analysis (default: statistical_analysis)')
    
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level (default: 0.05)')
    
    parser.add_argument('--datasets', nargs='+', 
                        default=['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe'],
                        help='Datasets to analyze (default: all)')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations of statistical results')
    
    return parser.parse_args()

def get_absolute_path(path):
    """Convert relative path to absolute path if needed."""
    if os.path.isabs(path):
        return path
    
    # Try to find the project root
    current_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Check if path exists relative to current directory
    if os.path.exists(os.path.join(current_dir, path)):
        return os.path.join(current_dir, path)
    
    # Check if path exists relative to project root
    if os.path.exists(os.path.join(project_root, path)):
        return os.path.join(project_root, path)
    
    # Just return the absolute version of the path
    return os.path.abspath(path)

def load_baseline_results(baseline_dir, datasets):
    """Load baseline results from CSV files."""
    # Convert to absolute path if needed
    baseline_dir = get_absolute_path(baseline_dir)
    logger.info(f"Looking for baseline results in: {baseline_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(baseline_dir, exist_ok=True)
    
    results = {}
    
    for dataset in datasets:
        # Try to find baseline results with format: dataset_NB.csv
        baseline_path = os.path.join(baseline_dir, f"{dataset}_NB.csv")
        
        if not os.path.exists(baseline_path):
            logger.warning(f"Could not find baseline results for {dataset} at {baseline_path}")
            continue
        
        try:
            df = pd.read_csv(baseline_path)
            results[dataset] = df
            logger.info(f"Loaded baseline results for {dataset}: {len(df)} rows")
        except Exception as e:
            logger.error(f"Error loading baseline results for {dataset}: {e}")
    
    return results

def load_enhanced_results(enhanced_dir, datasets):
    """Load enhanced results from CSV files."""
    # Convert to absolute path if needed
    enhanced_dir = get_absolute_path(enhanced_dir)
    logger.info(f"Looking for enhanced results in: {enhanced_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(enhanced_dir, exist_ok=True)
    
    results = {}
    
    for dataset in datasets:
        # Try to find enhanced results with patterns like: dataset_SVM.csv or dataset_RF.csv
        enhanced_paths = glob.glob(os.path.join(enhanced_dir, f"{dataset}_*.csv"))
        
        if not enhanced_paths:
            logger.warning(f"Could not find enhanced results for {dataset} in {enhanced_dir}")
            continue
        
        dataset_results = []
        
        for path in enhanced_paths:
            try:
                df = pd.read_csv(path)
                classifier_type = os.path.basename(path).split('_')[1].split('.')[0]
                df['classifier'] = classifier_type
                dataset_results.append(df)
                logger.info(f"Loaded enhanced results for {dataset} with {classifier_type}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Error loading enhanced results from {path}: {e}")
        
        if dataset_results:
            results[dataset] = pd.concat(dataset_results, ignore_index=True)
    
    return results

def perform_statistical_tests(baseline_results, enhanced_results, alpha):
    """Perform paired t-tests to compare baseline and enhanced results."""
    if not baseline_results or not enhanced_results:
        logger.error("Could not load baseline or enhanced results. Exiting.")
        return pd.DataFrame()
        
    test_results = []
    
    for dataset in baseline_results.keys():
        if dataset not in enhanced_results:
            logger.warning(f"Skipping statistical tests for {dataset}: enhanced results not available")
            continue
        
        baseline_df = baseline_results[dataset]
        enhanced_df = enhanced_results[dataset]
        
        # Get unique classifier types in enhanced results
        classifier_types = enhanced_df['classifier'].unique() if 'classifier' in enhanced_df.columns else ['unknown']
        
        for classifier_type in classifier_types:
            # Filter enhanced results for specific classifier type
            if 'classifier' in enhanced_df.columns:
                classifier_df = enhanced_df[enhanced_df['classifier'] == classifier_type]
            else:
                classifier_df = enhanced_df
            
            # Common columns between baseline and enhanced results
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
            
            for metric in metrics:
                # Skip if the metric is not available in both datasets
                if metric not in baseline_df.columns or metric not in classifier_df.columns:
                    continue
                
                # Extract values
                baseline_values = baseline_df[metric].values
                enhanced_values = classifier_df[metric].values
                
                # If there are multiple runs for each dataset, we need to make sure we compare the same number
                min_length = min(len(baseline_values), len(enhanced_values))
                if min_length == 0:
                    continue
                
                baseline_values = baseline_values[:min_length]
                enhanced_values = enhanced_values[:min_length]
                
                # Calculate means
                baseline_mean = np.mean(baseline_values)
                enhanced_mean = np.mean(enhanced_values)
                
                # Calculate improvement
                improvement = enhanced_mean - baseline_mean
                improvement_percent = (improvement / baseline_mean) * 100 if baseline_mean > 0 else 0
                
                # Perform paired t-test if we have more than one sample
                if min_length > 1:
                    t_stat, p_value = stats.ttest_rel(enhanced_values, baseline_values)
                    significant = p_value < alpha
                else:
                    t_stat, p_value = 0, 1
                    significant = False
                
                # Store results
                test_results.append({
                    'Dataset': dataset,
                    'Classifier': classifier_type,
                    'Metric': metric,
                    'Baseline Mean': baseline_mean,
                    'Enhanced Mean': enhanced_mean,
                    'Improvement': improvement,
                    'Improvement %': improvement_percent,
                    't-statistic': t_stat,
                    'p-value': p_value,
                    'Significant': significant
                })
    
    return pd.DataFrame(test_results)

def create_visualizations(test_results, output_dir):
    """Create visualizations of statistical test results."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Plot improvement percentages by metric and classifier
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Metric', y='Improvement %', hue='Classifier', data=test_results)
    
    # Add significance indicators
    bars = ax.patches
    for bar, significant in zip(bars, test_results['Significant']):
        if significant:
            ax.text(bar.get_x() + bar.get_width() / 2, 
                   bar.get_height() + 0.5, 
                   '*', 
                   ha='center', 
                   fontsize=20)
    
    plt.title('Percentage Improvement of Enhanced Classifier over Baseline')
    plt.xlabel('Metric')
    plt.ylabel('Improvement (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_percentage.png'))
    plt.close()
    
    # Plot p-values by metric and classifier
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Metric', y='p-value', hue='Classifier', data=test_results)
    
    # Add horizontal line at alpha = 0.05
    plt.axhline(y=0.05, color='r', linestyle='--', linewidth=1)
    
    plt.title('P-values for Statistical Significance Tests')
    plt.xlabel('Metric')
    plt.ylabel('p-value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'p_values.png'))
    plt.close()
    
    # Plot improvement by dataset and classifier for each metric
    metrics = test_results['Metric'].unique()
    
    for metric in metrics:
        metric_results = test_results[test_results['Metric'] == metric]
        
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='Dataset', y='Improvement %', hue='Classifier', data=metric_results)
        
        # Add significance indicators
        bars = ax.patches
        significants = metric_results['Significant'].values
        if len(bars) == len(significants):
            for bar, significant in zip(bars, significants):
                if significant:
                    ax.text(bar.get_x() + bar.get_width() / 2, 
                       bar.get_height() + 0.5, 
                       '*', 
                       ha='center', 
                       fontsize=20)
        
        plt.title(f'Percentage Improvement for {metric} by Dataset')
        plt.xlabel('Dataset')
        plt.ylabel('Improvement (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_improvement_by_dataset.png'))
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def main():
    """Main function to run the statistical tests."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Statistical analysis started.")
    print(f"Results will be saved to {args.output}")
    
    # Load results
    baseline_results = load_baseline_results(args.baseline_dir, args.datasets)
    enhanced_results = load_enhanced_results(args.enhanced_dir, args.datasets)
    
    if not baseline_results or not enhanced_results:
        print("Error: Could not load baseline or enhanced results. Exiting.")
        sys.exit(1)
    
    # Perform statistical tests
    test_results = perform_statistical_tests(baseline_results, enhanced_results, args.alpha)
    
    # Create summary
    summary = test_results.groupby(['Classifier', 'Metric']).agg({
        'Improvement %': 'mean',
        'Significant': lambda x: sum(x) / len(x)
    }).reset_index()
    
    summary = summary.rename(columns={'Significant': 'Significant %'})
    summary['Significant %'] = summary['Significant %'] * 100
    
    # Save results
    test_results.to_csv(os.path.join(args.output, 'statistical_tests.csv'), index=False)
    summary.to_csv(os.path.join(args.output, 'statistical_summary.csv'), index=False)
    
    # Print tabular results
    print("\nStatistical Test Results Summary:")
    print(tabulate(summary, headers='keys', tablefmt='grid', floatfmt='.2f'))
    
    # Print detailed results
    print("\nDetailed Statistical Test Results:")
    detailed = test_results[['Dataset', 'Classifier', 'Metric', 'Improvement %', 'p-value', 'Significant']]
    detailed = detailed.sort_values(by=['Dataset', 'Classifier', 'Metric'])
    print(tabulate(detailed, headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    # Create visualizations if requested
    if args.visualize:
        create_visualizations(test_results, args.output)
    
    print(f"\nStatistical analysis completed. Results saved to {args.output}")

if __name__ == "__main__":
    main() 