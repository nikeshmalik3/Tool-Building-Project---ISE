"""
Benchmark script for comparing the performance and execution time of baseline and enhanced
bug report classifiers across different datasets and configurations.

The baseline code for comparison can be downloaded from:
https://github.com/ideas-labo/ISE-solution/tree/main/lab1
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add the correct paths to import the classifiers
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'core'))

# Import both baseline and enhanced classifier
try:
    # Try multiple possible paths for baseline classifier
    # Note: The baseline classifier can be obtained from https://github.com/ideas-labo/ISE-solution/tree/main/lab1
    # Clone the repository: git clone https://github.com/ideas-labo/ISE-solution.git lab1-baseline
    baseline_module_found = False
    possible_paths = [
        os.path.join(project_root, "Lab 1"),
        os.path.join(project_root, "Lab_1"),
        os.path.join(os.path.dirname(project_root), "Lab 1"),
        os.path.join(os.path.dirname(project_root), "Lab_1"),
        os.path.join(project_root, "lab1-baseline", "lab1")  # Path if cloned from GitHub
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            sys.path.append(path)
            try:
                from br_classification import BugReportClassifier as BaselineClassifier
                print(f"Baseline classifier imported successfully from {path}")
                baseline_module_found = True
                break
            except ImportError:
                pass
    
    if not baseline_module_found:
        print("Warning: Could not import baseline classifier. Some functionality will be limited.")
        BaselineClassifier = None
        
except Exception as e:
    print(f"Warning: Error importing baseline classifier: {e}")
    BaselineClassifier = None

try:
    # Try multiple possible paths for enhanced classifier
    from enhanced_classifier import EnhancedBugReportClassifier
    print("Enhanced classifier imported successfully from enhanced_classifier")
except ImportError:
    try:
        # Try alternative import paths
        sys.path.append(os.path.join(project_root, 'src'))
        from core.enhanced_classifier import EnhancedBugReportClassifier
        print("Enhanced classifier imported successfully from core.enhanced_classifier")
    except ImportError as e:
        print(f"Error: Could not import enhanced classifier: {e}")
        print("Make sure enhanced_classifier.py is in the src/core directory.")
        sys.exit(1)

# Define datasets
DATASETS = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Bug Report Classification Benchmark')
    
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS,
                        help='Datasets to benchmark (default: all)')
    
    parser.add_argument('--repeat', type=int, default=3,
                        help='Number of times to repeat each experiment (default: 3)')
    
    parser.add_argument('--output', type=str, default='benchmark_results',
                        help='Output directory for benchmark results (default: benchmark_results)')
    
    parser.add_argument('--classifiers', nargs='+', choices=['nb', 'svm', 'rf'], default=['nb', 'svm'],
                        help='Classifiers to benchmark (default: nb, svm)')
    
    parser.add_argument('--embeddings', action='store_true',
                        help='Use word embeddings in enhanced classifier')
    
    parser.add_argument('--stemming', action='store_true',
                        help='Use stemming in enhanced classifier')
    
    parser.add_argument('--lemmatization', action='store_true',
                        help='Use lemmatization in enhanced classifier')
    
    parser.add_argument('--max-features', type=int, default=1000,
                        help='Maximum number of features for TF-IDF (default: 1000)')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations of benchmark results')
    
    parser.add_argument('--baseline-dir', type=str, default='baseline_results',
                        help='Directory containing baseline results')
                        
    return parser.parse_args()

def load_data(dataset_name):
    """Load a dataset from the data directory."""
    try:
        # Try multiple possible locations for the dataset files
        possible_paths = [
            # Try absolute path with the project root
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                        'data', f'{dataset_name}.csv'),
            # Try relative to the current directory
            os.path.join('data', f'{dataset_name}.csv'),
            # Try going up one level
            os.path.join('..', 'data', f'{dataset_name}.csv'),
            # Try going up two levels
            os.path.join('..', '..', 'data', f'{dataset_name}.csv'),
            # Try a direct path to the Final Assignment directory
            os.path.join('D:', 'Uob Assignment - nIkesh', 'Intellignet software engineering', 
                        'Final Assignment', 'data', f'{dataset_name}.csv')
        ]
        
        # Try each path
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found dataset at {path}")
                df = pd.read_csv(path)
                return df
        
        print(f"Error: Could not find dataset {dataset_name}.csv")
        print(f"Tried paths: {possible_paths}")
        return None
    except Exception as e:
        print(f"Error loading {dataset_name} dataset: {e}")
        return None

def run_baseline_benchmark(dataset_name, df, args):
    """Run benchmark for the baseline classifier."""
    if BaselineClassifier is None:
        print("Baseline classifier not available. Skipping baseline benchmark.")
        return None
    
    print(f"\nRunning baseline benchmark for {dataset_name} dataset...")
    
    results = []
    total_time = 0
    
    for i in range(args.repeat):
        print(f"  Repeat {i+1}/{args.repeat}")
        
        try:
            # Create baseline classifier (Naive Bayes)
            classifier = BaselineClassifier(project=dataset_name)
            
            # Prepare data (simplified from original baseline)
            X = df[['Title', 'Body']].apply(lambda x: x['Title'] + ' ' + x['Body'], axis=1)
            y = df['class']
            
            # Time the training and prediction
            start_time = time.time()
            classifier.fit(X, y)
            predictions = classifier.predict(X)
            end_time = time.time()
            
            # Calculate metrics
            accuracy = accuracy_score(y, predictions)
            precision = precision_score(y, predictions)
            recall = recall_score(y, predictions)
            f1 = f1_score(y, predictions)
            
            # Try to get probability predictions for AUC
            try:
                proba = classifier.predict_proba(X)
                auc = roc_auc_score(y, proba[:, 1])
            except:
                auc = 0
            
            execution_time = end_time - start_time
            total_time += execution_time
            
            results.append({
                'Dataset': dataset_name,
                'Model': 'Baseline (NB)',
                'Repeat': i+1,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'AUC': auc,
                'Time (s)': execution_time
            })
            
            print(f"    Time: {execution_time:.2f}s, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
        except Exception as e:
            print(f"  Error running baseline benchmark: {e}")
    
    print(f"Baseline benchmark completed. Average time: {total_time/args.repeat:.2f}s")
    return results

def run_enhanced_benchmark(dataset_name, df, args, classifier_type):
    """Run benchmark for the enhanced classifier."""
    print(f"\nRunning enhanced benchmark for {dataset_name} dataset with {classifier_type}...")
    
    results = []
    total_time = 0
    
    for i in range(args.repeat):
        print(f"  Repeat {i+1}/{args.repeat}")
        
        try:
            # Create enhanced classifier with specified parameters
            classifier = EnhancedBugReportClassifier(
                use_stemming=args.stemming,
                use_lemmatization=args.lemmatization,
                use_embeddings=args.embeddings,
                tfidf_max_features=args.max_features,
                classifier_type=classifier_type
            )
            
            # Prepare data
            X = df[['Title', 'Body']]
            y = df['class']
            
            # Time the training and prediction
            start_time = time.time()
            classifier.fit(X, y)
            predictions = classifier.predict(X)
            end_time = time.time()
            
            # Calculate metrics
            accuracy = accuracy_score(y, predictions)
            precision = precision_score(y, predictions)
            recall = recall_score(y, predictions)
            f1 = f1_score(y, predictions)
            
            # Try to get probability predictions for AUC
            try:
                proba = classifier.predict_proba(X)
                auc = roc_auc_score(y, proba[:, 1])
            except:
                auc = 0
            
            execution_time = end_time - start_time
            total_time += execution_time
            
            results.append({
                'Dataset': dataset_name,
                'Model': f'Enhanced ({classifier_type.upper()})',
                'Repeat': i+1,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'AUC': auc,
                'Time (s)': execution_time
            })
            
            print(f"    Time: {execution_time:.2f}s, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
        except Exception as e:
            print(f"  Error running enhanced benchmark: {e}")
    
    print(f"Enhanced benchmark completed. Average time: {total_time/args.repeat:.2f}s")
    return results

def visualize_results(df_results, output_dir):
    """Create visualizations of benchmark results."""
    print("\nCreating visualizations...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Create mean results for easier visualization
    mean_results = df_results.groupby(['Dataset', 'Model']).mean().reset_index()
    
    # Plot metrics comparison across models
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Dataset', y=metric, hue='Model', data=mean_results)
        plt.title(f'{metric} Comparison Across Models')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric.lower()}_comparison.png'))
        plt.close()
    
    # Plot execution time comparison
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Dataset', y='Time (s)', hue='Model', data=mean_results)
    plt.title('Execution Time Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'))
    plt.close()
    
    # Create radar chart for comparing models
    models = mean_results['Model'].unique()
    for dataset in mean_results['Dataset'].unique():
        dataset_results = mean_results[mean_results['Dataset'] == dataset]
        
        # Set up radar chart
        categories = metrics
        n = len(categories)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Plot each model
        for model in models:
            model_results = dataset_results[dataset_results['Model'] == model]
            if model_results.empty:
                continue
                
            values = model_results[metrics].values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Set category labels
        plt.xticks(angles[:-1], categories)
        
        # Add legend
        plt.legend(loc='upper right')
        
        plt.title(f'Performance Comparison for {dataset} Dataset')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset}_radar.png'))
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def main():
    """Main function to run the benchmark."""
    args = parse_arguments()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Benchmark started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results will be saved to {output_dir}")
    
    all_results = []
    
    for dataset_name in args.datasets:
        df = load_data(dataset_name)
        if df is None:
            continue
        
        print(f"\nBenchmarking {dataset_name} dataset (size: {len(df)})")
        
        # Run baseline if selected
        if 'nb' in args.classifiers:
            baseline_results = run_baseline_benchmark(dataset_name, df, args)
            if baseline_results:
                all_results.extend(baseline_results)
        
        # Run enhanced classifier with SVM if selected
        if 'svm' in args.classifiers:
            enhanced_results_svm = run_enhanced_benchmark(dataset_name, df, args, 'svm')
            if enhanced_results_svm:
                all_results.extend(enhanced_results_svm)
        
        # Run enhanced classifier with Random Forest if selected
        if 'rf' in args.classifiers:
            enhanced_results_rf = run_enhanced_benchmark(dataset_name, df, args, 'rf')
            if enhanced_results_rf:
                all_results.extend(enhanced_results_rf)
    
    # Save results to CSV
    if all_results:
        df_results = pd.DataFrame(all_results)
        results_path = os.path.join(output_dir, 'benchmark_results.csv')
        df_results.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")
        
        # Create summary
        summary = df_results.groupby(['Dataset', 'Model']).mean().reset_index()
        summary_path = os.path.join(output_dir, 'benchmark_summary.csv')
        summary.to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")
        
        # Create visualizations if requested
        if args.visualize:
            visualize_results(df_results, output_dir)
    
    print(f"\nBenchmark completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 