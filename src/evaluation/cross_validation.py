"""
Cross-validation script for the Enhanced Bug Report Classifier to evaluate consistency
of performance across different data partitions.
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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tabulate import tabulate

# Add the correct paths to import the enhanced classifier
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'core'))

# Import enhanced classifier
try:
    # Try direct import first
    from enhanced_classifier import EnhancedBugReportClassifier
    print("Enhanced classifier imported successfully from enhanced_classifier")
except ImportError:
    try:
        # Try alternative import path
        sys.path.append(os.path.join(project_root, 'src'))
        from core.enhanced_classifier import EnhancedBugReportClassifier
        print("Enhanced classifier imported successfully from core.enhanced_classifier")
    except ImportError as e:
        print(f"Error: Could not import enhanced classifier: {e}")
        print("Make sure enhanced_classifier.py exists in the src/core directory.")
        sys.exit(1)

# Define datasets
DATASETS = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cross Validation for Bug Report Classification')
    
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS,
                        help='Datasets to analyze (default: all)')
    
    parser.add_argument('--k-folds', type=int, default=5,
                        help='Number of folds for cross-validation (default: 5)')
    
    parser.add_argument('--output', type=str, default='cross_validation_results',
                        help='Output directory for cross-validation results (default: cross_validation_results)')
    
    parser.add_argument('--classifiers', nargs='+', choices=['svm', 'rf'], default=['svm'],
                        help='Classifiers to evaluate (default: svm)')
    
    parser.add_argument('--embeddings', action='store_true',
                        help='Use word embeddings in enhanced classifier')
    
    parser.add_argument('--stemming', action='store_true',
                        help='Use stemming in enhanced classifier')
    
    parser.add_argument('--lemmatization', action='store_true',
                        help='Use lemmatization in enhanced classifier')
    
    parser.add_argument('--max-features', type=int, default=1000,
                        help='Maximum number of features for TF-IDF (default: 1000)')
    
    parser.add_argument('--stratified', action='store_true',
                        help='Use stratified cross-validation (default: False)')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations of cross-validation results')
    
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    
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

def run_cross_validation(df, dataset_name, args, classifier_type):
    """Run k-fold cross-validation on a dataset."""
    print(f"\nRunning {args.k_folds}-fold cross-validation on {dataset_name} dataset with {classifier_type}...")
    
    # Prepare data
    X = df[['Title', 'Body']]
    y = df['class']
    
    # Initialize cross-validation
    if args.stratified:
        kfold = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_state)
        fold_type = "Stratified"
    else:
        kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_state)
        fold_type = "Standard"
    
    # Metrics for each fold
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"  Fold {fold+1}/{args.k_folds}")
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        try:
            # Initialize classifier with specified parameters
            classifier = EnhancedBugReportClassifier(
                use_stemming=args.stemming,
                use_lemmatization=args.lemmatization,
                use_embeddings=args.embeddings,
                tfidf_max_features=args.max_features,
                classifier_type=classifier_type
            )
            
            # Time the training and prediction
            start_time = time.time()
            classifier.fit(X_train, y_train)
            predictions = classifier.predict(X_test)
            end_time = time.time()
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            
            # Try to get probability predictions for AUC
            try:
                proba = classifier.predict_proba(X_test)
                auc = roc_auc_score(y_test, proba[:, 1])
            except:
                auc = 0
            
            execution_time = end_time - start_time
            
            # Store results
            results.append({
                'Dataset': dataset_name,
                'Classifier': classifier_type.upper(),
                'Fold': fold+1,
                'Fold Type': fold_type,
                'Train Size': len(X_train),
                'Test Size': len(X_test),
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'AUC': auc,
                'Time (s)': execution_time
            })
            
            print(f"    Time: {execution_time:.2f}s, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
        except Exception as e:
            print(f"  Error in fold {fold+1}: {e}")
    
    return results

def calculate_statistics(results_df):
    """Calculate statistics for cross-validation results."""
    # Group by dataset and classifier
    grouped = results_df.groupby(['Dataset', 'Classifier'])
    
    statistics = []
    
    for (dataset, classifier), group in grouped:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        
        for metric in metrics:
            values = group[metric].values
            
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                range_val = max_val - min_val
                
                statistics.append({
                    'Dataset': dataset,
                    'Classifier': classifier,
                    'Metric': metric,
                    'Mean': mean_val,
                    'Std': std_val,
                    'Min': min_val,
                    'Max': max_val,
                    'Range': range_val,
                    'Coefficient of Variation': std_val / mean_val if mean_val > 0 else 0
                })
    
    return pd.DataFrame(statistics)

def visualize_results(results_df, statistics_df, output_dir):
    """Create visualizations of cross-validation results."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Plot box plots for each metric across datasets
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    
    for metric in metrics:
        plt.figure(figsize=(14, 8))
        ax = sns.boxplot(x='Dataset', y=metric, hue='Classifier', data=results_df)
        plt.title(f'{metric} Distribution Across Datasets')
        plt.xlabel('Dataset')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_boxplot.png'))
        plt.close()
    
    # Plot coefficient of variation for each metric
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Metric', y='Coefficient of Variation', hue='Classifier', 
                    data=statistics_df)
    plt.title('Coefficient of Variation by Metric')
    plt.xlabel('Metric')
    plt.ylabel('Coefficient of Variation (lower is more consistent)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coefficient_of_variation.png'))
    plt.close()
    
    # Plot fold-to-fold variation for each dataset and classifier
    for (dataset, classifier), group in results_df.groupby(['Dataset', 'Classifier']):
        plt.figure(figsize=(14, 8))
        
        # Melt the dataframe to get it in the right format for line plot
        metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        melted = pd.melt(group, id_vars=['Fold'], value_vars=metrics_cols, 
                         var_name='Metric', value_name='Value')
        
        # Plot line for each metric across folds
        ax = sns.lineplot(x='Fold', y='Value', hue='Metric', data=melted, markers=True, dashes=False)
        
        plt.title(f'Metric Variation Across Folds - {dataset} - {classifier}')
        plt.xlabel('Fold')
        plt.ylabel('Value')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset}_{classifier}_fold_variation.png'))
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def main():
    """Main function to run the cross-validation."""
    args = parse_arguments()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Cross-validation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results will be saved to {output_dir}")
    
    all_results = []
    
    for dataset_name in args.datasets:
        df = load_data(dataset_name)
        if df is None:
            continue
        
        print(f"\nProcessing {dataset_name} dataset (size: {len(df)})")
        
        # Check class imbalance
        class_counts = df['class'].value_counts()
        print(f"Class distribution: {class_counts.to_dict()}")
        
        for classifier_type in args.classifiers:
            results = run_cross_validation(df, dataset_name, args, classifier_type)
            all_results.extend(results)
    
    # Convert results to DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save detailed results
        results_path = os.path.join(output_dir, 'cross_validation_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\nDetailed results saved to {results_path}")
        
        # Calculate and save statistics
        statistics_df = calculate_statistics(results_df)
        stats_path = os.path.join(output_dir, 'cross_validation_statistics.csv')
        statistics_df.to_csv(stats_path, index=False)
        print(f"Statistics saved to {stats_path}")
        
        # Print statistical summary
        print("\nCross-Validation Statistical Summary:")
        summary = statistics_df.pivot_table(
            index=['Dataset', 'Classifier'], 
            columns='Metric', 
            values=['Mean', 'Std', 'Coefficient of Variation']
        )
        print(tabulate(summary, headers='keys', tablefmt='grid', floatfmt='.4f'))
        
        # Create visualizations if requested
        if args.visualize:
            visualize_results(results_df, statistics_df, output_dir)
    
    print(f"\nCross-validation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()