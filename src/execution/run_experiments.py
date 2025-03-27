import os
import time
import argparse
import pandas as pd
import numpy as np
import logging
import sys

# Add parent directory to path so we can import from core
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))
from enhanced_classifier import EnhancedBugReportClassifier, run_experiments

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run bug report classification experiments')
    
    parser.add_argument('--datasets', nargs='+', 
                        default=['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe'],
                        help='List of datasets to process')
    
    parser.add_argument('--repeat', type=int, default=10,
                        help='Number of times to repeat each experiment')
    
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    
    parser.add_argument('--stemming', action='store_true', default=True,
                        help='Use stemming in preprocessing')
    
    parser.add_argument('--lemmatization', action='store_true', default=False,
                        help='Use lemmatization in preprocessing')
    
    parser.add_argument('--embeddings', action='store_true', default=True,
                        help='Use word embeddings for feature extraction')
    
    parser.add_argument('--embedding-type', type=str, default='glove-wiki-gigaword-100',
                        help='Type of word embeddings to use')
    
    parser.add_argument('--classifier', type=str, default='svm', choices=['svm', 'rf'],
                        help='Type of classifier to use (svm or rf)')
    
    parser.add_argument('--ngram-min', type=int, default=1,
                        help='Minimum n-gram size for TF-IDF')
    
    parser.add_argument('--ngram-max', type=int, default=2,
                        help='Maximum n-gram size for TF-IDF')
    
    parser.add_argument('--max-features', type=int, default=2000,
                        help='Maximum number of features for TF-IDF')
    
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print verbose output')
    
    parser.add_argument('--compare-baseline', action='store_true', default=True,
                        help='Compare results with baseline')
    
    return parser.parse_args()

def load_baseline_results(dataset_name):
    """
    Load baseline results from CSV file.
    
    Args:
        dataset_name: Name of the dataset.
        
    Returns:
        Dictionary of baseline metrics or None if not found.
    """
    # Try multiple possible locations for baseline results
    possible_paths = [
        f"baseline_results/{dataset_name}_NB.csv",
        f"../baseline_results/{dataset_name}_NB.csv",
        f"../../baseline_results/{dataset_name}_NB.csv"
    ]
    
    for baseline_file in possible_paths:
        try:
            if os.path.exists(baseline_file):
                df = pd.read_csv(baseline_file)
                # Get the latest row in case there are multiple
                latest = df.iloc[-1]
                
                baseline = {
                    'accuracy': latest['Accuracy'],
                    'precision': latest['Precision'],
                    'recall': latest['Recall'],
                    'f1_score': latest['F1'],
                    'auc': latest['AUC']
                }
                logger.info(f"Loaded baseline results from {baseline_file}")
                return baseline
        except Exception as e:
            pass
    
    logger.warning(f"Could not load baseline results for {dataset_name}")
    return None

def compare_with_baseline(enhanced_results, dataset_name):
    """
    Compare enhanced results with baseline.
    
    Args:
        enhanced_results: Dictionary of enhanced metrics.
        dataset_name: Name of the dataset.
        
    Returns:
        Dictionary of improvement percentages.
    """
    baseline = load_baseline_results(dataset_name)
    
    if baseline is None:
        logger.warning(f"Skipping baseline comparison for {dataset_name}")
        return None
    
    # Extract enhanced metrics
    enhanced = enhanced_results[dataset_name]['avg_metrics']
    
    # Calculate improvements
    improvements = {}
    for metric in enhanced:
        if metric in baseline:
            baseline_value = baseline[metric]
            enhanced_value = enhanced[metric]
            change = enhanced_value - baseline_value
            percent_change = (change / baseline_value) * 100 if baseline_value != 0 else float('inf')
            
            improvements[metric] = {
                'baseline': baseline_value,
                'enhanced': enhanced_value,
                'change': change,
                'percent_change': percent_change
            }
    
    # Print comparison
    logger.info(f"\n=== Comparison with Baseline for {dataset_name} ===")
    for metric, data in improvements.items():
        change_str = f"{data['change']:.4f}"
        if data['change'] > 0:
            change_str = '+' + change_str
        
        logger.info(f"{metric.capitalize()}: {data['baseline']:.4f} → {data['enhanced']:.4f} ({change_str}, {data['percent_change']:.2f}%)")
    
    return improvements

def create_dataset_paths(datasets, base_dir='data'):
    """
    Create dictionary mapping dataset names to file paths.
    
    Args:
        datasets: List of dataset names.
        base_dir: Base directory containing the datasets.
        
    Returns:
        Dictionary mapping dataset names to file paths.
    """
    # Try multiple possible locations for data files
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    datasets_paths = {}
    for name in datasets:
        # Check different possible locations
        possible_paths = [
            os.path.join(base_dir, f"{name}.csv"),
            os.path.join(root_dir, base_dir, f"{name}.csv"),
            os.path.join(root_dir, "data", f"{name}.csv")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                datasets_paths[name] = path
                logger.info(f"Found dataset {name} at {path}")
                break
        
        if name not in datasets_paths:
            logger.warning(f"Could not find dataset {name} in any of the expected locations")
    
    return datasets_paths

def main():
    """Main function to run experiments."""
    args = parse_arguments()
    
    # Create absolute path for output directory if it's a relative path
    if not os.path.isabs(args.output_dir):
        # First try relative to this script
        output_dir = os.path.abspath(args.output_dir)
        # If that doesn't exist and doesn't start with the script's directory,
        # try relative to the project root
        if not os.path.exists(os.path.dirname(output_dir)) and not output_dir.startswith(os.path.dirname(os.path.abspath(__file__))):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(project_root, args.output_dir)
    else:
        output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Create dataset paths
    dataset_paths = create_dataset_paths(args.datasets)
    
    if not dataset_paths:
        logger.error("No datasets found. Exiting.")
        return
    
    # Log configuration
    logger.info("=== Experiment Configuration ===")
    logger.info(f"Datasets: {', '.join(dataset_paths.keys())}")
    logger.info(f"Repetitions: {args.repeat}")
    logger.info(f"Preprocessing: {'stemming' if args.stemming else ''} {'lemmatization' if args.lemmatization else ''}")
    logger.info(f"Feature extraction: TF-IDF + {'Word Embeddings' if args.embeddings else ''}")
    logger.info(f"Classifier: {args.classifier.upper()}")
    logger.info(f"Output directory: {output_dir}")
    
    # Run experiments
    start_time = time.time()
    results = run_experiments(
        dataset_paths,
        output_dir=output_dir,
        repeat=args.repeat,
        use_stemming=args.stemming,
        use_lemmatization=args.lemmatization,
        use_embeddings=args.embeddings,
        embedding_type=args.embedding_type,
        classifier_type=args.classifier,
        tfidf_ngram_range=(args.ngram_min, args.ngram_max),
        tfidf_max_features=args.max_features,
        verbose=args.verbose
    )
    total_time = time.time() - start_time
    
    # Compare with baseline if requested
    if args.compare_baseline:
        all_improvements = {}
        for dataset_name in results:
            improvements = compare_with_baseline(results, dataset_name)
            if improvements:
                all_improvements[dataset_name] = improvements
    
    # Log overall execution time
    logger.info(f"\nTotal execution time: {total_time:.2f} seconds")
    logger.info(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 