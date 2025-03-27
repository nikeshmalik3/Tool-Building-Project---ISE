#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for running the Enhanced Bug Report Classification tool.

This script serves as the central entry point for the entire system, orchestrating:
1. Running experiments with the enhanced classifier
2. Benchmarking against the baseline
3. Performing cross-validation
4. Running statistical tests
5. Generating comparative analysis and visualizations

The project builds upon and compares against the baseline implementation from Lab 1:
https://github.com/ideas-labo/ISE-solution/tree/main/lab1

Usage:
    python main.py [--help]
"""

import os
import sys
import time
import argparse
import logging
import subprocess
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add src directory to path so we can import modules
sys.path.insert(0, os.path.abspath('src'))

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced Bug Report Classification Tool')
    
    parser.add_argument('--datasets', nargs='+', 
                        default=['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe'],
                        help='Datasets to process (default: all)')
    
    parser.add_argument('--skip-experiments', action='store_true',
                        help='Skip running experiments (use existing results)')
    
    parser.add_argument('--skip-benchmark', action='store_true',
                        help='Skip running benchmark comparison')
    
    parser.add_argument('--skip-cross-validation', action='store_true',
                        help='Skip running cross-validation')
    
    parser.add_argument('--skip-statistical-tests', action='store_true',
                        help='Skip running statistical tests')
    
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Skip running comparative analysis')
    
    parser.add_argument('--stemming', action='store_true', default=True,
                        help='Use stemming in preprocessing')
    
    parser.add_argument('--lemmatization', action='store_true', default=False,
                        help='Use lemmatization in preprocessing')
    
    parser.add_argument('--embeddings', action='store_true', default=True,
                        help='Use word embeddings for feature extraction')
    
    parser.add_argument('--classifier', type=str, default='svm', choices=['svm', 'rf'],
                        help='Type of classifier to use (svm or rf)')
    
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Base directory for results')
    
    parser.add_argument('--baseline-dir', type=str, default='baseline_results',
                        help='Directory for baseline results')
    
    parser.add_argument('--k-folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    
    parser.add_argument('--repeat', type=int, default=10,
                        help='Number of times to repeat each experiment')
    
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Create visualizations')
    
    return parser.parse_args()

def get_absolute_path(path):
    """Convert relative path to absolute path based on project root."""
    if os.path.isabs(path):
        return path
    
    # Try to find the project root
    current_dir = os.path.abspath(os.path.dirname(__file__))
    
    # First check if path exists relative to current directory
    if os.path.exists(os.path.join(current_dir, path)) or not os.path.exists(os.path.dirname(os.path.join(current_dir, path))):
        return os.path.join(current_dir, path)
    
    # If not, try to use the path as is
    return os.path.abspath(path)

def run_module(module_path, args, description):
    """Import and run a module programmatically."""
    logger.info(f"\n{'='*80}\nRunning: {description}\n{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # Add the core directory to sys.path to ensure enhanced_classifier can be found
        project_root = os.path.abspath(os.path.dirname(__file__))
        sys.path.append(os.path.join(project_root, 'src', 'core'))
        
        # Add the module's directory to sys.path
        module_dir = os.path.dirname(module_path)
        if module_dir and module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        
        # Check if the module file exists
        full_module_path = os.path.join(project_root, module_path)
        if not os.path.exists(full_module_path):
            logger.warning(f"Module file not found at {full_module_path}")
            # Try alternate path formulations
            alternate_path = os.path.abspath(module_path)
            if os.path.exists(alternate_path):
                full_module_path = alternate_path
                logger.info(f"Found module at alternate path: {full_module_path}")
            else:
                logger.error(f"Could not find module file: {module_path}")
                return
        
        # Import the module
        module_name = os.path.basename(module_path).replace('.py', '')
        if module_dir:
            module_name = os.path.basename(module_dir) + '.' + module_name
        
        # Set environment variables to help modules find their dependencies
        os.environ['PROJECT_ROOT'] = project_root
        os.environ['ENHANCED_CLASSIFIER_PATH'] = os.path.join(project_root, 'src', 'core', 'enhanced_classifier.py')
        
        # Remove .py if it exists in the name
        module = __import__(module_name, fromlist=['main'])
        
        # Get the main function and run it with args
        if hasattr(module, 'main'):
            # Override sys.argv temporarily
            old_argv = sys.argv
            sys.argv = [module_path] + args
            
            # Run the main function
            module.main()
            
            # Restore sys.argv
            sys.argv = old_argv
        else:
            logger.error(f"Module {module_name} does not have a main function")
    except Exception as e:
        logger.error(f"Error running {description}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    end_time = time.time()
    logger.info(f"\nCompleted in {end_time - start_time:.2f} seconds\n")

def run_experiments(args):
    """Run experiments on all datasets."""
    logger.info("Starting experiments with enhanced classifier...")
    
    # Create command-line arguments for run_experiments.py
    cmd_args = [
        '--datasets'] + args.datasets + [
        '--output-dir', args.output_dir,
        '--repeat', str(args.repeat),
        '--classifier', args.classifier
    ]
    
    if args.stemming:
        cmd_args.append('--stemming')
    if args.lemmatization:
        cmd_args.append('--lemmatization')
    if args.embeddings:
        cmd_args.append('--embeddings')
    
    # Run the module
    run_module('src/execution/run_experiments.py', cmd_args, 
               "Running Enhanced Classifier Experiments")

def run_benchmark(args):
    """Run benchmark comparison between baseline and enhanced models."""
    logger.info("Starting benchmark comparison...")
    
    # Create benchmark results directory
    benchmark_dir = os.path.join(args.output_dir, 'benchmark')
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Create command-line arguments for benchmark.py
    cmd_args = [
        '--datasets'] + args.datasets + [
        '--output', benchmark_dir,
        '--repeat', str(args.repeat),
        '--classifiers', 'nb', args.classifier,
        '--baseline-dir', args.baseline_dir
    ]
    
    if args.stemming:
        cmd_args.append('--stemming')
    if args.lemmatization:
        cmd_args.append('--lemmatization')
    if args.embeddings:
        cmd_args.append('--embeddings')
    if args.visualize:
        cmd_args.append('--visualize')
    
    # Run the module
    run_module('src/evaluation/benchmark.py', cmd_args, 
               "Running Benchmark Comparison")

def run_cross_validation(args):
    """Run cross-validation."""
    logger.info("Starting cross-validation...")
    
    # Create cross-validation directory
    cv_dir = os.path.join(args.output_dir, 'cross_validation')
    os.makedirs(cv_dir, exist_ok=True)
    
    # Create command-line arguments for cross_validation.py
    cmd_args = [
        '--datasets'] + args.datasets + [
        '--output', cv_dir,
        '--k-folds', str(args.k_folds),
        '--classifiers', args.classifier,
        '--stratified'
    ]
    
    if args.stemming:
        cmd_args.append('--stemming')
    if args.lemmatization:
        cmd_args.append('--lemmatization')
    if args.embeddings:
        cmd_args.append('--embeddings')
    if args.visualize:
        cmd_args.append('--visualize')
    
    # Run the module
    run_module('src/evaluation/cross_validation.py', cmd_args, 
               "Running Cross-Validation")

def run_statistical_tests(args):
    """Run statistical tests."""
    logger.info("Starting statistical tests...")
    
    # Create statistical test directory
    stats_dir = os.path.join(args.output_dir, 'statistical_tests')
    os.makedirs(stats_dir, exist_ok=True)
    
    # Create command-line arguments for statistical_test.py
    cmd_args = [
        '--datasets'] + args.datasets + [
        '--baseline-dir', args.baseline_dir,
        '--enhanced-dir', args.output_dir,
        '--output', stats_dir
    ]
    
    if args.visualize:
        cmd_args.append('--visualize')
    
    # Run the module
    run_module('src/evaluation/statistical_test.py', cmd_args, 
               "Running Statistical Tests")

def run_analysis(args):
    """Run comparative analysis."""
    logger.info("Starting comparative analysis...")
    
    # Create analysis directory
    analysis_dir = os.path.join(args.output_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Set environment variables to help analysis scripts find the right directories
    os.environ['BASELINE_RESULTS_DIR'] = args.baseline_dir
    os.environ['ENHANCED_RESULTS_DIR'] = args.output_dir
    os.environ['ANALYSIS_OUTPUT_DIR'] = analysis_dir
    
    # First, run the simple comparison for a tabular output
    logger.info("Running simple comparison...")
    try:
        # Import the module
        sys.path.insert(0, 'src/analysis')
        from analysis.simple_comparison import main as simple_comparison_main
        
        # Run the main function
        simple_comparison_main()
    except Exception as e:
        logger.error(f"Error running simple comparison: {e}")
    
    # Then run the comprehensive comparison with visualizations
    logger.info("Running comprehensive comparison...")
    try:
        # Import the module
        from analysis.compare_results import main as compare_results_main
        
        # Run the main function
        compare_results_main()
    except Exception as e:
        logger.error(f"Error running comprehensive comparison: {e}")

def main():
    """Main function to run the entire pipeline."""
    args = parse_arguments()
    
    # Convert output directory to absolute path if needed
    args.output_dir = get_absolute_path(args.output_dir)
    args.baseline_dir = get_absolute_path(args.baseline_dir)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"=== Enhanced Bug Report Classification Tool ===")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create main results directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.baseline_dir, exist_ok=True)
    
    # Create subdirectories
    for subdir in ['benchmark', 'cross_validation', 'statistical_tests', 'analysis']:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)
    
    logger.info(f"Results will be saved to: {args.output_dir}")
    logger.info(f"Baseline results will be compared from: {args.baseline_dir}")
    
    # Store start time for total execution
    total_start_time = time.time()
    
    # 1. Run experiments (if not skipped)
    if not args.skip_experiments:
        run_experiments(args)
    else:
        logger.info("Skipping experiments (using existing results)")
    
    # 2. Run benchmark comparison (if not skipped)
    if not args.skip_benchmark:
        run_benchmark(args)
    else:
        logger.info("Skipping benchmark comparison")
    
    # 3. Run cross-validation (if not skipped)
    if not args.skip_cross_validation:
        run_cross_validation(args)
    else:
        logger.info("Skipping cross-validation")
    
    # 4. Run statistical tests (if not skipped)
    if not args.skip_statistical_tests:
        run_statistical_tests(args)
    else:
        logger.info("Skipping statistical tests")
    
    # 5. Run comparative analysis (if not skipped)
    if not args.skip_analysis:
        run_analysis(args)
    else:
        logger.info("Skipping comparative analysis")
    
    # Calculate total execution time
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    # Print overall summary
    logger.info(f"\n{'='*80}")
    logger.info(f"Enhanced Bug Report Classification Tool - Execution Summary")
    logger.info(f"{'='*80}")
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total execution time: {total_execution_time:.2f} seconds")
    logger.info(f"Results saved to: {os.path.abspath(args.output_dir)}")
    logger.info(f"{'='*80}")
    
    # Print usage instructions
    logger.info("\nTo view detailed results:")
    logger.info(f"  - Model performance: {os.path.join(args.output_dir, '*.csv')}")
    logger.info(f"  - Benchmark results: {os.path.join(args.output_dir, 'benchmark')}")
    logger.info(f"  - Cross-validation: {os.path.join(args.output_dir, 'cross_validation')}")
    logger.info(f"  - Statistical tests: {os.path.join(args.output_dir, 'statistical_tests')}")
    logger.info(f"  - Comparison reports: {os.path.join(args.output_dir, 'analysis')}")

if __name__ == "__main__":
    main() 