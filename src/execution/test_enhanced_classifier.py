import pandas as pd
import numpy as np
import os
import time
import logging
from enhanced_classifier import EnhancedBugReportClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_sample_data(dataset_path, sample_size=100, random_state=42):
    """
    Load a sample from a dataset for testing.
    
    Args:
        dataset_path: Path to the dataset CSV file.
        sample_size: Number of samples to load.
        random_state: Random state for reproducibility.
        
    Returns:
        Sampled DataFrame.
    """
    try:
        # Read the entire dataset
        df = pd.read_csv(dataset_path)
        
        # Ensure required columns exist
        if 'Title' in df.columns and 'Body' in df.columns:
            # Merge Title and Body
            df['text'] = df.apply(
                lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
                axis=1
            )
        elif 'text' not in df.columns:
            raise ValueError(f"Dataset does not have required columns (Title and Body, or text).")
        
        # Ensure 'sentiment' column exists
        if 'class' in df.columns and 'sentiment' not in df.columns:
            df['sentiment'] = df['class']
        elif 'sentiment' not in df.columns:
            raise ValueError(f"Dataset does not have 'sentiment' or 'class' column.")
        
        # Sample the data
        if sample_size < len(df):
            df = df.sample(sample_size, random_state=random_state)
        
        return df
    
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_path}: {e}")
        return None

def run_test():
    """Run a test of the enhanced classifier on a small sample."""
    
    # Test configuration
    config = {
        'use_stemming': True,
        'use_lemmatization': False,
        'use_embeddings': False,  # Set to False to speed up the test
        'classifier_type': 'svm',
        'tfidf_ngram_range': (1, 2),
        'tfidf_max_features': 1000,
        'verbose': True
    }
    
    # Path to test dataset
    dataset_path = 'data/pytorch.csv'
    
    logger.info(f"Testing enhanced classifier on a sample from {dataset_path}")
    
    # Load sample data
    df = load_sample_data(dataset_path, sample_size=100)
    
    if df is None:
        logger.error("Failed to load sample data. Exiting test.")
        return
    
    logger.info(f"Loaded sample data with shape: {df.shape}")
    
    # Initialize classifier
    classifier = EnhancedBugReportClassifier(**config)
    
    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    # Preprocess data
    train_df = classifier.preprocess_dataset(train_df)
    test_df = classifier.preprocess_dataset(test_df)
    
    # Train classifier
    start_time = time.time()
    classifier.train(
        train_df['text'].values, 
        train_df['sentiment'].values,
        grid_search=True
    )
    train_time = time.time() - start_time
    
    # Evaluate classifier
    start_time = time.time()
    metrics = classifier.evaluate(
        test_df['text'].values,
        test_df['sentiment'].values
    )
    eval_time = time.time() - start_time
    
    # Print results
    logger.info("\n=== Test Results ===")
    logger.info(f"Training time:       {train_time:.2f} seconds")
    logger.info(f"Evaluation time:     {eval_time:.2f} seconds")
    logger.info(f"Accuracy:            {metrics['accuracy']:.4f}")
    logger.info(f"Precision:           {metrics['precision']:.4f}")
    logger.info(f"Recall:              {metrics['recall']:.4f}")
    logger.info(f"F1 Score:            {metrics['f1_score']:.4f}")
    logger.info(f"AUC:                 {metrics['auc']:.4f}")

if __name__ == "__main__":
    run_test() 