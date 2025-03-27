import pandas as pd
import numpy as np
import re
import math
import os
import time
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings('ignore')

# Text and feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_curve, auc, classification_report)

# Classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Text cleaning & stopwords
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Word embeddings
import gensim.downloader as api

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

########## 1. Define text preprocessing methods ##########

def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                              u"\U0001F600-\U0001F64F"  # emoticons
                              u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                              u"\U0001F680-\U0001F6FF"  # transport & map symbols
                              u"\U0001F1E0-\U0001F1FF"  # flags
                              u"\U00002702-\U000027B0"
                              u"\U000024C2-\U0001F251"  # enclosed characters
                              "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Stopwords
NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']  # You can customize this list as needed
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

def remove_stopwords(text):
    """Remove stopwords from the text."""
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])

def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def apply_stemming(text):
    """Apply stemming to reduce words to their root form."""
    words = word_tokenize(text)
    return " ".join([stemmer.stem(word) for word in words])

def apply_lemmatization(text):
    """Apply lemmatization to reduce words to their base form."""
    words = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(word) for word in words])

########## 2. Feature extraction functions ##########

def load_word_embeddings(embedding_type='glove-wiki-gigaword-100'):
    """
    Load pre-trained word embeddings.
    
    Args:
        embedding_type: The type of pre-trained embedding to load.
                        Default is 'glove-wiki-gigaword-100'.
    
    Returns:
        The word embedding model.
    """
    logger.info(f"Loading {embedding_type} word embeddings...")
    try:
        word_vectors = api.load(embedding_type)
        logger.info(f"Word embeddings loaded successfully. Vocabulary size: {len(word_vectors.key_to_index)}")
        return word_vectors
    except Exception as e:
        logger.error(f"Failed to load word embeddings: {e}")
        return None

def get_document_embedding(doc, word_vectors, embedding_size=100):
    """
    Calculate the average word embedding for a document.
    
    Args:
        doc: The document text (string).
        word_vectors: The pre-trained word embedding model.
        embedding_size: The size of the word embeddings.
    
    Returns:
        A numpy array representing the average embedding of all words in the document.
    """
    words = word_tokenize(doc.lower())
    embeddings = []
    
    for word in words:
        if word in word_vectors:
            embeddings.append(word_vectors[word])
    
    if len(embeddings) > 0:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(embedding_size)

def create_embedding_features(texts, word_vectors, embedding_size=100):
    """
    Create document embeddings for a collection of texts.
    
    Args:
        texts: List of text documents.
        word_vectors: The pre-trained word embedding model.
        embedding_size: The size of the word embeddings.
    
    Returns:
        A numpy array where each row is a document embedding.
    """
    doc_embeddings = []
    for doc in tqdm(texts, desc="Creating document embeddings"):
        embedding = get_document_embedding(doc, word_vectors, embedding_size)
        doc_embeddings.append(embedding)
    
    return np.array(doc_embeddings)

########## 3. Main classifier class ##########

class EnhancedBugReportClassifier:
    """
    Enhanced classifier for bug reports that improves on the baseline Naive Bayes model
    by using better preprocessing, hybrid features (TF-IDF + Word Embeddings),
    and an SVM classifier.
    """
    
    def __init__(self, 
                use_stemming=True, 
                use_lemmatization=False,
                use_embeddings=True,
                embedding_type='glove-wiki-gigaword-100',
                classifier_type='svm',
                tfidf_ngram_range=(1, 2),
                tfidf_max_features=2000,
                verbose=True):
        """
        Initialize the classifier with configuration options.
        
        Args:
            use_stemming: Whether to apply stemming (default: True).
            use_lemmatization: Whether to apply lemmatization (default: False).
            use_embeddings: Whether to use word embeddings (default: True).
            embedding_type: Type of word embeddings to use (default: 'glove-wiki-gigaword-100').
            classifier_type: Type of classifier to use, 'svm' or 'rf' (default: 'svm').
            tfidf_ngram_range: Range of n-grams for TF-IDF (default: (1, 2)).
            tfidf_max_features: Maximum number of features for TF-IDF (default: 2000).
            verbose: Whether to print progress information (default: True).
        """
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.use_embeddings = use_embeddings
        self.embedding_type = embedding_type
        self.classifier_type = classifier_type
        self.tfidf_ngram_range = tfidf_ngram_range
        self.tfidf_max_features = tfidf_max_features
        self.verbose = verbose
        
        # Initialize components
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=self.tfidf_ngram_range,
            max_features=self.tfidf_max_features
        )
        
        if self.use_embeddings:
            self.word_vectors = load_word_embeddings(self.embedding_type)
            self.embedding_size = 100  # Default for glove-wiki-gigaword-100
        else:
            self.word_vectors = None
            
        # Initialize classifier
        if self.classifier_type == 'svm':
            self.classifier = SVC(probability=True, class_weight='balanced')
            self.param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear']
            }
        elif self.classifier_type == 'rf':
            self.classifier = RandomForestClassifier(class_weight='balanced')
            self.param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    def preprocess_text(self, text):
        """
        Apply the full text preprocessing pipeline.
        
        Args:
            text: The text to preprocess.
            
        Returns:
            The preprocessed text.
        """
        # Apply the baseline preprocessing steps
        text = remove_html(text)
        text = remove_emoji(text)
        text = remove_stopwords(text)
        text = clean_str(text)
        
        # Apply enhanced preprocessing
        if self.use_stemming:
            text = apply_stemming(text)
            
        if self.use_lemmatization:
            text = apply_lemmatization(text)
            
        return text
    
    def extract_features(self, texts, train=False):
        """
        Extract features from texts.
        
        Args:
            texts: List of preprocessed text documents.
            train: Whether this is training data (to fit the vectorizer).
            
        Returns:
            Feature matrix combining TF-IDF and (optionally) embeddings.
        """
        # Extract TF-IDF features
        if train:
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts)
        
        # If not using embeddings, return just TF-IDF features
        if not self.use_embeddings or self.word_vectors is None:
            return tfidf_features
        
        # Extract embedding features
        embedding_features = create_embedding_features(texts, self.word_vectors, self.embedding_size)
        
        # Combine TF-IDF and embedding features
        # First convert sparse TF-IDF matrix to dense
        tfidf_dense = tfidf_features.toarray()
        
        # Concatenate horizontally
        combined_features = np.hstack((tfidf_dense, embedding_features))
        
        return combined_features
    
    def train(self, texts, labels, grid_search=True, cv=5):
        """
        Train the classifier on the given texts and labels.
        
        Args:
            texts: List of preprocessed text documents.
            labels: Corresponding labels for the texts.
            grid_search: Whether to perform grid search for hyperparameters.
            cv: Number of cross-validation folds.
            
        Returns:
            Self, for method chaining.
        """
        if self.verbose:
            logger.info("Extracting features for training data...")
        
        X_train = self.extract_features(texts, train=True)
        
        if grid_search:
            if self.verbose:
                logger.info("Performing grid search for hyperparameter tuning...")
            
            grid = GridSearchCV(
                self.classifier,
                self.param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1 if self.verbose else 0
            )
            
            grid.fit(X_train, labels)
            
            if self.verbose:
                logger.info(f"Best parameters: {grid.best_params_}")
                logger.info(f"Best score: {grid.best_score_:.4f}")
            
            self.classifier = grid.best_estimator_
        else:
            if self.verbose:
                logger.info("Training classifier with default parameters...")
            
            self.classifier.fit(X_train, labels)
        
        return self
    
    def predict(self, texts):
        """
        Make predictions on the given texts.
        
        Args:
            texts: List of preprocessed text documents.
            
        Returns:
            Predicted labels.
        """
        if self.verbose:
            logger.info("Extracting features for prediction...")
        
        X = self.extract_features(texts, train=False)
        
        if self.verbose:
            logger.info("Making predictions...")
        
        return self.classifier.predict(X)
    
    def predict_proba(self, texts):
        """
        Get prediction probabilities for the given texts.
        
        Args:
            texts: List of preprocessed text documents.
            
        Returns:
            Predicted probabilities.
        """
        if self.verbose:
            logger.info("Extracting features for prediction...")
        
        X = self.extract_features(texts, train=False)
        
        if self.verbose:
            logger.info("Calculating prediction probabilities...")
        
        return self.classifier.predict_proba(X)
    
    def preprocess_dataset(self, data, text_column='text'):
        """
        Preprocess a dataset.
        
        Args:
            data: DataFrame containing the data.
            text_column: Name of the column containing the text.
            
        Returns:
            DataFrame with processed text.
        """
        if self.verbose:
            logger.info("Preprocessing dataset...")
        
        processed_data = data.copy()
        processed_data[text_column] = processed_data[text_column].apply(self.preprocess_text)
        
        return processed_data
    
    def evaluate(self, texts, true_labels):
        """
        Evaluate the classifier on the given data.
        
        Args:
            texts: List of preprocessed text documents.
            true_labels: True labels for the texts.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        y_pred = self.predict(texts)
        y_pred_proba = self.predict_proba(texts)[:, 1]  # Probability for class 1
        
        # Calculate metrics
        acc = accuracy_score(true_labels, y_pred)
        prec = precision_score(true_labels, y_pred, average='macro')
        rec = recall_score(true_labels, y_pred, average='macro')
        f1 = f1_score(true_labels, y_pred, average='macro')
        
        # AUC
        fpr, tpr, _ = roc_curve(true_labels, y_pred_proba)
        auc_val = auc(fpr, tpr)
        
        metrics = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'auc': auc_val
        }
        
        if self.verbose:
            logger.info("Evaluation metrics:")
            logger.info(f"  Accuracy:  {acc:.4f}")
            logger.info(f"  Precision: {prec:.4f}")
            logger.info(f"  Recall:    {rec:.4f}")
            logger.info(f"  F1 Score:  {f1:.4f}")
            logger.info(f"  AUC:       {auc_val:.4f}")
            
            logger.info("\nClassification Report:")
            logger.info(classification_report(true_labels, y_pred))
        
        return metrics

########## 4. Main function ##########

def run_experiments(datasets, 
                   output_dir='../results', 
                   repeat=10, 
                   use_stemming=True,
                   use_lemmatization=False,
                   use_embeddings=True,
                   embedding_type='glove-wiki-gigaword-100',
                   classifier_type='svm',
                   tfidf_ngram_range=(1, 2),
                   tfidf_max_features=2000,
                   verbose=True):
    """
    Run experiments on multiple datasets.
    
    Args:
        datasets: Dictionary mapping dataset names to file paths.
        output_dir: Directory to save results.
        repeat: Number of times to repeat each experiment.
        use_stemming: Whether to apply stemming.
        use_lemmatization: Whether to apply lemmatization.
        use_embeddings: Whether to use word embeddings.
        embedding_type: Type of word embeddings to use.
        classifier_type: Type of classifier to use ('svm' or 'rf').
        tfidf_ngram_range: Range of n-grams for TF-IDF.
        tfidf_max_features: Maximum number of features for TF-IDF.
        verbose: Whether to print progress information.
        
    Returns:
        Dictionary of results for each dataset.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for dataset_name, dataset_path in datasets.items():
        logger.info(f"Processing dataset: {dataset_name}")
        
        # Read dataset
        try:
            df = pd.read_csv(dataset_path)
            logger.info(f"Dataset loaded: {dataset_path}")
            logger.info(f"Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_path}: {e}")
            continue
        
        # Ensure required columns exist and handle different column names
        if 'Title' in df.columns and 'Body' in df.columns:
            # Merge Title and Body
            df['text'] = df.apply(
                lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
                axis=1
            )
        elif 'text' not in df.columns:
            logger.error(f"Dataset {dataset_name} does not have required columns. Skipping.")
            continue
        
        # Ensure 'sentiment' column exists
        if 'class' in df.columns and 'sentiment' not in df.columns:
            df['sentiment'] = df['class']
        elif 'sentiment' not in df.columns:
            logger.error(f"Dataset {dataset_name} does not have 'sentiment' or 'class' column. Skipping.")
            continue
        
        # Initialize result lists
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        auc_values = []
        
        # Run multiple experiments
        for i in range(repeat):
            logger.info(f"Running experiment {i+1}/{repeat} for {dataset_name}")
            
            # Initialize classifier
            classifier = EnhancedBugReportClassifier(
                use_stemming=use_stemming,
                use_lemmatization=use_lemmatization,
                use_embeddings=use_embeddings,
                embedding_type=embedding_type,
                classifier_type=classifier_type,
                tfidf_ngram_range=tfidf_ngram_range,
                tfidf_max_features=tfidf_max_features,
                verbose=verbose
            )
            
            # Split data
            indices = np.arange(df.shape[0])
            train_index, test_index = train_test_split(
                indices, test_size=0.2, random_state=i
            )
            
            train_df = df.iloc[train_index]
            test_df = df.iloc[test_index]
            
            # Preprocess data
            train_df = classifier.preprocess_dataset(train_df)
            test_df = classifier.preprocess_dataset(test_df)
            
            # Train classifier
            classifier.train(
                train_df['text'].values, 
                train_df['sentiment'].values,
                grid_search=True
            )
            
            # Evaluate classifier
            metrics = classifier.evaluate(
                test_df['text'].values,
                test_df['sentiment'].values
            )
            
            # Store metrics
            accuracies.append(metrics['accuracy'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1_score'])
            auc_values.append(metrics['auc'])
        
        # Calculate average metrics
        avg_metrics = {
            'accuracy': np.mean(accuracies),
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1_score': np.mean(f1_scores),
            'auc': np.mean(auc_values)
        }
        
        # Store detailed results
        results = {
            'dataset': dataset_name,
            'avg_metrics': avg_metrics,
            'accuracies': accuracies,
            'precisions': precisions,
            'recalls': recalls,
            'f1_scores': f1_scores,
            'auc_values': auc_values
        }
        
        all_results[dataset_name] = results
        
        # Save results to CSV
        out_csv_name = os.path.join(output_dir, f"{dataset_name}_{classifier_type.upper()}.csv")
        
        try:
            # Check if file exists to determine if header is needed
            header_needed = not os.path.exists(out_csv_name)
            
            df_log = pd.DataFrame(
                {
                    'repeated_times': [repeat],
                    'Accuracy': [avg_metrics['accuracy']],
                    'Precision': [avg_metrics['precision']],
                    'Recall': [avg_metrics['recall']],
                    'F1': [avg_metrics['f1_score']],
                    'AUC': [avg_metrics['auc']],
                    'CV_list(AUC)': [str(auc_values)]
                }
            )
            
            df_log.to_csv(out_csv_name, mode='a', header=header_needed, index=False)
            logger.info(f"Results saved to: {out_csv_name}")
        except Exception as e:
            logger.error(f"Failed to save results to CSV: {e}")
        
        # Print average metrics
        logger.info(f"\n=== Enhanced Classifier Results for {dataset_name} ===")
        logger.info(f"Number of repeats:     {repeat}")
        logger.info(f"Average Accuracy:      {avg_metrics['accuracy']:.4f}")
        logger.info(f"Average Precision:     {avg_metrics['precision']:.4f}")
        logger.info(f"Average Recall:        {avg_metrics['recall']:.4f}")
        logger.info(f"Average F1 score:      {avg_metrics['f1_score']:.4f}")
        logger.info(f"Average AUC:           {avg_metrics['auc']:.4f}")
    
    return all_results

if __name__ == "__main__":
    # Define datasets
    datasets = {
        'pytorch': 'Final Assignment/data/pytorch.csv',
        'tensorflow': 'Final Assignment/data/tensorflow.csv',
        'keras': 'Final Assignment/data/keras.csv',
        'incubator-mxnet': 'Final Assignment/data/incubator-mxnet.csv',
        'caffe': 'Final Assignment/data/caffe.csv'
    }
    
    # Configuration
    config = {
        'repeat': 10,
        'use_stemming': True,
        'use_lemmatization': False,
        'use_embeddings': True,
        'embedding_type': 'glove-wiki-gigaword-100',
        'classifier_type': 'svm',
        'tfidf_ngram_range': (1, 2),
        'tfidf_max_features': 2000,
        'verbose': True
    }
    
    # Run experiments
    start_time = time.time()
    results = run_experiments(datasets, **config)
    end_time = time.time()
    
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds") 