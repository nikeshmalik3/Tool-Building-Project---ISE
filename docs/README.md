# Enhanced Bug Report Classification

This project implements an enhanced bug report classifier that improves upon the baseline Naive Bayes with TF-IDF approach by:

1. Adding improved text preprocessing with stemming/lemmatization
2. Using hybrid feature extraction (TF-IDF + Word Embeddings)
3. Implementing a more powerful classifier (SVM)

## Project Structure

- `enhanced_classifier.py`: Main implementation of the enhanced classifier
- `test_enhanced_classifier.py`: Script to run a quick test of the classifier
- `run_experiments.py`: Script to run experiments on all datasets and compare with baseline
- `requirements.txt`: List of required packages
- `results/`: Directory containing results of experiments

## Datasets

The classifier is designed to work with bug report datasets from five different projects:

- PyTorch
- TensorFlow
- Keras
- Apache MXNet
- Caffe

Each dataset contains bug reports with a binary label indicating whether the bug is performance-related (1) or not (0).

## Setup

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Download NLTK resources (this happens automatically when you run the classifier, but you can also run it manually):

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

3. Ensure the datasets are in the correct location (by default, they should be in `data/`).

## Usage

### Basic Usage

```python
from enhanced_classifier import EnhancedBugReportClassifier

# Initialize classifier
classifier = EnhancedBugReportClassifier(
    use_stemming=True,
    use_lemmatization=False,
    use_embeddings=True,
    classifier_type='svm'
)

# Preprocess and train
X_train = ["Bug report text 1", "Bug report text 2", ...]
y_train = [0, 1, ...]
classifier.train(X_train, y_train)

# Make predictions
X_test = ["New bug report", ...]
predictions = classifier.predict(X_test)
```

### Running Experiments

To run experiments on all datasets:

```bash
python run_experiments.py
```

This will:
1. Process each dataset
2. Run 10 repeated experiments for each dataset
3. Calculate and save the average metrics
4. Compare the results with the baseline

## Key Improvements

### Enhanced Preprocessing

The classifier adds the following preprocessing steps to the baseline:

- Stemming: Reducing words to their root form (e.g., "running" → "run")
- Optional lemmatization: Converting words to their base form using linguistic knowledge

### Hybrid Feature Extraction

The classifier combines two feature extraction methods:

1. TF-IDF: Captures term frequency information (like the baseline)
2. Word Embeddings: Uses pre-trained GloVe embeddings to capture semantic relationships between words

This hybrid approach provides both frequency-based features and semantic context.

### Support Vector Machine Classifier

The classifier uses an SVM instead of Naive Bayes:

- SVMs are more robust in high-dimensional spaces
- SVMs can capture more complex decision boundaries
- GridSearchCV is used to find optimal hyperparameters

## Results

Compared to the baseline Naive Bayes model, the enhanced classifier shows improvements in:

- Accuracy
- F1 Score
- AUC

The detailed results can be found in the `results/` directory.

## Author

[Your Name]

## License

This project is licensed under the MIT License - see the LICENSE file for details. 