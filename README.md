# Enhanced Bug Report Classification Tool

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/key_metrics_comparison.png" alt="Key Metrics Comparison" width="800"/>
</p>

## 📋 Project Overview

This project implements an enhanced bug report classification system that significantly outperforms baseline Naive Bayes classifiers. The solution uses a Support Vector Machine (SVM) with advanced text preprocessing techniques and feature extraction including GloVe word embeddings to classify software bug reports as either bugs or non-bugs.

The system addresses the challenge of automatically classifying software bug reports to help development teams better prioritize and manage their workflow, focusing on improving key metrics like Recall and F1 score.

## 🌟 Key Results

The enhanced SVM-based classifier demonstrates significant improvements over the baseline Naive Bayes classifier:

| Metric | Average Improvement |
|--------|---------------------|
| **Recall** | **+22.68%** |
| **F1 Score** | **+15.17%** |
| **AUC** | **+8.44%** |
| Accuracy | -1.03% |
| Precision | -2.56% |

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/percentage_improvement_heatmap.png" alt="Improvement Heatmap" width="700"/>
</p>

## 🛠 Technologies Used

<p align="center">
  <img src="https://raw.githubusercontent.com/get-icon/geticon/master/icons/python.svg" alt="Python" width="40" height="40"/>
  <img src="https://raw.githubusercontent.com/get-icon/geticon/master/icons/numpy-icon.svg" alt="NumPy" width="40" height="40"/>
  <img src="https://raw.githubusercontent.com/get-icon/geticon/master/icons/pandas-icon.svg" alt="Pandas" width="40" height="40"/>
  <img src="https://raw.githubusercontent.com/get-icon/geticon/master/icons/matplotlib-icon.svg" alt="Matplotlib" width="40" height="40"/>
  <img src="https://raw.githubusercontent.com/get-icon/geticon/master/icons/scikit-learn.svg" alt="Scikit-learn" width="40" height="40"/>
  <img src="https://raw.githubusercontent.com/get-icon/geticon/master/icons/nltk.svg" alt="NLTK" width="40" height="40"/>
</p>

- **Python 3.6+**: Core programming language
- **NumPy & Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and evaluation metrics
- **NLTK**: Natural language processing
- **Gensim**: Word embeddings
- **Matplotlib & Seaborn**: Data visualization

## 📊 Detailed Performance Analysis

### Dataset-Specific Results

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/F1_comparison.png" alt="F1 Score Comparison" width="700"/>
</p>

#### PyTorch Dataset
- Recall: **+33.70%**
- F1: **+22.37%**
- AUC: **+8.31%**
- Accuracy: +0.23%
- Precision: +8.28%

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/pytorch_radar.png" alt="PyTorch Performance Radar" width="500"/>
</p>

#### TensorFlow Dataset
- Recall: **+43.91%**
- F1: **+37.82%**
- AUC: **+3.49%**
- Accuracy: +2.74%
- Precision: -10.53%

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/tensorflow_radar.png" alt="TensorFlow Performance Radar" width="500"/>
</p>

#### Keras Dataset
- Recall: **+6.69%**
- F1: **+2.69%**
- AUC: **+7.68%**
- Accuracy: -1.55%
- Precision: -3.61%

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/keras_radar.png" alt="Keras Performance Radar" width="500"/>
</p>

#### MXNet Dataset
- Recall: **+8.54%**
- F1: **+1.39%**
- AUC: **+1.12%**
- Accuracy: -1.40%
- Precision: -7.41%

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/incubator-mxnet_radar.png" alt="MXNet Performance Radar" width="500"/>
</p>

#### Caffe Dataset
- Recall: **+20.58%**
- F1: **+11.57%**
- AUC: **+21.58%**
- Accuracy: -5.19%
- Precision: +0.47%

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/caffe_radar.png" alt="Caffe Performance Radar" width="500"/>
</p>

### Key Metrics Analysis

#### Recall Performance
<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/Recall_comparison.png" alt="Recall Comparison" width="700"/>
</p>

#### Precision Performance
<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/Precision_comparison.png" alt="Precision Comparison" width="700"/>
</p>

#### Accuracy Performance
<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/Accuracy_comparison.png" alt="Accuracy Comparison" width="700"/>
</p>

#### AUC Performance
<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/AUC_comparison.png" alt="AUC Comparison" width="700"/>
</p>

## 🧠 Core Implementation

### Enhanced Bug Report Classifier

The heart of this project is the `EnhancedBugReportClassifier` class that implements:

1. **Advanced Text Preprocessing**:
   - HTML and emoji removal
   - Improved stopword filtering
   - Text normalization
   - Optional stemming or lemmatization

2. **Sophisticated Feature Engineering**:
   - TF-IDF vectorization with n-grams
   - GloVe word embeddings integration
   - Feature combination and normalization

3. **Optimized Classification**:
   - Support Vector Machine with hyperparameter tuning
   - Class weight balancing
   - Grid search cross-validation

### Architecture

The solution follows a modular architecture:

```
Final Assignment/
├── data/                  # Input datasets (5 deep learning frameworks)
├── src/                   # Source code
│   ├── core/              # Core implementation
│   │   └── enhanced_classifier.py  # Main classifier implementation
│   ├── execution/         # Execution scripts
│   │   ├── run_experiments.py      # Main experiment runner
│   │   └── test_enhanced_classifier.py  # Classifier testing
│   ├── evaluation/        # Evaluation scripts
│   │   ├── benchmark.py            # Benchmark testing
│   │   ├── cross_validation.py     # Cross-validation
│   │   └── statistical_test.py     # Statistical significance testing
│   └── analysis/          # Analysis scripts
│       ├── compare_results.py      # Detailed comparison analysis
│       └── simple_comparison.py    # Basic comparison utilities
├── results/               # Enhanced model results
│   ├── analysis/          # Analysis output and visualizations
│   ├── benchmark/         # Benchmark results
│   └── cross_validation/  # Cross-validation results
├── baseline_results/      # Baseline model results
├── docs/                  # Documentation
│   ├── manual.md          # User manual
│   ├── replication.md     # Replication instructions
│   └── requirements.md    # Requirements specification
├── main.py                # Main execution script
└── requirements.txt       # Project dependencies
```

## 📊 Datasets

The project uses five bug report datasets from major deep learning frameworks:

| Dataset | Size | Bug Reports | Non-Bug Reports | Source |
|---------|------|-------------|----------------|--------|
| PyTorch | 752 | 95 (12.6%) | 657 (87.4%) | [GitHub Issues](https://github.com/pytorch/pytorch) |
| TensorFlow | 1,490 | 279 (18.7%) | 1,211 (81.3%) | [GitHub Issues](https://github.com/tensorflow/tensorflow) |
| Keras | 668 | 135 (20.2%) | 533 (79.8%) | [GitHub Issues](https://github.com/keras-team/keras) |
| MXNet | 516 | 65 (12.6%) | 451 (87.4%) | [GitHub Issues](https://github.com/apache/incubator-mxnet) |
| Caffe | 286 | 33 (11.5%) | 253 (88.5%) | [GitHub Issues](https://github.com/BVLC/caffe) |

Each dataset is in CSV format with the following columns:
- `Title`: The title of the bug report
- `Body`: The description of the bug report
- `class`: Binary label (1 for bugs, 0 for non-bugs)

## 🚀 Getting Started

### Prerequisites

- Python 3.6 or higher
- pip package manager
- git (for cloning the repository)
- Baseline code from [Lab 1](https://github.com/ideas-labo/ISE-solution/tree/main/lab1)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/nikeshmalik3/Tool-Building-Project---ISE.git
   cd Tool-Building-Project---ISE
   ```

2. **Download the baseline code from Lab 1**
   ```bash
   git clone https://github.com/ideas-labo/ISE-solution.git lab1-baseline
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK resources**
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
   ```

5. **Optional: Download Word Embeddings in advance**
   ```bash
   python -c "import gensim.downloader as api; api.load('glove-wiki-gigaword-100')"
   ```

### Running the Project

#### Full Pipeline

To run the entire pipeline (experiments, benchmarking, cross-validation, and analysis):

```bash
python main.py
```

#### Using Existing Results

If you already have results and just want to run the analysis:

```bash
python main.py --skip-experiments
```

#### Running on Specific Datasets

```bash
python main.py --datasets pytorch tensorflow
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--datasets` | Datasets to process | All five datasets |
| `--output-dir` | Base directory for results | results |
| `--baseline-dir` | Directory for baseline results | baseline_results |
| `--skip-experiments` | Skip running experiments | False |
| `--skip-benchmark` | Skip running benchmark comparison | False |
| `--skip-cross-validation` | Skip running cross-validation | False |
| `--skip-statistical-tests` | Skip running statistical tests | False |
| `--skip-analysis` | Skip running comparative analysis | False |
| `--stemming` | Use stemming in preprocessing | True |
| `--lemmatization` | Use lemmatization in preprocessing | False |
| `--embeddings` | Use word embeddings for feature extraction | True |
| `--classifier` | Type of classifier to use (svm or rf) | svm |
| `--k-folds` | Number of folds for cross-validation | 5 |
| `--repeat` | Number of times to repeat each experiment | 10 |
| `--visualize` | Create visualizations | True |

## 📑 Documentation

For more detailed information, refer to:

- [User Manual](docs/manual.md) - Comprehensive guide to using the tool
- [Replication Document](docs/replication.md) - Details on implementation and methodology
- [Requirements Specification](docs/requirements.md) - Project requirements and objectives

## 💡 Key Findings & Insights

1. **Enhanced Recall**: The SVM classifier shows dramatic improvements in recall (up to 43.91%), indicating much better ability to identify actual bugs.

2. **Improved F1 Score**: F1 score improvements (up to 37.82%) demonstrate a better balance between precision and recall.

3. **Better Discrimination**: Consistent AUC improvements across all datasets show enhanced ability to discriminate between bug and non-bug reports.

4. **Trade-offs**: Minor decreases in accuracy and precision on some datasets represent acceptable trade-offs given the significant gains in recall and F1 score.

5. **Consistency**: Performance improvements are consistent across all five datasets, demonstrating the robustness of the approach.

## 👨‍💻 Author

**Nikesh Malik** - [GitHub Profile](https://github.com/nikeshmalik3)
- GitHub Repository: [Tool-Building-Project---ISE](https://github.com/nikeshmalik3/Tool-Building-Project---ISE)
- Email: nikeshmalik66@gmail.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- This project was completed as part of the Intelligent Software Engineering course.
- Thanks to the open-source communities behind the libraries used in this project.
- Deep learning framework communities for making their bug reports available.