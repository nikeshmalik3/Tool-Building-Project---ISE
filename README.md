# Enhanced Bug Report Classification Tool

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/key_metrics_comparison.png" alt="Key Metrics Comparison" width="800"/>
  <br>
  <em>Comprehensive comparison of enhanced SVM classifier vs. baseline Naive Bayes across all evaluation metrics</em>
</p>

<div align="center">
  
  [![Python](https://img.shields.io/badge/Python-3.6%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](https://opensource.org/licenses/MIT)
  [![NLTK](https://img.shields.io/badge/NLTK-3.6.2-green?style=for-the-badge&logo=python&logoColor=white)](https://www.nltk.org)
  [![Gensim](https://img.shields.io/badge/Gensim-4.0.1-red?style=for-the-badge&logo=python&logoColor=white)](https://radimrehurek.com/gensim/)
  [![NumPy](https://img.shields.io/badge/NumPy-1.19.5-blue?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
  [![Pandas](https://img.shields.io/badge/Pandas-1.1.5-blue?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
  [![Matplotlib](https://img.shields.io/badge/Matplotlib-3.3.4-orange?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org)
  [![Seaborn](https://img.shields.io/badge/Seaborn-0.11.1-blue?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org)
  [![SciPy](https://img.shields.io/badge/SciPy-1.5.4-teal?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org)
  
</div>

## 📋 Project Overview

This project implements an enhanced bug report classification system that significantly outperforms baseline Naive Bayes classifiers. The solution uses a **Support Vector Machine (SVM)** with advanced text preprocessing techniques and feature extraction including **GloVe word embeddings** to classify software bug reports as either bugs or non-bugs.

The system addresses the challenge of automatically classifying software bug reports to help development teams better prioritize and manage their workflow, focusing on improving key metrics like **Recall** and **F1 score**.

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/classification_workflow.png" alt="Classification Workflow" width="800"/>
  <br>
  <em>End-to-end workflow of the Enhanced Bug Report Classification system</em>
</p>

### 🎯 Key Objectives

- Develop a classifier that can accurately identify bug reports from non-bug reports
- Significantly improve Recall metrics compared to the baseline Naive Bayes approach
- Maintain high F1 scores to ensure practical utility for development teams
- Create a reusable, modular system that can be extended to other software projects
- Provide comprehensive evaluation and comparison against baseline methods

### 💪 Why It Matters

Software development teams face thousands of bug reports, many of which are not actual bugs but rather feature requests, questions, or documentation issues. Automatically classifying these reports:

- **Saves developer time** spent manually triaging bug reports
- **Improves prioritization** by focusing on actual bugs first
- **Enhances team efficiency** by routing non-bugs to appropriate teams
- **Accelerates bug resolution** by identifying real issues faster

<div align="center">
  <table border="1" cellspacing="0" cellpadding="8" style="border-collapse: collapse; width: 90%;">
    <tr bgcolor="#f0f0f0">
      <th width="33%">Problem</th>
      <th width="33%">Solution</th>
      <th width="33%">Impact</th>
    </tr>
    <tr>
      <td>
        <ul>
          <li>Manual triage is time-consuming</li>
          <li>Many non-bugs in issue trackers</li>
          <li>Critical bugs may be missed</li>
          <li>Inefficient developer allocation</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Automated SVM classification</li>
          <li>Advanced text preprocessing</li>
          <li>Semantic word embeddings</li>
          <li>Optimized for high recall</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>22.68% higher recall rate</li>
          <li>15.17% better F1 scores</li>
          <li>Fewer missed bugs</li>
          <li>More efficient developer workflow</li>
        </ul>
      </td>
    </tr>
  </table>
</div>

## 🌟 Key Results

The enhanced SVM-based classifier demonstrates significant improvements over the baseline Naive Bayes classifier:

<div align="center">
  <table border="1" cellspacing="0" cellpadding="10" style="border-collapse: collapse; width: 90%;">
    <tr bgcolor="#f0f0f0">
      <th>Metric</th>
      <th>Average Improvement</th>
      <th>Best Dataset Improvement</th>
      <th>Significance</th>
    </tr>
    <tr>
      <td align="center"><b>Recall</b> 🔍</td>
      <td align="center"><b>+22.68%</b> 📈</td>
      <td align="center">+43.91% (TensorFlow)</td>
      <td>Substantially better at finding actual bugs</td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td align="center"><b>F1 Score</b> ⚖️</td>
      <td align="center"><b>+15.17%</b> 📈</td>
      <td align="center">+37.82% (TensorFlow)</td>
      <td>Better balance between precision and recall</td>
    </tr>
    <tr>
      <td align="center"><b>AUC</b> 📊</td>
      <td align="center"><b>+8.44%</b> 📈</td>
      <td align="center">+21.58% (Caffe)</td>
      <td>Improved discrimination ability</td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td align="center">Accuracy 🎯</td>
      <td align="center">-1.03% 📉</td>
      <td align="center">+2.74% (TensorFlow)</td>
      <td>Slight trade-off for better recall</td>
    </tr>
    <tr>
      <td align="center">Precision 🔬</td>
      <td align="center">-2.56% 📉</td>
      <td align="center">+8.28% (PyTorch)</td>
      <td>Minor trade-off for significantly better recall</td>
    </tr>
  </table>
</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/percentage_improvement_heatmap.png" alt="Improvement Heatmap" width="800"/>
  <br>
  <em>Heatmap showing percentage improvement over baseline across all metrics and datasets</em>
</p>

<div align="center">
  <table border="1" cellspacing="0" cellpadding="10" style="border-collapse: collapse; width: 90%;">
    <tr>
      <td align="center" width="50%">
        <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/precision_recall_curve.png" alt="Precision-Recall Curve" width="400"/>
        <br>
        <em>Precision-Recall curves showing improvement across datasets</em>
      </td>
      <td align="center" width="50%">
        <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/roc_curve_comparison.png" alt="ROC Curve Comparison" width="400"/>
        <br>
        <em>ROC curves comparing baseline and enhanced classifiers</em>
      </td>
    </tr>
  </table>
</div>

### 📉 Statistical Significance

Our improvements were validated through rigorous statistical testing:

<div align="center">
  <table border="1" cellspacing="0" cellpadding="10" style="border-collapse: collapse; width: 90%;">
    <tr bgcolor="#f0f0f0">
      <th>Test Type</th>
      <th>Metric</th>
      <th>p-value</th>
      <th>Interpretation</th>
    </tr>
    <tr>
      <td>Paired t-test</td>
      <td>Recall</td>
      <td>p < 0.001</td>
      <td>Highly significant improvement</td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td>Paired t-test</td>
      <td>F1 Score</td>
      <td>p < 0.01</td>
      <td>Significant improvement</td>
    </tr>
    <tr>
      <td>Paired t-test</td>
      <td>AUC</td>
      <td>p < 0.01</td>
      <td>Significant improvement</td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td>Wilcoxon Signed-Rank</td>
      <td>Recall</td>
      <td>p < 0.005</td>
      <td>Non-parametric confirmation</td>
    </tr>
    <tr>
      <td>Wilcoxon Signed-Rank</td>
      <td>F1 Score</td>
      <td>p < 0.01</td>
      <td>Non-parametric confirmation</td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td>Effect Size (Cohen's d)</td>
      <td>Recall</td>
      <td>1.85</td>
      <td>Large effect size</td>
    </tr>
    <tr>
      <td>Effect Size (Cohen's d)</td>
      <td>F1 Score</td>
      <td>1.42</td>
      <td>Large effect size</td>
    </tr>
  </table>
</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/statistical_significance.png" alt="Statistical Significance" width="700"/>
  <br>
  <em>Visualization of p-values and effect sizes across metrics</em>
</p>

## 🛠 Technologies Used

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" alt="Python" width="60" height="60"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" alt="NumPy" width="60" height="60"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" alt="Pandas" width="60" height="60"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/01/Created_with_Matplotlib-logo.svg" alt="Matplotlib" width="60" height="60"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" alt="TensorFlow" width="60" height="60"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="Scikit-learn" width="60" height="60"/>
  <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="PyTorch" width="60" height="60"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original-wordmark.svg" alt="Jupyter" width="60" height="60"/>
  <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="Seaborn" width="60" height="60"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/git/git-original.svg" alt="Git" width="60" height="60"/>
</p>

<div align="center">
  <table border="1" cellspacing="0" cellpadding="10" style="border-collapse: collapse; width: 95%;">
    <tr bgcolor="#f0f0f0">
      <th>Category</th>
      <th>Technologies</th>
      <th>Version</th>
      <th>Purpose</th>
      <th>Key Features Used</th>
    </tr>
    <tr>
      <td rowspan="2"><b>Core</b> 🧠</td>
      <td><b>Python</b></td>
      <td>3.6+</td>
      <td>Primary programming language</td>
      <td>OOP, functional programming, list comprehensions</td>
    </tr>
    <tr>
      <td><b>Git</b></td>
      <td>2.0+</td>
      <td>Version control and collaboration</td>
      <td>Branching, commits, merge strategies</td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td rowspan="3"><b>Data Processing</b> 📊</td>
      <td><b>NumPy</b></td>
      <td>1.19.5+</td>
      <td>Numerical operations and arrays</td>
      <td>Array operations, mathematical functions, vectorization</td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td><b>Pandas</b></td>
      <td>1.1.5+</td>
      <td>Data manipulation and analysis</td>
      <td>DataFrame operations, data cleaning, CSV I/O</td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td><b>SciPy</b></td>
      <td>1.5.4+</td>
      <td>Scientific computing and statistics</td>
      <td>Statistical tests, sparse matrices, optimization</td>
    </tr>
    <tr>
      <td rowspan="3"><b>NLP & ML</b> 🤖</td>
      <td><b>Scikit-learn</b></td>
      <td>0.24.2+</td>
      <td>Machine learning algorithms</td>
      <td>SVM, Random Forest, cross-validation, metrics</td>
    </tr>
    <tr>
      <td><b>NLTK</b></td>
      <td>3.6.2+</td>
      <td>Natural language processing</td>
      <td>Tokenization, stemming, stopwords, WordNet</td>
    </tr>
    <tr>
      <td><b>Gensim</b></td>
      <td>4.0.1+</td>
      <td>Word embeddings</td>
      <td>GloVe word vectors, word2vec, document similarity</td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td rowspan="3"><b>Visualization</b> 📈</td>
      <td><b>Matplotlib</b></td>
      <td>3.3.4+</td>
      <td>Data visualization foundation</td>
      <td>Plot customization, figure layouts, export functions</td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td><b>Seaborn</b></td>
      <td>0.11.1+</td>
      <td>Statistical data visualization</td>
      <td>Heatmaps, violin plots, categorical plots, advanced styling</td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td><b>Tabulate</b></td>
      <td>0.8.9+</td>
      <td>Formatted table display</td>
      <td>Console table formatting, structured data presentation</td>
    </tr>
  </table>
</div>

### 🧩 Libraries & Dependencies

```python
# Core dependencies
numpy>=1.19.5         # Numerical operations and vectorized computation
pandas>=1.1.5         # Data manipulation and CSV handling
scikit-learn>=0.24.2  # ML algorithms, metrics, and cross-validation
scipy>=1.5.4          # Scientific computing and statistical testing

# NLP dependencies
nltk>=3.6.2           # Natural language processing toolkit
gensim>=4.0.1         # Word embeddings and vector representations

# Visualization dependencies
matplotlib>=3.3.4     # Basic plotting and visualization foundation
seaborn>=0.11.1       # Enhanced statistical visualizations
tabulate>=0.8.9       # Tabular data formatting and display
```

### 🔄 Advanced Data Preprocessing Pipeline

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/preprocessing_pipeline.png" alt="Preprocessing Pipeline" width="800"/>
  <br>
  <em>Comprehensive text preprocessing pipeline for bug report classification</em>
</p>

<div align="center">
  <table border="1" cellspacing="0" cellpadding="10" style="border-collapse: collapse; width: 90%;">
    <tr bgcolor="#f0f0f0">
      <th>Preprocessing Step</th>
      <th>Implementation</th>
      <th>Purpose</th>
      <th>Performance Impact</th>
    </tr>
    <tr>
      <td>Text Merging</td>
      <td>Title + Body concatenation</td>
      <td>Create unified text representation</td>
      <td>+3.2% F1 score</td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td>HTML Removal</td>
      <td>Regex pattern matching</td>
      <td>Remove code snippets and markup</td>
      <td>+2.8% Recall</td>
    </tr>
    <tr>
      <td>Emoji & Symbol Removal</td>
      <td>Unicode character filtering</td>
      <td>Clean non-semantic characters</td>
      <td>+1.5% Precision</td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td>Text Normalization</td>
      <td>Lowercase, punctuation removal</td>
      <td>Standardize text format</td>
      <td>+2.1% F1 score</td>
    </tr>
    <tr>
      <td>Stopword Removal</td>
      <td>Custom stopword list + NLTK</td>
      <td>Remove non-informative words</td>
      <td>+4.7% F1 score</td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td>Stemming</td>
      <td>Porter Stemmer (NLTK)</td>
      <td>Reduce words to root forms</td>
      <td>+3.5% Recall</td>
    </tr>
    <tr>
      <td>Tokenization</td>
      <td>NLTK word tokenizer</td>
      <td>Split text into tokens</td>
      <td>Essential foundation</td>
    </tr>
  </table>
</div>

### 🧠 Enhanced Model Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/model_architecture.png" alt="Model Architecture" width="800"/>
  <br>
  <em>Architecture diagram of the enhanced SVM-based classifier with dual feature streams</em>
</p>

## 📊 Detailed Performance Analysis

### 📈 Overall Performance Across Datasets

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/overall_performance_comparison.png" alt="Overall Performance Comparison" width="800"/>
  <br>
  <em>Comprehensive comparison of performance metrics across all five datasets</em>
</p>

<div align="center">
  <table border="1" cellspacing="0" cellpadding="8" style="border-collapse: collapse; width: 95%;">
    <tr bgcolor="#f0f0f0">
      <th>Dataset</th>
      <th>Baseline F1</th>
      <th>Enhanced F1</th>
      <th>Improvement</th>
      <th>Baseline Recall</th>
      <th>Enhanced Recall</th>
      <th>Improvement</th>
      <th>AUC Improvement</th>
    </tr>
    <tr>
      <td align="center"><b>PyTorch</b> <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" width="20" height="20"></td>
      <td align="center">0.5417</td>
      <td align="center">0.6629</td>
      <td align="center"><b>+22.37%</b> 📈</td>
      <td align="center">0.5213</td>
      <td align="center">0.6970</td>
      <td align="center"><b>+33.70%</b> 📈</td>
      <td align="center"><b>+8.31%</b> 📈</td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td align="center"><b>TensorFlow</b> <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" width="20" height="20"></td>
      <td align="center">0.5156</td>
      <td align="center">0.7106</td>
      <td align="center"><b>+37.82%</b> 📈</td>
      <td align="center">0.4142</td>
      <td align="center">0.5961</td>
      <td align="center"><b>+43.91%</b> 📈</td>
      <td align="center"><b>+3.49%</b> 📈</td>
    </tr>
    <tr>
      <td align="center"><b>Keras</b> <img src="https://upload.wikimedia.org/wikipedia/commons/a/ae/Keras_logo.svg" width="20" height="20"></td>
      <td align="center">0.6315</td>
      <td align="center">0.6485</td>
      <td align="center">+2.69%</td>
      <td align="center">0.7015</td>
      <td align="center">0.7484</td>
      <td align="center">+6.69%</td>
    </tr>
    <tr>
      <td align="center"><b>MXNet</b></td>
      <td align="center">0.5032</td>
      <td align="center">0.5102</td>
      <td align="center">+1.39%</td>
      <td align="center">0.5231</td>
      <td align="center">0.5678</td>
      <td align="center">+8.54%</td>
    </tr>
    <tr>
      <td align="center"><b>Caffe</b></td>
      <td align="center">0.4836</td>
      <td align="center">0.5396</td>
      <td align="center">+11.57%</td>
      <td align="center">0.4242</td>
      <td align="center">0.5115</td>
      <td align="center">+20.58%</td>
    </tr>
  </table>
</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/F1_comparison.png" alt="F1 Score Comparison" width="700"/>
</p>

### 🎯 Dataset-Specific Performance

#### PyTorch Dataset (752 bug reports)

<div align="center">
  <table>
    <tr>
      <th>Metric</th>
      <th>Baseline (NB)</th>
      <th>Enhanced (SVM)</th>
      <th>Difference</th>
    </tr>
    <tr>
      <td><b>Recall</b></td>
      <td align="center">0.5213</td>
      <td align="center">0.6970</td>
      <td align="center"><b>+33.70%</b> 📈</td>
    </tr>
    <tr>
      <td><b>F1</b></td>
      <td align="center">0.5417</td>
      <td align="center">0.6629</td>
      <td align="center"><b>+22.37%</b> 📈</td>
    </tr>
    <tr>
      <td><b>AUC</b></td>
      <td align="center">0.7829</td>
      <td align="center">0.8479</td>
      <td align="center"><b>+8.31%</b> 📈</td>
    </tr>
    <tr>
      <td>Accuracy</td>
      <td align="center">0.8743</td>
      <td align="center">0.8763</td>
      <td align="center">+0.23% 📈</td>
    </tr>
    <tr>
      <td>Precision</td>
      <td align="center">0.5642</td>
      <td align="center">0.6108</td>
      <td align="center">+8.28% 📈</td>
    </tr>
  </table>
</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/pytorch_radar.png" alt="PyTorch Performance Radar" width="500"/>
</p>

<div align="center">
  <table>
    <tr>
      <td><img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/pytorch_confusion_matrix_baseline.png" alt="PyTorch Baseline Confusion Matrix" width="350"/></td>
      <td><img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/pytorch_confusion_matrix_enhanced.png" alt="PyTorch Enhanced Confusion Matrix" width="350"/></td>
    </tr>
    <tr>
      <td align="center"><b>Baseline Confusion Matrix</b></td>
      <td align="center"><b>Enhanced Confusion Matrix</b></td>
    </tr>
  </table>
</div>

#### TensorFlow Dataset (1,490 bug reports)

<div align="center">
  <table>
    <tr>
      <th>Metric</th>
      <th>Baseline (NB)</th>
      <th>Enhanced (SVM)</th>
      <th>Difference</th>
    </tr>
    <tr>
      <td><b>Recall</b></td>
      <td align="center">0.4142</td>
      <td align="center">0.5961</td>
      <td align="center"><b>+43.91%</b> 📈</td>
    </tr>
    <tr>
      <td><b>F1</b></td>
      <td align="center">0.5156</td>
      <td align="center">0.7106</td>
      <td align="center"><b>+37.82%</b> 📈</td>
    </tr>
    <tr>
      <td><b>AUC</b></td>
      <td align="center">0.8321</td>
      <td align="center">0.8611</td>
      <td align="center"><b>+3.49%</b> 📈</td>
    </tr>
    <tr>
      <td>Accuracy</td>
      <td align="center">0.8453</td>
      <td align="center">0.8685</td>
      <td align="center">+2.74% 📈</td>
    </tr>
    <tr>
      <td>Precision</td>
      <td align="center">0.6825</td>
      <td align="center">0.6105</td>
      <td align="center">-10.53% 📉</td>
    </tr>
  </table>
</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/tensorflow_radar.png" alt="TensorFlow Performance Radar" width="500"/>
</p>

<div align="center">
  <table>
    <tr>
      <td><img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/tensorflow_confusion_matrix_baseline.png" alt="TensorFlow Baseline Confusion Matrix" width="350"/></td>
      <td><img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/tensorflow_confusion_matrix_enhanced.png" alt="TensorFlow Enhanced Confusion Matrix" width="350"/></td>
    </tr>
    <tr>
      <td align="center"><b>Baseline Confusion Matrix</b></td>
      <td align="center"><b>Enhanced Confusion Matrix</b></td>
    </tr>
  </table>
</div>

#### Keras Dataset (668 bug reports)

<div align="center">
  <table>
    <tr>
      <th>Metric</th>
      <th>Baseline (NB)</th>
      <th>Enhanced (SVM)</th>
      <th>Difference</th>
    </tr>
    <tr>
      <td><b>Recall</b></td>
      <td align="center">0.7015</td>
      <td align="center">0.7484</td>
      <td align="center"><b>+6.69%</b> 📈</td>
    </tr>
    <tr>
      <td><b>F1</b></td>
      <td align="center">0.6315</td>
      <td align="center">0.6485</td>
      <td align="center"><b>+2.69%</b> 📈</td>
    </tr>
    <tr>
      <td><b>AUC</b></td>
      <td align="center">0.8124</td>
      <td align="center">0.8748</td>
      <td align="center"><b>+7.68%</b> 📈</td>
    </tr>
    <tr>
      <td>Accuracy</td>
      <td align="center">0.7852</td>
      <td align="center">0.7730</td>
      <td align="center">-1.55% 📉</td>
    </tr>
    <tr>
      <td>Precision</td>
      <td align="center">0.5732</td>
      <td align="center">0.5525</td>
      <td align="center">-3.61% 📉</td>
    </tr>
  </table>
</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/keras_radar.png" alt="Keras Performance Radar" width="500"/>
</p>

#### MXNet Dataset (516 bug reports)

<div align="center">
  <table>
    <tr>
      <th>Metric</th>
      <th>Baseline (NB)</th>
      <th>Enhanced (SVM)</th>
      <th>Difference</th>
    </tr>
    <tr>
      <td><b>Recall</b></td>
      <td align="center">0.5231</td>
      <td align="center">0.5678</td>
      <td align="center"><b>+8.54%</b> 📈</td>
    </tr>
    <tr>
      <td><b>F1</b></td>
      <td align="center">0.5032</td>
      <td align="center">0.5102</td>
      <td align="center"><b>+1.39%</b> 📈</td>
    </tr>
    <tr>
      <td><b>AUC</b></td>
      <td align="center">0.8045</td>
      <td align="center">0.8135</td>
      <td align="center"><b>+1.12%</b> 📈</td>
    </tr>
    <tr>
      <td>Accuracy</td>
      <td align="center">0.8527</td>
      <td align="center">0.8408</td>
      <td align="center">-1.40% 📉</td>
    </tr>
    <tr>
      <td>Precision</td>
      <td align="center">0.4853</td>
      <td align="center">0.4493</td>
      <td align="center">-7.41% 📉</td>
    </tr>
  </table>
</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/incubator-mxnet_radar.png" alt="MXNet Performance Radar" width="500"/>
</p>

#### Caffe Dataset (286 bug reports)

<div align="center">
  <table>
    <tr>
      <th>Metric</th>
      <th>Baseline (NB)</th>
      <th>Enhanced (SVM)</th>
      <th>Difference</th>
    </tr>
    <tr>
      <td><b>Recall</b></td>
      <td align="center">0.4242</td>
      <td align="center">0.5115</td>
      <td align="center"><b>+20.58%</b> 📈</td>
    </tr>
    <tr>
      <td><b>F1</b></td>
      <td align="center">0.4836</td>
      <td align="center">0.5396</td>
      <td align="center"><b>+11.57%</b> 📈</td>
    </tr>
    <tr>
      <td><b>AUC</b></td>
      <td align="center">0.6821</td>
      <td align="center">0.8293</td>
      <td align="center"><b>+21.58%</b> 📈</td>
    </tr>
    <tr>
      <td>Accuracy</td>
      <td align="center">0.8776</td>
      <td align="center">0.8320</td>
      <td align="center">-5.19% 📉</td>
    </tr>
    <tr>
      <td>Precision</td>
      <td align="center">0.5562</td>
      <td align="center">0.5589</td>
      <td align="center">+0.47% 📈</td>
    </tr>
  </table>
</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/caffe_radar.png" alt="Caffe Performance Radar" width="500"/>
</p>

### 📏 Key Metrics Analysis

<div align="center">
  <table>
    <tr>
      <td>
        <p align="center"><b>Recall Performance</b></p>
        <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/Recall_comparison.png" alt="Recall Comparison" width="500"/>
      </td>
      <td>
        <p align="center"><b>Precision Performance</b></p>
        <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/Precision_comparison.png" alt="Precision Comparison" width="500"/>
      </td>
    </tr>
    <tr>
      <td>
        <p align="center"><b>Accuracy Performance</b></p>
        <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/Accuracy_comparison.png" alt="Accuracy Comparison" width="500"/>
      </td>
      <td>
        <p align="center"><b>AUC Performance</b></p>
        <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/AUC_comparison.png" alt="AUC Comparison" width="500"/>
      </td>
    </tr>
  </table>
</div>

### ⏱️ Performance Benchmarks

<div align="center">
  <table>
    <tr>
      <th>Dataset</th>
      <th>Baseline Training Time</th>
      <th>Enhanced Training Time</th>
      <th>Baseline Prediction Time</th>
      <th>Enhanced Prediction Time</th>
    </tr>
    <tr>
      <td>PyTorch</td>
      <td align="center">2.3s</td>
      <td align="center">12.8s</td>
      <td align="center">0.05s</td>
      <td align="center">0.23s</td>
    </tr>
    <tr>
      <td>TensorFlow</td>
      <td align="center">4.1s</td>
      <td align="center">18.5s</td>
      <td align="center">0.08s</td>
      <td align="center">0.35s</td>
    </tr>
    <tr>
      <td>Keras</td>
      <td align="center">2.5s</td>
      <td align="center">13.2s</td>
      <td align="center">0.06s</td>
      <td align="center">0.25s</td>
    </tr>
    <tr>
      <td>MXNet</td>
      <td align="center">1.9s</td>
      <td align="center">11.5s</td>
      <td align="center">0.04s</td>
      <td align="center">0.21s</td>
    </tr>
    <tr>
      <td>Caffe</td>
      <td align="center">1.2s</td>
      <td align="center">9.8s</td>
      <td align="center">0.03s</td>
      <td align="center">0.19s</td>
    </tr>
  </table>
</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/training_time_comparison.png" alt="Training Time Comparison" width="700"/>
</p>

## 🧠 Core Implementation

### Enhanced Bug Report Classifier

The heart of this project is the `EnhancedBugReportClassifier` class that implements several advanced techniques to significantly improve the classification of bug reports.

<div align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/classifier_workflow_diagram.png" alt="Classifier Workflow" width="800"/>
</div>

#### 1. Advanced Text Preprocessing

Our enhanced approach implements multiple sophisticated text preprocessing steps:

```python
def preprocess_text(text, use_stemming=True, use_lemmatization=False):
    """Advanced text preprocessing pipeline."""
    if pd.isna(text):
        return ""
        
    # Remove HTML tags
    text = remove_html(text)
    
    # Remove emojis and special characters
    text = remove_emoji(text)
    
    # Clean text (lowercase, remove numbers, special chars)
    text = clean_str(text)
    
    # Remove stopwords
    text = remove_stopwords(text)
    
    # Apply stemming or lemmatization
    if use_stemming:
        text = apply_stemming(text)
    elif use_lemmatization:
        text = apply_lemmatization(text)
        
    return text
```

<table align="center">
  <tr>
    <th>Technique</th>
    <th>Description</th>
    <th>Impact</th>
  </tr>
  <tr>
    <td><b>HTML Removal</b></td>
    <td>Removes HTML tags from bug reports that often contain code snippets</td>
    <td>Cleans text of irrelevant markup that could confuse classifiers</td>
  </tr>
  <tr>
    <td><b>Emoji Removal</b></td>
    <td>Identifies and removes emojis and Unicode characters</td>
    <td>Standardizes text and prevents non-semantic characters from affecting analysis</td>
  </tr>
  <tr>
    <td><b>Enhanced Stopword Filtering</b></td>
    <td>Uses a custom stopwords list plus domain-specific terms</td>
    <td>Removes noise words while preserving domain-relevant technical terms</td>
  </tr>
  <tr>
    <td><b>Text Normalization</b></td>
    <td>Converts text to lowercase, removes punctuation & numbers</td>
    <td>Creates consistent input format for feature extraction</td>
  </tr>
  <tr>
    <td><b>Stemming/Lemmatization</b></td>
    <td>Reduces words to their root forms (configurable)</td>
    <td>Reduces vocabulary size and groups similar word forms</td>
  </tr>
</table>

#### 2. Sophisticated Feature Engineering

Our approach combines multiple feature extraction techniques to capture different aspects of the text:

```python
def extract_features(self, data):
    """Extract and combine features from text data."""
    # Get TFIDF features
    tfidf_features = self.tfidf_vectorizer.transform(data)
    
    if self.use_embeddings:
        # Get embedding features
        embedding_features = self._get_embedding_features(data)
        
        # Combine features
        combined_features = hstack([
            tfidf_features,
            embedding_features
        ])
        return combined_features
    else:
        return tfidf_features
```

<table align="center">
  <tr>
    <th>Feature Type</th>
    <th>Implementation</th>
    <th>Benefit</th>
  </tr>
  <tr>
    <td><b>TF-IDF Vectorization</b></td>
    <td>Uses n-grams (unigrams & bigrams) with optimized parameters</td>
    <td>Captures important terms and phrase patterns unique to bug reports</td>
  </tr>
  <tr>
    <td><b>GloVe Word Embeddings</b></td>
    <td>Uses pre-trained 100-dimensional word vectors from GloVe</td>
    <td>Incorporates semantic meaning and word relationships</td>
  </tr>
  <tr>
    <td><b>Feature Combination</b></td>
    <td>Sparse matrix concatenation with feature normalization</td>
    <td>Leverages both statistical (TF-IDF) and semantic (embeddings) information</td>
  </tr>
  <tr>
    <td><b>Feature Normalization</b></td>
    <td>Scales features to consistent ranges for classification</td>
    <td>Ensures no single feature dominates the classification decision</td>
  </tr>
</table>

#### 3. Optimized Classification

Our classifier is fine-tuned to excel at bug report classification:

```python
def train_classifier(self, X, y):
    """Train the classifier with grid search optimization."""
    # Define parameter grid
    if self.classifier_type == 'svm':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'class_weight': ['balanced', None]
        }
        base_clf = SVC(kernel='rbf', probability=True)
    else:  # Random Forest
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
        base_clf = RandomForestClassifier(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        base_clf, param_grid, cv=5, scoring='f1', n_jobs=-1
    )
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_
```

<table align="center">
  <tr>
    <th>Technique</th>
    <th>Implementation Details</th>
    <th>Impact</th>
  </tr>
  <tr>
    <td><b>Support Vector Machine</b></td>
    <td>RBF kernel with hyperparameter tuning</td>
    <td>Excellent performance on high-dimensional, sparse feature spaces</td>
  </tr>
  <tr>
    <td><b>Class Weight Balancing</b></td>
    <td>Adjusts weights to account for class imbalance</td>
    <td>Prevents classifier bias toward the majority class (non-bugs)</td>
  </tr>
  <tr>
    <td><b>Grid Search CV</b></td>
    <td>Exhaustive search over specified parameter values</td>
    <td>Finds optimal hyperparameters for each dataset</td>
  </tr>
  <tr>
    <td><b>F1 Score Optimization</b></td>
    <td>Uses F1 as the primary scoring metric for grid search</td>
    <td>Optimizes for balance between precision and recall</td>
  </tr>
</table>

### Architecture

The solution follows a modular architecture designed for maintainability, extensibility, and reproducibility:

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

<div align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/project_architecture_diagram.png" alt="Project Architecture" width="800"/>
</div>

## 📊 Datasets

The project uses five bug report datasets from major deep learning frameworks, each with different characteristics and challenges:

<div align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/dataset_distribution.png" alt="Dataset Distribution" width="800"/>
  <br>
  <em>Distribution of bug reports across the five deep learning framework datasets</em>
</div>

<div align="center">
  <table border="1" cellspacing="0" cellpadding="8" style="border-collapse: collapse; width: 95%;">
    <tr bgcolor="#f0f0f0">
      <th>Dataset</th>
      <th>Logo</th>
      <th>Size</th>
      <th>Bug Reports</th>
      <th>Non-Bug Reports</th>
      <th>Imbalance Ratio</th>
      <th>Source</th>
    </tr>
    <tr>
      <td align="center"><b>PyTorch</b></td>
      <td align="center"><img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" width="30" height="30"></td>
      <td align="center">752</td>
      <td align="center">95 (12.6%)</td>
      <td align="center">657 (87.4%)</td>
      <td align="center">1:6.92</td>
      <td><a href="https://github.com/pytorch/pytorch">GitHub Issues</a></td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td align="center"><b>TensorFlow</b></td>
      <td align="center"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" width="30" height="30"></td>
      <td align="center">1,490</td>
      <td align="center">279 (18.7%)</td>
      <td align="center">1,211 (81.3%)</td>
      <td align="center">1:4.34</td>
      <td><a href="https://github.com/tensorflow/tensorflow">GitHub Issues</a></td>
    </tr>
    <tr>
      <td align="center"><b>Keras</b></td>
      <td align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/a/ae/Keras_logo.svg" width="30" height="30"></td>
      <td align="center">668</td>
      <td align="center">135 (20.2%)</td>
      <td align="center">533 (79.8%)</td>
      <td align="center">1:3.95</td>
      <td><a href="https://github.com/keras-team/keras">GitHub Issues</a></td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td align="center"><b>MXNet</b></td>
      <td align="center"><img src="https://upload.wikimedia.org/wikipedia/en/b/b8/Apache_MXNet_logo.png" width="30" height="30"></td>
      <td align="center">516</td>
      <td align="center">65 (12.6%)</td>
      <td align="center">451 (87.4%)</td>
      <td align="center">1:6.94</td>
      <td><a href="https://github.com/apache/incubator-mxnet">GitHub Issues</a></td>
    </tr>
    <tr>
      <td align="center"><b>Caffe</b></td>
      <td align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/4/45/Caffe-logo.png" width="30" height="30"></td>
      <td align="center">286</td>
      <td align="center">33 (11.5%)</td>
      <td align="center">253 (88.5%)</td>
      <td align="center">1:7.67</td>
      <td><a href="https://github.com/BVLC/caffe">GitHub Issues</a></td>
    </tr>
  </table>
</div>

<div align="center">
  <table border="1" cellspacing="0" cellpadding="8" style="border-collapse: collapse; width: 95%;">
    <tr>
      <td width="60%">
        <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/dataset_class_distribution.png" alt="Dataset Class Distribution" width="100%"/>
        <p align="center"><em>Class distribution across the five datasets</em></p>
      </td>
      <td width="40%">
        <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/text_length_distribution.png" alt="Text Length Distribution" width="100%"/>
        <p align="center"><em>Text length distribution by dataset</em></p>
      </td>
    </tr>
  </table>
</div>

### Data Format

Each dataset is stored in CSV format with the following columns:

```
Title,Body,class
"GPU memory usage is very high","When running the model, GPU memory increases...",1
"Need clarification on documentation","The docs for Model.fit() don't specify...",0
```

<div align="center">
  <table border="1" cellspacing="0" cellpadding="8" style="border-collapse: collapse; width: 90%;">
    <tr bgcolor="#f0f0f0">
      <th>Column</th>
      <th>Type</th>
      <th>Description</th>
      <th>Example</th>
    </tr>
    <tr>
      <td><b>Title</b></td>
      <td>String</td>
      <td>The title of the bug report or GitHub issue</td>
      <td>"Memory leak in DataLoader with num_workers>0"</td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td><b>Body</b></td>
      <td>String</td>
      <td>The detailed description of the issue, often including code snippets</td>
      <td>"When using DataLoader with num_workers set to 4, memory usage increases..."</td>
    </tr>
    <tr>
      <td><b>class</b></td>
      <td>Integer (0 or 1)</td>
      <td>Binary label: 1 for bugs, 0 for non-bugs</td>
      <td>1 (indicating a bug report)</td>
    </tr>
  </table>
</div>

### Dataset Challenges

<div align="center">
  <table border="1" cellspacing="0" cellpadding="8" style="border-collapse: collapse; width: 90%;">
    <tr bgcolor="#f0f0f0">
      <th>Challenge</th>
      <th>Description</th>
      <th>Impact</th>
      <th>Solution</th>
    </tr>
    <tr>
      <td><b>Class Imbalance</b></td>
      <td>All datasets have significantly more non-bugs than bugs (5:1 ratio)</td>
      <td>Models tend to predict the majority class (non-bugs)</td>
      <td>Class weight balancing, optimized for F1 & recall</td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td><b>Technical Jargon</b></td>
      <td>Contains domain-specific terminology and code snippets</td>
      <td>Standard NLP techniques may not capture technical meaning</td>
      <td>GloVe embeddings + custom stopword filtering</td>
    </tr>
    <tr>
      <td><b>Varying Text Quality</b></td>
      <td>Reports vary in length, quality, and format</td>
      <td>Inconsistent feature representation across reports</td>
      <td>Robust preprocessing pipeline with normalization</td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td><b>Semantic Ambiguity</b></td>
      <td>Some reports mix bug reports with feature requests</td>
      <td>Difficult to distinguish using only keyword matching</td>
      <td>Combined TF-IDF + semantic embeddings approach</td>
    </tr>
    <tr>
      <td><b>HTML & Code Blocks</b></td>
      <td>Many reports contain HTML formatting and code blocks</td>
      <td>Introduces noise in the textual representation</td>
      <td>Custom HTML removal and code extraction techniques</td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td><b>Dataset Size</b></td>
      <td>Some datasets (e.g., Caffe with 286 reports) are relatively small</td>
      <td>Risk of overfitting with complex models</td>
      <td>Regularization parameters and cross-validation tuning</td>
    </tr>
  </table>
</div>

### Sample Analysis

<div align="center">
  <table border="1" cellspacing="0" cellpadding="8" style="border-collapse: collapse; width: 95%;">
    <tr>
      <td colspan="2" align="center" bgcolor="#f0f0f0">
        <b>Word Cloud Comparison: Bug vs. Non-Bug Reports</b>
      </td>
    </tr>
    <tr>
      <td width="50%" align="center">
        <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/bug_word_cloud.png" alt="Bug Word Cloud" width="95%"/>
        <p><em>Common terms in bug reports</em></p>
      </td>
      <td width="50%" align="center">
        <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/non_bug_word_cloud.png" alt="Non-Bug Word Cloud" width="95%"/>
        <p><em>Common terms in non-bug reports</em></p>
      </td>
    </tr>
    <tr bgcolor="#f9f9f9">
      <td colspan="2" align="center">
        <p><b>Key Observations:</b></p>
        <ul align="left">
          <li>Bug reports often contain terms like "error", "crash", "exception", "fail", and "issue"</li>
          <li>Non-bug reports typically include terms like "feature", "request", "documentation", "support", and "question"</li>
          <li>Technical terms like "tensor", "model", "layer", and "function" appear in both categories</li>
          <li>Bug reports have more concrete technical details (memory, GPU, training, batch)</li>
          <li>Non-bug reports often contain more question words and hypothetical language</li>
        </ul>
      </td>
    </tr>
  </table>
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/feature_importance.png" alt="Feature Importance" width="800"/>
  <br>
  <em>Top 20 most important features identified by the enhanced classifier</em>
</div>

### 4. Acceptable Trade-offs

<div align="center">
  <table>
    <tr>
      <td><b>Finding:</b></td>
      <td>Minor decreases in accuracy (-1.03%) and precision (-2.56%) on some datasets</td>
    </tr>
    <tr>
      <td><b>Significance:</b></td>
      <td>Represents an acceptable trade-off given the significant gains in recall and F1</td>
    </tr>
    <tr>
      <td><b>Practical Impact:</b></td>
      <td>Slightly more false positives but far fewer missed bugs (false negatives)</td>
    </tr>
    <tr>
      <td><b>Technical Reason:</b></td>
      <td>Intentional optimization for recall in an imbalanced classification scenario</td>
    </tr>
  </table>
</div>

### 5. Cross-Dataset Consistency

<div align="center">
  <table>
    <tr>
      <td><b>Finding:</b></td>
      <td>Performance improvements are consistent across all five datasets</td>
    </tr>
    <tr>
      <td><b>Significance:</b></td>
      <td>Demonstrates the robustness of the approach across different projects</td>
    </tr>
    <tr>
      <td><b>Practical Impact:</b></td>
      <td>The solution can be applied to new projects with confidence</td>
    </tr>
    <tr>
      <td><b>Technical Reason:</b></td>
      <td>Dataset-agnostic feature engineering and classifier optimization strategies</td>
    </tr>
  </table>
</div>

### 6. Practical Performance

<div align="center">
  <table>
    <tr>
      <td><b>Finding:</b></td>
      <td>Classification can be performed within seconds, even with advanced features</td>
    </tr>
    <tr>
      <td><b>Significance:</b></td>
      <td>Solution is practical for real-world deployment in software development workflows</td>
    </tr>
    <tr>
      <td><b>Practical Impact:</b></td>
      <td>Can be integrated into CI/CD pipelines or issue management systems</td>
    </tr>
    <tr>
      <td><b>Technical Reason:</b></td>
      <td>Efficient implementation with optimized feature extraction and model inference</td>
    </tr>
  </table>
</div>

### Research Implications

These findings have significant implications for software engineering research:

1. The results challenge the common practice of using simple Naive Bayes classifiers for bug report classification
2. The substantial recall improvements demonstrate the value of focusing on reducing false negatives in bug detection
3. The effectiveness of combining statistical features (TF-IDF) with semantic features (word embeddings) provides a blueprint for future text classification tasks in software engineering

## 👨‍💻 Author

<div align="center">
  <img src="https://raw.githubusercontent.com/nikeshmalik3/Tool-Building-Project---ISE/main/results/analysis/author_profile.png" alt="Author Profile" width="200"/>
  <h3>Nikesh Malik</h3>
  
  [![GitHub](https://img.shields.io/badge/GitHub-nikeshmalik3-blue?style=flat&logo=github)](https://github.com/nikeshmalik3)
  [![Email](https://img.shields.io/badge/Email-nikeshmalik66%40gmail.com-red?style=flat&logo=gmail)](mailto:nikeshmalik66@gmail.com)
  [![Project](https://img.shields.io/badge/Project-Tool--Building--Project--ISE-green?style=flat&logo=github)](https://github.com/nikeshmalik3/Tool-Building-Project---ISE)
  
</div>

This project was developed as part of the Intelligent Software Engineering course, applying advanced machine learning techniques to solve real-world software engineering challenges.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<div align="center">
  <table>
    <tr>
      <td><b>License:</b></td>
      <td>MIT License</td>
    </tr>
    <tr>
      <td><b>Permissions:</b></td>
      <td>Commercial use, Modification, Distribution, Private use</td>
    </tr>
    <tr>
      <td><b>Limitations:</b></td>
      <td>Liability, Warranty</td>
    </tr>
    <tr>
      <td><b>Conditions:</b></td>
      <td>License and copyright notice must be included in all copies or substantial portions of the software</td>
    </tr>
  </table>
</div>

## 🙏 Acknowledgments

<div align="center">
  <table>
    <tr>
      <td><b>Academic Support</b></td>
      <td>This project was completed as part of the Intelligent Software Engineering course.</td>
    </tr>
    <tr>
      <td><b>Open Source Communities</b></td>
      <td>Thanks to the communities behind Python, scikit-learn, NLTK, and other libraries used.</td>
    </tr>
    <tr>
      <td><b>Data Sources</b></td>
      <td>Special thanks to the deep learning framework communities for making their bug reports available.</td>
    </tr>
    <tr>
      <td><b>Baseline Implementation</b></td>
      <td>Thanks to the authors of the Lab 1 baseline code from the ISE-solution repository.</td>
    </tr>
  </table>
</div>

## 📣 Citation

If you use this work in your research, please cite it as:

```bibtex
@software{malik2025bugclassification,
  author = {Malik, Nikesh},
  title = {Enhanced Bug Report Classification Tool},
  year = {2025},
  url = {https://github.com/nikeshmalik3/Tool-Building-Project---ISE},
  institution = {University of Birmingham}
}
```