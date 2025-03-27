# Bug Report Classification: Baseline vs. Enhanced Model Comparison

Report generated on: 2025-03-27 19:55:31

## Overall Performance Comparison

| Dataset         | Model   |   Accuracy |   Precision |   Recall |     F1 |    AUC |
|:----------------|:--------|-----------:|------------:|---------:|-------:|-------:|
| pytorch         | NB      |     0.8669 |      0.6812 |   0.5947 | 0.6121 | 0.8336 |
| pytorch         | SVM     |     0.8689 |      0.7376 |   0.7950 | 0.7490 | 0.9029 |
| tensorflow      | NB      |     0.8463 |      0.8822 |   0.5718 | 0.5814 | 0.8914 |
| tensorflow      | SVM     |     0.8695 |      0.7893 |   0.8228 | 0.8014 | 0.9225 |
| keras           | NB      |     0.8664 |      0.8040 |   0.7393 | 0.7587 | 0.8453 |
| keras           | SVM     |     0.8530 |      0.7750 |   0.7887 | 0.7791 | 0.9103 |
| incubator-mxnet | NB      |     0.8952 |      0.7851 |   0.6786 | 0.7078 | 0.8916 |
| incubator-mxnet | SVM     |     0.8827 |      0.7269 |   0.7366 | 0.7176 | 0.9015 |
| caffe           | NB      |     0.8966 |      0.6623 |   0.5699 | 0.5790 | 0.7180 |
| caffe           | SVM     |     0.8500 |      0.6654 |   0.6872 | 0.6459 | 0.8729 |

## Detailed Analysis by Metric

### Accuracy

| Dataset         |   Baseline (NB) |   Enhanced (SVM) |   Absolute Difference | Percentage Difference   |
|:----------------|----------------:|-----------------:|----------------------:|:------------------------|
| pytorch         |          0.8669 |           0.8689 |                0.0020 | 0.23%                   |
| tensorflow      |          0.8463 |           0.8695 |                0.0232 | 2.74%                   |
| keras           |          0.8664 |           0.8530 |               -0.0134 | -1.55%                  |
| incubator-mxnet |          0.8952 |           0.8827 |               -0.0125 | -1.40%                  |
| caffe           |          0.8966 |           0.8500 |               -0.0466 | -5.19%                  |

### Precision

| Dataset         |   Baseline (NB) |   Enhanced (SVM) |   Absolute Difference | Percentage Difference   |
|:----------------|----------------:|-----------------:|----------------------:|:------------------------|
| pytorch         |          0.6812 |           0.7376 |                0.0564 | 8.28%                   |
| tensorflow      |          0.8822 |           0.7893 |               -0.0929 | -10.53%                 |
| keras           |          0.8040 |           0.7750 |               -0.0290 | -3.61%                  |
| incubator-mxnet |          0.7851 |           0.7269 |               -0.0582 | -7.41%                  |
| caffe           |          0.6623 |           0.6654 |                0.0031 | 0.47%                   |

### Recall

| Dataset         |   Baseline (NB) |   Enhanced (SVM) |   Absolute Difference | Percentage Difference   |
|:----------------|----------------:|-----------------:|----------------------:|:------------------------|
| pytorch         |          0.5947 |           0.7950 |                0.2004 | 33.70%                  |
| tensorflow      |          0.5718 |           0.8228 |                0.2511 | 43.91%                  |
| keras           |          0.7393 |           0.7887 |                0.0495 | 6.69%                   |
| incubator-mxnet |          0.6786 |           0.7366 |                0.0580 | 8.54%                   |
| caffe           |          0.5699 |           0.6872 |                0.1173 | 20.58%                  |

### F1

| Dataset         |   Baseline (NB) |   Enhanced (SVM) |   Absolute Difference | Percentage Difference   |
|:----------------|----------------:|-----------------:|----------------------:|:------------------------|
| pytorch         |          0.6121 |           0.7490 |                0.1369 | 22.37%                  |
| tensorflow      |          0.5814 |           0.8014 |                0.2199 | 37.82%                  |
| keras           |          0.7587 |           0.7791 |                0.0204 | 2.69%                   |
| incubator-mxnet |          0.7078 |           0.7176 |                0.0098 | 1.39%                   |
| caffe           |          0.5790 |           0.6459 |                0.0670 | 11.57%                  |

### AUC

| Dataset         |   Baseline (NB) |   Enhanced (SVM) |   Absolute Difference | Percentage Difference   |
|:----------------|----------------:|-----------------:|----------------------:|:------------------------|
| pytorch         |          0.8336 |           0.9029 |                0.0693 | 8.31%                   |
| tensorflow      |          0.8914 |           0.9225 |                0.0311 | 3.49%                   |
| keras           |          0.8453 |           0.9103 |                0.0650 | 7.68%                   |
| incubator-mxnet |          0.8916 |           0.9015 |                0.0100 | 1.12%                   |
| caffe           |          0.7180 |           0.8729 |                0.1549 | 21.58%                  |

## Key Observations

1. **Most Improved Metric**: Recall with an average improvement of 22.68%
2. **Second Most Improved**: F1 with an average improvement of 15.17%

### Dataset-Specific Observations

- **tensorflow**: F1 score improved by 37.82%
- **pytorch**: F1 score improved by 22.37%
- **caffe**: F1 score improved by 11.57%
- **keras**: F1 score improved by 2.69%
- **incubator-mxnet**: F1 score improved by 1.39%

## Overall Assessment

The enhanced SVM model shows improvements in 3 out of 5 metrics on average across all datasets.

Notable improvements:

- **Recall**: Average improvement of 22.68%
- **F1**: Average improvement of 15.17%
- **AUC**: Average improvement of 8.44%

## Conclusion

The enhanced SVM classifier demonstrates significant improvements over the baseline Naive Bayes model, particularly in F1 score and AUC, indicating better overall classification performance and discriminative ability.

*Note: Refer to the generated visualizations for graphical representation of these comparisons.*
