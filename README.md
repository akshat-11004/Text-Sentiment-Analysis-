# Text Sentiment Analysis

## Overview
This project implements text sentiment analysis using machine learning techniques. It preprocesses textual data and classifies sentiments (e.g., positive, negative, neutral) using Naïve Bayes and Support Vector Machine (SVM) models.

## Features
- **Text Preprocessing**: Tokenization, stopword removal, stemming, and TF-IDF vectorization.
- **Machine Learning Models**: Implements Multinomial Naïve Bayes and Support Vector Machine (SVM) for sentiment classification.
- **Evaluation Metrics**: Accuracy, precision, recall, and F1-score.

## Dataset
- The data contains textual data labeled with sentiment categories.
- The `text` column is preprocessed and converted into numerical features using TF-IDF.

## Model Training
### Multinomial Naïve Bayes
- Applied to TF-IDF-transformed text data.
- Trained using an 80-20 train-test split.
- Evaluated using accuracy and classification report.

### Support Vector Machine (SVM)
- Implemented as an alternative model.
- Trained and tested using the same dataset split.
- Performance compared with Naïve Bayes.

## Evaluation
- The trained models are evaluated using:
  - **Accuracy**: Measures the correctness of predictions.
  - **Classification Report**: Displays precision, recall, and F1-score.
- Performance visualization is done using Seaborn.

## Output of the Classification Report:
```plaintext
Classification Report for Naïve Bayes Classifier:
              precision    recall  f1-score   support

           0       0.73      0.49      0.58      1562
           1       0.56      0.76      0.64      2230
           2       0.73      0.60      0.66      1705

    accuracy                           0.63      5497
   macro avg       0.67      0.62      0.63      5497
weighted avg       0.66      0.63      0.63      5497


Classification Report for SVM:

              precision    recall  f1-score   support

           0       0.78      0.58      0.66      1562
           1       0.63      0.81      0.71      2230
           2       0.80      0.69      0.74      1705

    accuracy                           0.71      5497
   macro avg       0.74      0.69      0.70      5497
weighted avg       0.73      0.71      0.71      5497

```

## Future Enhancements
- Implementing Deep Learning models.
- Fine-tuning hyperparameters for better accuracy.
- Deploying the model as a web API.
