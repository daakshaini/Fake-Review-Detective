# 🔍 Fake Review Detective

## NLP and Machine Learning Based System for Detecting Potentially Deceptive Hotel Reviews

Fake Review Detective is a Natural Language Processing (NLP) and Machine Learning based application that analyzes hotel review text and predicts whether a review is **Likely Truthful** or **Potentially Deceptive**.

The system learns linguistic patterns from a labeled hotel-review dataset and uses a **TF-IDF + Linear Support Vector Machine (SVM)** pipeline for classification.

> **Important:** The system provides a machine-learning prediction based on patterns learned from the dataset. It does not prove that a review is actually genuine or fake.

---

## 📌 Project Overview

Online review platforms contain a large number of customer reviews. Some reviews may contain deceptive or artificially generated content, making it difficult to assess their reliability.

This project explores how Natural Language Processing and Machine Learning can be used to analyze review text and identify linguistic patterns associated with deceptive reviews.

The complete project includes:

- Data understanding and exploration
- Data cleaning
- Exploratory Data Analysis (EDA)
- Text preprocessing
- TF-IDF feature extraction
- Machine Learning model development
- Model comparison
- Cross-validation
- Hyperparameter tuning
- Error analysis
- Model serialization
- Streamlit web application

---

## 🎯 Objective

The main objective of this project is to develop a machine-learning system that analyzes a hotel review and classifies it into one of two categories:

- **Likely Truthful**
- **Potentially Deceptive**

The project focuses specifically on **hotel review text**.

---

## 📊 Dataset

The project uses the **Deceptive Opinion Spam Corpus** containing hotel reviews labeled as truthful or deceptive.

After removing duplicate records, the dataset contains:

- **1,596 unique reviews**
- **800 deceptive reviews**
- **796 truthful reviews**

The dataset includes fields such as:

- Deceptive label
- Hotel
- Polarity
- Source
- Review text

The current machine-learning application focuses primarily on the **review text**.

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Duplicate records were identified and removed.
2. Text was converted to lowercase.
3. HTML tags were removed.
4. URLs were removed.
5. Non-alphabetic characters were removed.
6. Extra whitespace was normalized.
7. Cleaned review text was used for feature extraction.

---

## 📈 Exploratory Data Analysis

Exploratory analysis was performed to understand the dataset before model training.

The analysis included:

- Target-class distribution
- Review polarity distribution
- Hotel distribution
- Review-length analysis
- Comparison of review lengths between truthful and deceptive classes

The dataset was approximately balanced between truthful and deceptive reviews.

---

## 🔤 Feature Extraction

### TF-IDF

Term Frequency-Inverse Document Frequency (**TF-IDF**) was used to convert the cleaned review text into numerical features.

The final text-classification pipeline used:

- Maximum features: **5,000**
- Stop words: **English**
- N-gram range: **Unigrams and bigrams**

The TF-IDF transformation is included inside the machine-learning pipeline so that feature learning is performed correctly using the training data during evaluation.

---

## 🤖 Machine Learning Models

Three classification algorithms were evaluated:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine (SVM)

### Final Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 88.75% | 86.90% | 91.25% | 89.02% |
| Multinomial Naive Bayes | 86.88% | 83.91% | 91.25% | 87.43% |
| Linear SVM | **90.00%** | **89.02%** | **91.25%** | **90.12%** |

The **Linear SVM** achieved the highest accuracy and F1-score on the held-out test set among the evaluated models and was selected as the final application model.

---

## 🔬 Cross-Validation

A **5-fold stratified cross-validation** experiment was performed on the training data for the Linear SVM pipeline.

Average cross-validation results:

| Metric | Average Score |
|---|---:|
| Accuracy | 86.99% |
| Precision | 87.20% |
| Recall | 86.88% |
| F1-Score | 86.98% |

The variation between folds shows that model performance can differ depending on the subset used for validation.

---

## ⚙️ Hyperparameter Tuning

Grid Search with 5-fold cross-validation was performed for the Linear SVM pipeline.

The search explored:

- TF-IDF maximum features
- TF-IDF n-gram range
- SVM regularization parameter `C`

The best tuning configuration obtained a cross-validation F1-score of **0.878**.

On the separate held-out test set, the tuned model achieved:

- Accuracy: **89.69%**
- Precision: **88.02%**
- Recall: **91.87%**
- F1-Score: **89.91%**

The original Linear SVM pipeline was retained as the final application model because it produced better held-out test accuracy and F1-score in this experiment.

---

## 📊 Final Model Evaluation

The final Linear SVM was evaluated on **320 held-out test reviews**.

Results:

- **Correct predictions:** 288
- **Misclassified reviews:** 32
- **Test accuracy:** 90.00%
- **Test error rate:** 10.00%

### Error Breakdown

- **False Positives:** 18
- **False Negatives:** 14

The misclassified reviews demonstrate that truthful and deceptive reviews can sometimes contain similar linguistic patterns. This is one limitation of a text-based classification approach.

---

## 🧠 Model Decision Score

The Streamlit application also displays the **SVM decision score** for a prediction.

The decision score indicates the position of the input review relative to the SVM decision boundary.

> The SVM decision score is **not a probability and not a confidence percentage**.

For the trained classifier:

- A positive score corresponds to the **deceptive** side of the decision boundary.
- A negative score corresponds to the **truthful** side of the decision boundary.

---

## 🏗️ Machine Learning Pipeline

```text
Hotel Review
     ↓
Text Cleaning
     ↓
TF-IDF Feature Extraction
     ↓
Linear SVM
     ↓
Prediction
     ↓
Likely Truthful / Potentially Deceptive