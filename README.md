# 🔍 Fake Review Detective

### NLP and Machine Learning Based System for Detecting Potentially Deceptive Hotel Reviews

**Fake Review Detective** is a Data Science and Machine Learning project that analyzes hotel review text and classifies reviews as **Likely Truthful** or **Potentially Deceptive** using Natural Language Processing (NLP).

The application uses TF-IDF for text feature extraction and a trained Linear Support Vector Machine (SVM) classifier to identify linguistic patterns associated with the two classes.

## 🌐 Live Demo

**Try the deployed application here:**

https://fake-review-detective.streamlit.app/

**GitHub Repository:**  
https://github.com/daakshaini/Fake-Review-Detective

## 📌 Project Overview

Online customer reviews influence purchasing decisions, hotel bookings, and customer expectations. However, some reviews may contain deceptive content that makes it difficult for readers to assess their reliability.

This project explores how NLP and supervised machine learning can be used to analyze hotel review text and identify patterns associated with truthful and deceptive reviews.

The application allows users to select a hotel, enter or select a sample review, analyze its text, and view the model's prediction and SVM decision score.

> **Important:** The application provides a machine-learning prediction. It cannot independently verify whether a review is genuinely truthful or deceptive.

## 🎯 Objectives

- Analyze hotel reviews using Natural Language Processing.
- Explore and preprocess a labeled review dataset.
- Convert review text into numerical features using TF-IDF.
- Train and compare multiple machine-learning classifiers.
- Evaluate model performance using appropriate classification metrics.
- Build an interactive Streamlit web application.
- Demonstrate the limitations of automated review classification.

## 📊 Dataset

The project uses a labeled hotel-review dataset containing truthful and deceptive reviews.

After removing duplicate records, the dataset contains:

| Dataset Information | Count |
|---|---:|
| Original records | 1,600 |
| Unique reviews after duplicate removal | 1,596 |
| Deceptive reviews | 800 |
| Truthful reviews | 796 |

The dataset includes fields such as:

- Deceptive label
- Hotel
- Polarity
- Source
- Review text

The final classifier uses the review text as its predictive input.

## 🧹 Data Preprocessing

The data preparation workflow included:

1. Inspecting the dataset structure and column types.
2. Checking missing values and duplicate records.
3. Removing duplicate records.
4. Converting review text to lowercase.
5. Preparing cleaned text for feature extraction.
6. Splitting the data into training and held-out test sets.

The cleaned dataset was used to build and evaluate the classification pipelines.

## 📈 Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the dataset before model training.

The analysis included:

- Dataset dimensions and data types.
- Missing-value and duplicate checks.
- Distribution of truthful and deceptive labels.
- Review-length statistics.
- Comparison of review lengths between the two classes.
- Examination of relevant dataset categories.

These steps helped establish an understanding of the data before training the models.

## 🔤 Feature Extraction: TF-IDF

**Term Frequency-Inverse Document Frequency (TF-IDF)** converts text into numerical features that machine-learning algorithms can process.

TF-IDF represents the importance of terms in a review relative to the collection of documents.

The original feature-extraction experiment used a maximum vocabulary of 5,000 features.

The TF-IDF transformer is integrated into the trained classification pipeline so that the vocabulary and feature weights are learned from the training data.

## 🤖 Machine Learning Models

Three supervised classification algorithms were evaluated:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine (SVM)

### Model Comparison

The following results were obtained from the held-out test evaluation.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 88.75% | 86.90% | 91.25% | 89.02% |
| Multinomial Naive Bayes | 86.88% | 83.91% | 91.25% | 87.43% |
| **Linear SVM** | **90.00%** | **89.02%** | **91.25%** | **90.12%** |

Based on the evaluated test results, the original Linear SVM pipeline was selected for the application.

## 📊 Final Model Evaluation

The final Linear SVM was evaluated on 320 held-out test reviews.

| Metric | Result |
|---|---:|
| Test accuracy | 90.00% |
| Precision | 89.02% |
| Recall | 91.25% |
| F1-score | 90.12% |
| Correct predictions | 288 |
| Misclassified reviews | 32 |

The confusion matrix was:

| Actual Class | Predicted Truthful | Predicted Deceptive |
|---|---:|---:|
| Truthful | 142 | 18 |
| Deceptive | 14 | 146 |

These results show that the model performs well on the evaluated dataset, but it can still misclassify some reviews.

## 🔬 Cross-Validation

A five-fold cross-validation experiment was conducted on the training data for the original Linear SVM pipeline.

| Metric | Average Cross-Validation Score |
|---|---:|
| Accuracy | 86.99% |
| Precision | 87.20% |
| Recall | 86.88% |
| F1-score | 86.98% |

Cross-validation helped evaluate how consistently the model performed across different training and validation folds.

## ⚙️ Hyperparameter Tuning

Grid Search with five-fold cross-validation was used to explore alternative Linear SVM configurations.

The search considered parameters including:

- TF-IDF vocabulary size.
- TF-IDF n-gram range.
- SVM regularization parameter `C`.

The best tuning configuration achieved a cross-validation F1-score of 0.878.

However, the tuned model achieved 89.69% accuracy on the held-out test set, compared with 90.00% for the original model. Therefore, the original Linear SVM pipeline was retained for the application.

## 🏗️ Machine Learning Workflow

```text
Hotel Review Dataset
        ↓
Data Inspection and Cleaning
        ↓
Exploratory Data Analysis
        ↓
Train-Test Split
        ↓
TF-IDF Feature Extraction
        ↓
Machine Learning Model Comparison
        ↓
Linear SVM Selection
        ↓
Model Evaluation and Error Analysis
        ↓
Save Trained Model
        ↓
Streamlit Web Application
        ↓
Review Classification
```

## 🌐 Web Application Features

The Streamlit application provides the following features:

- **Hotel Selection:** Select a hotel from the dropdown.
- **Review Input:** Type or paste a hotel review.
- **Sample Reviews:** Choose an example review for demonstration.
- **Review Analysis:** Submit the review to the trained classifier.
- **Prediction Output:** View a Likely Truthful or Potentially Deceptive result.
- **SVM Decision Score:** View the numerical decision score generated by the classifier.
- **Model Information:** Access the dataset and evaluation metrics in the sidebar.
- **Project Details:** Review the workflow, model evaluation, and limitations.

### Important Note About Hotel Selection

The hotel selection is contextual information displayed by the application. The current trained classifier uses the **review text** for prediction; the selected hotel name is not a predictive feature.

## 🧠 Understanding the SVM Decision Score

The application displays an SVM decision score alongside the classification result.

The score indicates the review's position relative to the model's decision boundary.

For the trained binary classifier:

- A positive score corresponds to the deceptive class.
- A negative score corresponds to the truthful class.

The score is **not a probability or a calibrated confidence percentage**. A prediction should not be treated as proof that the review is genuine or deceptive.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data handling and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine-learning pipelines and evaluation |
| TF-IDF | Text feature extraction |
| Linear SVM | Final classification model |
| Joblib | Saving and loading the trained pipeline |
| Streamlit | Interactive web interface |
| Google Colab | Data analysis and model experiments |
| Visual Studio Code | Application development |
| Git | Version control |
| GitHub | Source-code repository |
| Streamlit Community Cloud | Application deployment |

## 📁 Project Structure

```text
Fake-Review-Detective/
│
├── app.py
├── fake_review_svm_pipeline.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

### File Description

- `app.py` — Contains the Streamlit web application and prediction logic.
- `fake_review_svm_pipeline.pkl` — Saved TF-IDF and Linear SVM pipeline.
- `requirements.txt` — Python dependencies required to run the application.
- `.gitignore` — Specifies files and folders that Git should ignore.
- `README.md` — Project documentation.

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/daakshaini/Fake-Review-Detective.git
```

### 2. Open the project directory

```bash
cd Fake-Review-Detective
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment on Windows

```powershell
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The local application will normally be available at:

http://localhost:8501

## ⚠️ Limitations

- The model was trained and evaluated on a particular hotel-review dataset.
- The classifier primarily uses textual patterns rather than independently verified facts.
- A negative review is not necessarily deceptive, and an extremely positive review is not necessarily fake.
- The model cannot independently confirm whether a reviewer actually stayed at a hotel.
- Some truthful and deceptive reviews can have similar linguistic patterns.
- The current evaluation does not establish performance on unrelated domains such as product reviews or restaurant reviews.
- A held-out test accuracy of 90% means that errors remain possible on unseen reviews.

## 🔮 Future Scope

Possible future improvements include:

- Evaluating larger and more diverse review datasets.
- Exploring character-level text features.
- Testing additional classification algorithms.
- Incorporating review metadata in a carefully designed and evaluated model.
- Exploring explainable NLP techniques.
- Calibrating model outputs if reliable probability estimates are needed.
- Testing the classifier on independently collected hotel reviews.
- Improving prediction analysis and model monitoring.
- Adding optional review history and user accounts if future requirements justify them.

## 🎓 Academic Significance

This project demonstrates an end-to-end Data Science workflow involving:

- Data inspection and preprocessing.
- Exploratory Data Analysis.
- Natural Language Processing.
- Text feature extraction.
- Supervised machine learning.
- Model comparison and evaluation.
- Cross-validation and hyperparameter tuning.
- Error analysis.
- Model serialization.
- Web application development and deployment.

It connects machine-learning experimentation with a usable application that demonstrates the final classifier.

## 👩‍💻 Project Author

**Daakshaini**

Data Science / Machine Learning Academic Project

## 📜 Disclaimer

Fake Review Detective is developed for academic and educational purposes. Its classifications are based on patterns learned from labeled training data. Results should be treated as decision-support information, not definitive evidence that a review is truthful or deceptive.