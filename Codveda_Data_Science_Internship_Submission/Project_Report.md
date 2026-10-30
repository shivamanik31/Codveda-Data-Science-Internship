# Codveda Data Science Internship – Project Report

## 1. Introduction

This project was developed as part of the Codveda Data Science Internship. It includes six selected tasks across Beginner, Intermediate, and Advanced levels.

The main objective was to gain practical experience in data preprocessing, exploratory data analysis, regression, classification, natural language processing, and neural networks.

## 2. Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Visual Studio Code

## 3. Task Implementations

### 3.1 Level 1 – Task 2: Data Cleaning and Preprocessing

**Dataset:** Iris

The Iris dataset was used to demonstrate data cleaning and preprocessing.

Operations performed:
- Inspected the dataset.
- Identified duplicate records.
- Removed duplicate records.
- Checked missing values.
- Scaled numerical columns.
- Encoded categorical columns.

Results:

| Metric | Result |
|---|---:|
| Original rows | 150 |
| Rows after duplicate removal | 147 |
| Duplicates removed | 3 |
| Missing values remaining | 0 |
| Numeric columns scaled | 4 |
| Categorical columns encoded | 1 |

### 3.2 Level 1 – Task 3: Exploratory Data Analysis

**Dataset:** Iris

Exploratory Data Analysis was performed to understand the dataset and identify patterns.

Activities:
- Generated descriptive statistics.
- Examined feature distributions.
- Created histograms.
- Created boxplots.
- Generated a correlation heatmap.

Generated outputs:
- `eda_summary.txt`
- `eda_histograms.png`
- `eda_boxplots.png`
- `eda_correlation.png`

### 3.3 Level 2 – Task 1: House Price Prediction

Regression models were developed to predict house prices using the supplied dataset.

Models implemented:
1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression

Evaluation metrics:

| Model | MSE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 24.291119 | 4.928602 | 0.668759 |
| Decision Tree | 17.815588 | 4.220852 | 0.757062 |
| Random Forest | 8.597743 | 2.932191 | 0.882759 |

**Observation:** Random Forest achieved the lowest RMSE and highest R² score among the tested models.

### 3.4 Level 2 – Task 2: Customer Churn Classification

Classification models were implemented to predict customer churn.

Models implemented:
1. Logistic Regression
2. Random Forest Classifier

Evaluation results:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.857571 | 0.500000 | 0.242105 | 0.326241 |
| Random Forest | 0.952024 | 0.862069 | 0.789474 | 0.824176 |

**Observation:** Random Forest performed better than Logistic Regression across the reported evaluation metrics.

Generated outputs:
- `churn_classification_report.txt`
- `churn_confusion_matrix.png`
- `churn_model_results.csv`

### 3.5 Level 3 – Task 2: Sentiment Classification using NLP

A Natural Language Processing workflow was implemented using the supplied sentiment dataset.

The task involved text classification and generating sentiment predictions.

Evaluation results:

| Metric | Result |
|---|---:|
| Accuracy | 0.1973 |
| Macro Precision | 0.1400 |
| Macro Recall | 0.1362 |
| Macro F1 Score | 0.1329 |

**Observation:** The model achieved low classification performance. The dataset's label distribution, label variations, and text complexity may have affected the results. Further preprocessing and model tuning are recommended.

Generated outputs:
- `sentiment_predictions.csv`
- `sentiment_report.txt`

### 3.6 Level 3 – Task 3: Neural Network

A neural network classification model was implemented using the Iris dataset.

**Algorithm:** Scikit-learn MLPClassifier

Evaluation results:

| Metric | Result |
|---|---:|
| Accuracy | 0.6667 |
| Macro Precision | 0.5000 |
| Macro Recall | 0.6667 |
| Macro F1 Score | 0.5556 |

Generated outputs:
- `neural_network_report.txt`
- `neural_network_loss.png`

**Implementation note:** The current implementation uses Scikit-learn MLPClassifier rather than TensorFlow/Keras.

## 4. Project Output Summary

The project generated the following categories of results:

1. Cleaned and preprocessed Iris datasets.
2. EDA summaries and visualization charts.
3. Regression model comparison and prediction chart.
4. Churn classification results and confusion matrix.
5. Sentiment predictions and evaluation report.
6. Neural network evaluation report and loss visualization.
7. Overall execution summary.

## 5. Challenges and Observations

- Data preprocessing was required to prepare datasets for modelling.
- Different machine learning algorithms produced different results.
- Random Forest achieved the best results in the regression and churn classification comparisons.
- Sentiment classification achieved low evaluation scores and needs further improvement.
- Neural network performance could be improved through additional model tuning.

## 6. Conclusion

All six selected internship tasks were executed successfully using Python and its Data Science libraries.

The project provided practical experience in data cleaning, exploratory data analysis, regression, classification, NLP, and neural network modelling.

The generated reports, datasets, charts, and predictions are available in the `outputs` directory.

## 7. Execution Instructions

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Execute all tasks from the project root directory:

```bash
python src/run_all.py
```

## 8. Project Status

**Completed tasks:** 6  
**Failed tasks:** 0  
**Execution status:** Successful