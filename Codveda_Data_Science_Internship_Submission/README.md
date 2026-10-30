# Codveda Data Science Internship – Project Submission

## Project Overview

This repository contains six Data Science internship tasks completed across three levels. The project demonstrates practical applications of data cleaning, exploratory data analysis, machine learning, natural language processing, and neural networks.

## Technologies Used

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code

## Tasks Completed

### Level 1 – Beginner

**Task 2: Data Cleaning and Preprocessing**
- Used the Iris dataset.
- Removed duplicate records.
- Checked and handled missing values.
- Scaled numerical features.
- Encoded categorical features.

**Task 3: Exploratory Data Analysis (EDA)**
- Generated descriptive statistics.
- Analyzed feature distributions.
- Created histograms and boxplots.
- Visualized feature correlations.

### Level 2 – Intermediate

**Task 1: House Price Prediction**
- Implemented Linear Regression.
- Implemented Decision Tree Regression.
- Implemented Random Forest Regression.
- Compared models using MSE, RMSE, and R².

**Task 2: Customer Churn Classification**
- Implemented Logistic Regression.
- Implemented Random Forest Classification.
- Evaluated accuracy, precision, recall, and F1-score.
- Generated a confusion matrix.

### Level 3 – Advanced

**Task 2: Sentiment Classification using NLP**
- Processed the supplied sentiment dataset.
- Built a text classification workflow.
- Generated sentiment predictions.
- Evaluated model performance.

**Task 3: Neural Network**
- Implemented a neural network using Scikit-learn MLPClassifier.
- Used the Iris dataset.
- Evaluated classification performance.
- Generated a training loss visualization.

## Model Performance

### Regression Results

| Model | RMSE | R² Score |
|---|---:|---:|
| Linear Regression | 4.9286 | 0.6688 |
| Decision Tree | 4.2209 | 0.7571 |
| Random Forest | 2.9322 | 0.8828 |

### Churn Classification Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 85.76% | 50.00% | 24.21% | 32.62% |
| Random Forest | 95.20% | 86.21% | 78.95% | 82.42% |

### Sentiment Classification Results

- Accuracy: 19.73%
- Macro Precision: 14.00%
- Macro Recall: 13.62%
- Macro F1 Score: 13.29%

### Neural Network Results

- Accuracy: 66.67%
- Macro Precision: 50.00%
- Macro Recall: 66.67%
- Macro F1 Score: 55.56%

Note: The sentiment classification model achieved low performance. Further preprocessing, label analysis, feature engineering, and model tuning may improve its results.

## Project Structure

```text
Codveda_Data_Science_Internship_Submission/
│
├── data/
├── outputs/
├── reports/
│   └── Project_Report.md
├── screenshots/
├── src/
│   └── run_all.py
├── requirements.txt
├── README.md
├── .gitignore
└── demo_script.txt
```

## Installation and Execution

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Run all tasks:

```bash
python src/run_all.py
```

## Output Files

The `outputs/` directory contains:
- Cleaned and preprocessed datasets.
- EDA reports and visualization charts.
- Regression results and prediction charts.
- Churn classification reports and confusion matrix.
- Sentiment predictions and evaluation report.
- Neural network evaluation report and loss chart.
- Execution summary.

## Conclusion

All six selected tasks executed successfully. This project demonstrates practical experience with important Data Science workflows, from data preprocessing and visualization to machine learning and NLP.

## Internship

**Organization:** Codveda Technologies  
**Project:** Data Science Internship  
**Status:** Six selected tasks executed successfully.