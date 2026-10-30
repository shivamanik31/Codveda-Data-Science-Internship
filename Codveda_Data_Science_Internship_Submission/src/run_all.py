from pathlib import Path
import warnings
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"

OUTPUT.mkdir(exist_ok=True)

RANDOM_STATE = 42


def save_report(filename, content):
    """Save a text report in the outputs folder."""
    (OUTPUT / filename).write_text(content, encoding="utf-8")


def load_csv(filename):
    """Load a CSV file and check that it exists."""
    path = DATA / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def find_column(df, possible_names):
    """Find a column using common column-name variations."""
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(col).lower()): col
        for col in df.columns
    }

    for name in possible_names:
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if key in normalized:
            return normalized[key]

    return None


def make_preprocessor(X):
    """Create preprocessing for numeric and categorical columns."""
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformer = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])

    return transformer


# ==================================================
# LEVEL 1 - TASK 2
# DATA CLEANING AND PREPROCESSING
# ==================================================

def task1_data_cleaning():
    print("\n[LEVEL 1 - TASK 2] Data Cleaning")

    df = load_csv("iris.csv")

    # Standardize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Remove duplicate records
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    # Fill missing values
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Standardize numeric columns
    scaler = StandardScaler()
    if len(numeric_cols) > 0:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Encode categorical columns
    encoded = pd.get_dummies(
        df,
        columns=categorical_cols,
        dtype=int
    )

    df.to_csv(OUTPUT / "iris_cleaned.csv", index=False)
    encoded.to_csv(OUTPUT / "iris_preprocessed.csv", index=False)

    report = (
        "DATA CLEANING AND PREPROCESSING REPORT\n"
        "======================================\n"
        f"Original rows: {before}\n"
        f"Rows after duplicate removal: {len(df)}\n"
        f"Duplicates removed: {removed}\n"
        f"Missing values remaining: {int(df.isna().sum().sum())}\n"
        f"Numeric columns scaled: {len(numeric_cols)}\n"
        f"Categorical columns encoded: {len(categorical_cols)}\n"
    )

    save_report("task1_cleaning_report.txt", report)
    print(report)


# ==================================================
# LEVEL 1 - TASK 3
# EXPLORATORY DATA ANALYSIS
# ==================================================

def task2_eda():
    print("\n[LEVEL 1 - TASK 3] Exploratory Data Analysis")

    df = load_csv("iris.csv")
    df.columns = [str(c).strip() for c in df.columns]

    numeric = df.select_dtypes(include=np.number)

    summary = df.describe(include="all").to_string()
    save_report("eda_summary.txt", summary)

    # Histograms
    numeric.hist(figsize=(10, 7), bins=15)
    plt.tight_layout()
    plt.savefig(OUTPUT / "eda_histograms.png")
    plt.close()

    # Correlation heatmap
    if numeric.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            numeric.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(OUTPUT / "eda_correlation.png")
        plt.close()

    # Boxplots
    if numeric.shape[1] > 0:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=numeric)
        plt.xticks(rotation=30)
        plt.title("Numerical Feature Boxplots")
        plt.tight_layout()
        plt.savefig(OUTPUT / "eda_boxplots.png")
        plt.close()

    print("EDA summary and charts saved.")


# ==================================================
# LEVEL 2 - TASK 1
# HOUSE PRICE REGRESSION
# ==================================================

def task3_regression():
    print("\n[LEVEL 2 - TASK 1] Regression Modeling")

    path = DATA / "house_prediction_raw.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    # This supplied dataset is whitespace-delimited and has no header.
    df = pd.read_csv(path, sep=r"\s+", header=None)

    if df.shape[1] == 14:
        df.columns = [
            "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
            "AGE", "DIS", "RAD", "TAX", "PTRATIO",
            "B", "LSTAT", "MEDV"
        ]
    else:
        # If the file has a header, try loading it as a regular CSV.
        df = pd.read_csv(path)

    target = find_column(
        df,
        ["MEDV", "price", "house_price", "target", "saleprice"]
    )

    if target is None:
        target = df.columns[-1]

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE
        )
    }

    results = []

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor(X_train)),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })

        if name == "Random Forest":
            plt.figure(figsize=(7, 5))
            plt.scatter(y_test, predictions)
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Random Forest: Actual vs Predicted")
            plt.tight_layout()
            plt.savefig(OUTPUT / "regression_predictions.png")
            plt.close()

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT / "regression_results.csv", index=False)

    save_report(
        "regression_report.txt",
        "REGRESSION MODEL COMPARISON\n\n" + result_df.to_string(index=False)
    )

    print(result_df.to_string(index=False))


# ==================================================
# LEVEL 2 - TASK 2
# CHURN CLASSIFICATION
# ==================================================

def task4_classification():
    print("\n[LEVEL 2 - TASK 2] Churn Classification")

    train_df = load_csv("churn_train.csv")
    test_df = load_csv("churn_test.csv")

    train_df.columns = [str(c).strip() for c in train_df.columns]
    test_df.columns = [str(c).strip() for c in test_df.columns]

    target = find_column(
        train_df,
        ["churn", "exited", "attrition", "target", "label"]
    )

    if target is None:
        # Fallback: use the last column as the target.
        target = train_df.columns[-1]

    if target not in test_df.columns:
        raise ValueError(
            f"Target column '{target}' is missing from churn_test.csv. "
            "Check the dataset's target column."
        )

    y_train = train_df[target]
    X_train = train_df.drop(columns=[target])

    y_test = test_df[target]
    X_test = test_df.drop(columns=[target])

    # Remove ID columns if they appear in both files.
    id_cols = [
        c for c in X_train.columns
        if str(c).lower() in ["id", "customerid", "customer_id", "rownumber"]
    ]

    X_train = X_train.drop(columns=id_cols, errors="ignore")
    X_test = X_test.drop(columns=id_cols, errors="ignore")

    # Match test columns to training columns.
    X_test = X_test.reindex(columns=X_train.columns)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        )
    }

    results = []
    best_predictions = None
    best_name = None
    best_f1 = -1

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor(X_train)),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average="binary", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average="binary", zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, average="binary", zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_predictions = predictions
            best_name = name

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT / "churn_model_results.csv", index=False)

    labels = np.unique(
        np.concatenate([
            np.asarray(y_test),
            np.asarray(best_predictions)
        ])
    )

    cm = confusion_matrix(y_test, best_predictions, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT / "churn_confusion_matrix.png")
    plt.close()

    save_report(
        "churn_classification_report.txt",
        "CHURN CLASSIFICATION RESULTS\n\n"
        + result_df.to_string(index=False)
        + "\n\nBest model:\n"
        + best_name
        + "\n\nDetailed classification report:\n"
        + classification_report(y_test, best_predictions, zero_division=0)
    )

    print(result_df.to_string(index=False))


# ==================================================
# LEVEL 3 - TASK 2
# NLP SENTIMENT CLASSIFICATION
# ==================================================

def task5_sentiment():
    print("\n[LEVEL 3 - TASK 2] Sentiment Classification")

    df = load_csv("sentiment.csv")
    df.columns = [str(c).strip() for c in df.columns]

    text_col = find_column(
        df,
        ["text", "review", "sentence", "tweet", "content"]
    )

    label_col = find_column(
        df,
        ["sentiment", "label", "category", "target", "class"]
    )

    # Fallback: select a text column and a different label column.
    if text_col is None:
        text_candidates = [
            c for c in df.columns
            if df[c].dtype == "object"
        ]
        if not text_candidates:
            raise ValueError("Could not identify the text column.")
        text_col = max(
            text_candidates,
            key=lambda c: df[c].astype(str).str.len().mean()
        )

    if label_col is None or label_col == text_col:
        remaining = [c for c in df.columns if c != text_col]
        if not remaining:
            raise ValueError("Could not identify the sentiment label column.")
        label_col = remaining[-1]

    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[label_col],
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2)
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    metrics = (
        "SENTIMENT CLASSIFICATION RESULTS\n"
        "================================\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Macro Precision: {precision:.4f}\n"
        f"Macro Recall: {recall:.4f}\n"
        f"Macro F1 Score: {f1:.4f}\n\n"
        + classification_report(y_test, predictions, zero_division=0)
    )

    save_report("sentiment_report.txt", metrics)

    pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    }).to_csv(OUTPUT / "sentiment_predictions.csv", index=False)

    print(metrics)


# ==================================================
# LEVEL 3 - TASK 3
# NEURAL NETWORK CLASSIFICATION
# ==================================================

def task6_neural_network():
    print("\n[LEVEL 3 - TASK 3] Neural Network")

    df = load_csv("iris.csv")
    df.columns = [str(c).strip() for c in df.columns]

    target = find_column(
        df,
        ["species", "variety", "class", "target"]
    )

    if target is None:
        target = df.columns[-1]

    X = df.drop(columns=[target])
    y = df[target].astype(str)

    # Keep numerical feature columns for the Iris neural network.
    X = X.select_dtypes(include=np.number)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("neural_network", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=RANDOM_STATE,
            early_stopping=True
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    metrics = (
        "NEURAL NETWORK RESULTS\n"
        "======================\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Macro Precision: {precision:.4f}\n"
        f"Macro Recall: {recall:.4f}\n"
        f"Macro F1 Score: {f1:.4f}\n"
    )

    save_report("neural_network_report.txt", metrics)

    # Plot training loss
    nn = model.named_steps["neural_network"]

    plt.figure(figsize=(7, 5))
    plt.plot(nn.loss_curve_)
    plt.title("Neural Network Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(OUTPUT / "neural_network_loss.png")
    plt.close()

    print(metrics)


# ==================================================
# RUN ALL SIX TASKS
# ==================================================

if __name__ == "__main__":
    print("=" * 55)
    print("CODVEDA DATA SCIENCE INTERNSHIP PROJECT")
    print("=" * 55)

    tasks = [
        ("Data Cleaning", task1_data_cleaning),
        ("Exploratory Data Analysis", task2_eda),
        ("Regression", task3_regression),
        ("Churn Classification", task4_classification),
        ("NLP Sentiment Classification", task5_sentiment),
        ("Neural Network", task6_neural_network)
    ]

    completed = []
    failed = []

    for task_name, task_function in tasks:
        try:
            task_function()
            completed.append(task_name)
            print(f"\nSUCCESS: {task_name}")
        except Exception as error:
            failed.append((task_name, str(error)))
            print(f"\nERROR in {task_name}: {error}")

    summary = [
        "CODVEDA PROJECT EXECUTION SUMMARY",
        "=================================",
        "",
        "COMPLETED TASKS:"
    ]

    summary.extend([f"- {name}" for name in completed])

    summary.extend(["", "FAILED TASKS:"])
    summary.extend([f"- {name}: {error}" for name, error in failed])

    save_report(
        "execution_summary.txt",
        "\n".join(summary)
    )

    print("\n" + "\n".join(summary))
    print("\nCheck the outputs folder for generated results.")