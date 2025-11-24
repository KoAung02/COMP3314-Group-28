import numpy as np
import pandas as pd
import time
import sys
import warnings
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
np.set_printoptions(precision=4)

def load_dataset(name, data_id):
    """
    Loads and preprocesses a dataset from OpenML.
    Handles imputation, scaling, and label encoding.
    """
    try:
        data = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
    except Exception as e:
        print(f"Error fetching data {name}: {e}")
        return None, None

    X = data.data
    y = data.target

    # Handle categorical features by one-hot encoding
    # Get object type columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        # Convert boolean columns to int
        for col in X.columns:
            if X[col].dtype == 'bool':
                X[col] = X[col].astype(int)

    # Impute missing values (e.g., in 'Hepatitis')
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Encode labels to 1 and -1
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = np.where(y == 0, -1, 1)

    if X.shape[0] != y.shape[0]:
        print(f"Error: Mismatch in sample sizes. X: {X.shape[0]}, y: {y.shape[0]}")
        return None, None

    return X, y

def load_wpbc_dataset():
    """
    Loads and preprocesses the 'Breast Cancer Wisconsin (Prognostic)' dataset from UCI.
    Handles specific CSV loading, missing values ('?'), imputation, scaling,
    and label encoding to -1/1.
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data'

    try:
        # Load the dataset, specifying no header and '?' as NA values
        df = pd.read_csv(url, header=None, na_values='?')
    except Exception as e:
        print(f"Error fetching WPBC data: {e}")
        return None, None

    # Drop the first column (ID number)
    df = df.drop(df.columns[0], axis=1)

    # Separate features (X) and target (y)
    # The target 'Diagnosis' is now in the first column (index 0)
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    # Impute missing values (e.g., in 'X' columns 1-33 originally)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Encode labels: 'R' (recurrence) and 'N' (non-recurrence)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Transform encoded labels to -1 and 1
    # Assuming 'N' maps to 0 and 'R' maps to 1 after LabelEncoder
    y = np.where(y == 0, -1, 1)

    if X.shape[0] != y.shape[0]:
        print(f"Error: Mismatch in sample sizes. X: {X.shape[0]}, y: {X.shape[0]}")
        return None, None

    return X, y

def load_bupa_liver_dataset():
    """
    Loads and preprocesses the 'BUPA Liver Disorders' dataset from UCI.
    Handles CSV loading, separates features and target, performs imputation, scaling,
    and label encoding to -1/1.
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data'

    try:
        # Load the dataset, specifying no header
        df = pd.read_csv(url, header=None)
    except Exception as e:
        print(f"Error fetching BUPA liver data: {e}")
        return None, None

    # Separate features (X) and target (y)
    # The target is the last column (index 6)
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Encode labels (assuming 1 and 2 need to be mapped to -1 and 1)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Transform encoded labels to -1 and 1
    # Assuming 1 maps to 0 and 2 maps to 1 after LabelEncoder
    y = np.where(y == 0, -1, 1)

    if X.shape[0] != y.shape[0]:
        print(f"Error: Mismatch in sample sizes. X: {X.shape[0]}, y: {y.shape[0]}")
        return None, None

    return X, y

def run_experiments():
    """
    Main function to run all experiments for Tables 1, 2, and 3.
    """
    # Datasets from the paper, mapped to OpenML data_id
    DATASETS_TABLE_1 = {
        "Heart-statlog": 53,
        "Heart-c": 49,
        "Hepatitis": 55,
        "Ionosphere": 59,
        "Sonar": 40,
        "Votes": 56,
        "Pima-Indian": 37,
        "Australian": 40981,
        "CMC": 23
    }

    DATASETS_TABLE_2 = {
        "Hepatitis": 55,
        "WPBC": "custom_wpbc",
        "BUPA liver": "custom_bupa",
        "Votes": 56
    }

    # Parameters
    n_splits = 10
    random_state = 42

    print("="*80)
    print("Standard SVM Experiment Reproduction Script (Tables 1, 2, 3)")
    print("="*80)
    print("WARNING: Results will differ from the paper due to:")
    print("1. Different 10-fold cross-validation splits.")
    print("2. Fixed hyperparameters (C, gamma) instead of tuning.")
    print("3. Standardized preprocessing (imputation, scaling) not detailed in paper.")
    print("4. Standard 'sklearn.svm.SVC' vs. paper's 'Gunn SVM [8]'.")
    print("="*80)

    # --- Run Experiment for Table 1 (Linear Kernel Accuracy) ---
    print("\n" + "="*80)
    print("Running Linear Kernel SVM Experiments (Table 1: Accuracy)")
    print("="*80)
    print(f"{'Dataset':<15} | {'SVM (Linear) Acc (%)':<22}") 
    print("-"*80)

    # Store results for Table 3
    table3_results = {}

    for name, data_id in DATASETS_TABLE_1.items():
        X, y = load_dataset(name, data_id)
        if X is None:
            continue

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        svm_scores = []
        svm_times = []

        fold_count = 0
        for train_idx, test_idx in kf.split(X, y):
            fold_count += 1
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Standard SVM
            svm = SVC(kernel='linear', C=1.0, random_state=random_state)
            t_start = time.time()
            svm.fit(X_train, y_train)
            t_end = time.time()
            svm_times.append(t_end - t_start)
            preds = svm.predict(X_test)
            svm_scores.append(accuracy_score(y_test, preds))

        # Calculate and print accuracy for Table 1
        svm_acc_mean = np.mean(svm_scores) * 100
        svm_acc_std = np.std(svm_scores) * 100 
        print(f"{name:<15} | {svm_acc_mean:6.2f} +/- {svm_acc_std:<12.2f}")

        # Store total time for Table 3
        table3_results[name] = np.sum(svm_times)

    # --- Run Experiment for Table 3 (Linear Kernel Time) ---
    print("\n" + "="*80)
    print("Running Linear Kernel SVM Experiments (Table 3: Time)")
    print("="*80)
    print(f"{'Dataset':<15} | {'SVM Time (s)':<15}")
    print("-"*80)

    for name in DATASETS_TABLE_1.keys():
        if name in table3_results:
            print(f"{name:<15} | {table3_results[name]:<15.4f}")
        else:
            print(f"{name:<15} | {'N/A':<15}")

    # --- Run Experiment for Table 2 (RBF Kernel Accuracy) ---
    print("\n" + "="*80)
    print("Running RBF Kernel SVM Experiments (Table 2)")
    print("="*80)
    print(f"{'Dataset':<15} | {'SVM (RBF) Acc (%)':<20}") 
    print("-"*80)

    for name, data_id in DATASETS_TABLE_2.items():
        # Conditional loading for custom datasets
        if name == "WPBC":
            X, y = load_wpbc_dataset()
        elif name == "BUPA liver":
            X, y = load_bupa_liver_dataset()
        else:
            X, y = load_dataset(name, data_id)

        if X is None:
            continue

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        svm_scores = []

        for train_idx, test_idx in kf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Standard SVM RBF
            svm_k = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=random_state)
            svm_k.fit(X_train, y_train)
            preds = svm_k.predict(X_test)
            svm_scores.append(accuracy_score(y_test, preds))

        # Calculate and print results for this dataset
        svm_acc_mean = np.mean(svm_scores) * 100
        svm_acc_std = np.std(svm_scores) * 100 

        print(f"{name:<15} | {svm_acc_mean:6.2f} +/- {svm_acc_std:<10.2f}") 

    print("\n" + "="*80)
    print("Experiment run complete.")
    print("="*80)

if __name__ == "__main__":
    run_experiments()
