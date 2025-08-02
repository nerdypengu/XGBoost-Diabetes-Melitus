# Diabetes Prediction using XGBoost

This project focuses on predicting diabetes using the XGBoost classifier. The project involves exploratory data analysis (EDA), outlier handling, feature selection, data normalization, and model training/evaluation with both imbalanced and balanced datasets.

## Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Key Steps](#key-steps)
  - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Outlier Handling](#2-outlier-handling)
  - [3. Feature Selection](#3-feature-selection)
  - [4. Data Normalization](#4-data-normalization)
  - [5. Class Imbalance Handling (SMOTE & Oversampling)](#5-class-imbalance-handling-smote--oversampling)
  - [6. Model Training and Evaluation (XGBoost)](#6-model-training-and-evaluation-xgboost)
- [Results](#results)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## About the Project

This project aims to build a robust machine learning model to predict diabetes. The core of the prediction is handled by the XGBoost algorithm, known for its efficiency and performance. The process includes thorough data cleaning and preparation to ensure the model performs optimally, especially when dealing with imbalanced datasets common in medical diagnostics.

## Dataset

The dataset used in this project is named `diabetes.csv`. It contains various health metrics and a target variable (`Outcome`) indicating the presence or absence of diabetes. The dataset is taken from https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

**Columns:**
* `Pregnancies`: Number of times pregnant.
* `Glucose`: Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
* `BloodPressure`: Diastolic blood pressure (mm Hg).
* `SkinThickness`: Triceps skin fold thickness (mm).
* `Insulin`: 2-Hour serum insulin (mu U/ml).
* `BMI`: Body mass index (weight in kg/(height in m)^2).
* `DiabetesPedigreeFunction`: Diabetes pedigree function.
* `Age`: Age in years.
* `Outcome`: Class variable (0 or 1, where 1 indicates diabetes).

## Key Steps

### 1. Exploratory Data Analysis (EDA)

* **Initial Data Inspection:** `data.head()` and `data.describe()` were used to get a first look at the data's structure, basic statistics, and identify potential issues.
* **Missing Values Check:** `data.isnull().sum()` confirmed no explicit missing values in the dataset.
* **Distribution Visualization:** Bar plots for categorical features (`Pregnancies`, `Age`) and scatter plots (`Glucose` vs. `Age`, `BMI` vs. `Age` colored by `Outcome`) were generated to understand their distributions and relationships with the target variable.
* **Class Distribution:** A pie chart and `value_counts()` were used to visualize the distribution of the `Outcome` variable, revealing class imbalance.

### 2. Outlier Handling

* **Box Plots:** Box plots for all numerical features (excluding `Outcome`) were generated to visually identify outliers.
* **Isolation Forest:** An Isolation Forest model was applied to detect outliers in the dataset.
* **IQR Method:** A custom function `detect_outliers` was implemented using the Interquartile Range (IQR) method to identify and list outlier indices. Outliers identified by this method (those appearing in more than 2 feature outlier lists) were removed from the dataset.

### 3. Feature Selection

* **Correlation Heatmap:** A correlation heatmap was generated to visualize the correlation matrix between all features.
* **Identifying Relevant Features:** Features with an absolute correlation coefficient of 0.2 or higher with the `Outcome` variable were selected for the model. This resulted in the removal of `BloodPressure`, `SkinThickness`, `Insulin`, and `DiabetesPedigreeFunction`.

### 4. Data Normalization

* `StandardScaler` from `sklearn.preprocessing` was used to standardize the features (`X`) by removing the mean and scaling to unit variance. This step is crucial for many machine learning algorithms to perform optimally.

### 5. Class Imbalance Handling (SMOTE & Oversampling)

* The dataset was found to be imbalanced, with a significantly higher number of non-diabetic cases (Outcome=0) compared to diabetic cases (Outcome=1).
* **SMOTE (Synthetic Minority Over-sampling Technique):** Applied SMOTE to the training data to balance the class distribution by creating synthetic samples of the minority class. The class distribution before and after SMOTE was verified.
* **Oversampling (using `resample`):** The minority class was oversampled to match the number of samples in the majority class, creating a perfectly balanced dataset. This balanced dataset was then used for subsequent model training.

### 6. Model Training and Evaluation (XGBoost)

XGBoost Classifier was used for diabetes prediction. Various approaches to model training and hyperparameter tuning were explored:

* **Pure XGBoost:**
    * Trained a basic `XGBClassifier` without any hyperparameter tuning.
    * Evaluated its performance using accuracy, precision, and recall scores.

* **XGBoost Manual Tuning:**
    * Iterated through different `max_depth` values (1 to 10) while keeping other hyperparameters fixed (`n_estimators=300`, `learning_rate=0.01`, `gamma=0`, `subsample=0.78`, `colsample_bytree=1`).
    * Calculated and printed mean/best accuracy, precision (weighted), recall (weighted), and F1-score (weighted) across these iterations.
    * Plotted these metrics against `max_depth` to visualize performance trends.

* **XGBoost Randomized Search CV:**
    * Defined a parameter grid for `learning_rate`, `max_depth`, and `n_estimators`.
    * Performed `RandomizedSearchCV` with 50 iterations and 5-fold cross-validation to find the best hyperparameters based on accuracy.
    * Printed the best parameters and the best randomized search accuracy.
    * Evaluated the best model's performance on the test set using accuracy, precision (weighted), recall (weighted), and F1-score (weighted).

* **XGBoost Grid Search CV:**
    * Attempted `GridSearchCV` with a wider range of hyperparameters. (Note: The notebook indicates this process was interrupted due to computational intensity on the laptop, but the best parameters found so far were printed.)
    * The best parameters from the partial run were: `{'learning_rate': 0.09, 'max_depth': 1, 'n_estimators': 1000}` with an accuracy score of `0.7883`.

* **XGBoost Cross-Validation (with best parameters):**
    * Used `KFold` cross-validation (5 splits) with specific hyperparameters (`n_estimators=100`, `max_depth=11`, `learning_rate=0.09`).
    * Calculated and printed cross-validation scores for accuracy.
    * Computed mean precision, recall, and F1-score by iterating through the folds and making predictions.

## Results

The project showcases the impact of data preprocessing and hyperparameter tuning on model performance. The balanced dataset and tuned XGBoost models are expected to provide better and more reliable predictions for diabetes.

**Summary of Key Model Performances:**

* **Pure XGBoost (Imbalanced Data):**
    * Accuracy: 73.33%
    * Precision: 54.55%
    * Recall: 66.67%

* **XGBoost Manual Tuning (Imbalanced Data - average across `max_depth` 1-10):**
    * Mean Accuracy: 74.33%
    * Best Accuracy: 75.33%
    * Mean Precision: 76.02%
    * Best Precision: 77.94%
    * Mean Recall: 74.33%
    * Best Recall: 75.33%
    * Mean F1 Score: 74.86%
    * Best F1 Score: 76.08%

* **XGBoost Randomized Search (Balanced Data - Best Model):**
    * Best Parameters: `{'n_estimators': 100, 'max_depth': 1, 'learning_rate': 0.09}`
    * Best Randomized Search Accuracy (Cross-validation): 78.33%
    * Test Accuracy: 76.00%
    * Test Precision: 75.22%
    * Test Recall: 76.00%
    * Test F1 Score: 75.48%

* **XGBoost Grid Search (Balanced Data - Partial Run):**
    * Best Parameters: `{'learning_rate': 0.09, 'max_depth': 1, 'n_estimators': 1000}`
    * Best Grid Search Accuracy (Cross-validation): 78.83%

* **XGBoost Cross-Validation (with fixed parameters on Balanced Data):**
    * Mean Cross-validation Accuracy: 74.40%
    * Mean Precision: 74.43%
    * Mean Recall: 74.40%
    * Mean F1 Score: 74.31%

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Python 3.x
* Jupyter Notebook or JupyterLab

### Installation

1.  Clone the repo:
    ```bash
    git clone [https://github.com/your_username/diabetes-prediction-xgboost.git](https://github.com/your_username/diabetes-prediction-xgboost.git)
    ```
2.  Navigate to the project directory:
    ```bash
    cd diabetes-prediction-xgboost
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    (You might need to create a `requirements.txt` file from the `import` statements in the notebook.)

## Usage

1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook "FP PAP XGBoost.ipynb"
    ```
2.  Run all cells in the notebook to execute the data preprocessing, model training, and evaluation steps.
3.  Modify the code as needed to experiment with different parameters or models.

## Dependencies

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `xgboost`
* `imblearn` (for SMOTE)
