# 🚢 Titanic Survival Prediction using Machine Learning

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/53026cff-ad65-42b6-a096-c1fe6e609b48" />


This project builds and evaluates Machine Learning models to predict
passenger survival on the Titanic dataset using **Logistic Regression**
and **Decision Tree Classifier**.

------------------------------------------------------------------------

## 📌 Project Overview

The goal of this project is to: - Perform **data cleaning and
preprocessing** - Handle **missing values** - Apply **encoding for
categorical data** - Train **machine learning models** - Evaluate
performance using **accuracy, F1-score, and confusion matrix**

Dataset used: Titanic dataset from `seaborn`.

------------------------------------------------------------------------

## 🧰 Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn

------------------------------------------------------------------------

## 📂 Dataset Information

-   **Total Rows:** 891
-   **Total Columns:** 15
-   **Target Variable:** `survived`
-   **Features Used:**
    -   `pclass`
    -   `sex`
    -   `age`
    -   `fare`

------------------------------------------------------------------------

## 🔧 Data Preprocessing Steps

1.  **Missing Value Handling**
    -   `age` filled with **median**
    -   `deck`, `embarked`, `embark_town` filled with **mode**
2.  **Categorical Encoding**
    -   `sex` encoded using **OneHotEncoder**
3.  **Feature Selection**

``` python
X = titanic[['pclass', 'sex', 'age', 'fare']]
y = titanic['survived']
```

4.  **Train-Test Split**

-   80% Training
-   20% Testing

------------------------------------------------------------------------

## 🤖 Machine Learning Models Used

### 1️⃣ Logistic Regression

-   Used for binary classification
-   Good baseline performance

### 2️⃣ Decision Tree Classifier

-   Max depth = 3
-   Helps capture non-linear relationships

------------------------------------------------------------------------

## ✅ Model Performance

  Model                 Accuracy     F1 Score
  --------------------- ------------ ------------
  Logistic Regression   **80.44%**   **75.52%**
  Decision Tree         79.88%       ---

------------------------------------------------------------------------

## 📊 Confusion Matrix (Decision Tree)

-   Visualized using **Seaborn Heatmap**
-   Helps understand:
    -   True Positives
    -   False Positives
    -   True Negatives
    -   False Negatives

------------------------------------------------------------------------

## 📎 Project Workflow

1.  Load Dataset
2.  Data Exploration
3.  Missing Value Treatment
4.  Feature Encoding
5.  Train-Test Split
6.  Model Training
7.  Prediction
8.  Model Evaluation
9.  Visualization

------------------------------------------------------------------------

## 🎯 Conclusion

-   Logistic Regression performed slightly better than Decision Tree.
-   Proper data cleaning and encoding significantly improved accuracy.
-   The project successfully demonstrates an **end-to-end machine
    learning classification workflow**.

------------------------------------------------------------------------

## 🧑‍💻 Author

**Adarsh**
Machine Learning & Data Analytics Enthusiast
