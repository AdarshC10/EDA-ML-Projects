# 🩺 Diabetes Prediction Using Machine Learning  

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/831fb66a-b4f4-4461-bd12-f35b74f4aa2b" />


## 📌 Project Overview
This project predicts whether a person has diabetes using machine learning classification models.  
It uses the **Pima Indians Diabetes Dataset**, which consists of multiple medical diagnostic features that help determine the likelihood of diabetes.

The workflow includes:
- Importing required libraries  
- Loading and exploring the dataset  
- Cleaning and preprocessing  
- Data standardization  
- Training ML models (Logistic Regression & SVM)  
- Evaluating model performance  
- Building a prediction system for new patient data  

---

## 📚 Libraries Used
```python
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

```
## 📥 Dataset Information

Dataset: diabetes.csv
Rows: 768
Columns: 9

---
## 📊 Dataset Sample

| Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI  | DiabetesPedigreeFunction | Age | Outcome |
| ----------- | ------- | ------------- | ------------- | ------- | ---- | ------------------------ | --- | ------- |
| 6           | 148     | 72            | 35            | 0       | 33.6 | 0.627                    | 50  | 1       |
| 1           | 85      | 66            | 29            | 0       | 26.6 | 0.351                    | 31  | 0       |
| 8           | 183     | 64            | 0             | 0       | 23.3 | 0.672                    | 32  | 1       |
| 1           | 89      | 66            | 23            | 94      | 28.1 | 0.167                    | 21  | 0       |
| 0           | 137     | 40            | 35            | 168     | 43.1 | 2.288                    | 33  | 1       |


Outcome Meaning
0 → Non-Diabetic

1 → Diabetic

## Dataset Shape
```python

(768, 9)
```

## Missing Values

0 missing values

## Outcome Distribution


0 : 65.10%
1 : 34.89%

## 🔍 Exploratory Data Analysis
- Summary statistics using .describe()
- Verified no missing values
- Checked distribution of outcome
- Identified varying scales of numeric data → Standardization required

## 🛠 Data Preprocessing

Splitting Features & Target
```python
X = diabetes.drop('Outcome', axis=1)
y = diabetes.Outcome
```
## Standardizing Input Features
```python

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
## Train-Test Split
```python

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```
## 🤖 Machine Learning Models
### 1️⃣ Logistic Regression
```python

lr = LogisticRegression()
lr.fit(X_train, y_train)
```
#### ✔ Accuracy Results
Training Accuracy: 0.7703
Testing Accuracy: 0.7532

### 2️⃣ Support Vector Machine (SVM)
```python

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)
```
#### ✔ Accuracy Results
Training Accuracy: 0.7719
Testing Accuracy: 0.7597

## 🔮 Prediction System
```python

input_data = (4,110,92,0,0,37.6,0.191,30)

input_data_numpy = np.asarray(input_data)
input_data_reshaped = input_data_numpy.reshape(1, -1)

std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)
print(prediction)
```
## Output
```pyhton
[0] → Non-Diabetic
```
## 📌 Conclusion
- Logistic Regression and SVM both provide ~75% accuracy.
- SVM performs slightly better on test data.
- The model successfully predicts diabetes based on a single patient input.
- This project demonstrates full ML workflow:
import → preprocess → model → evaluate → predict

## 📂 Project Workflow Summary

- Import necessary libraries
- Load dataset
- Explore data and check distributions
- Preprocess data (scaling, splitting)
- Train ML models
= Test and evaluate models
- Build prediction function

## 🚀 Future Improvements

- Add hyperparameter tuning
- Use advanced ML models (Random Forest, XGBoost, ANN)
- Improve EDA with visualizations



