
# 🏥 Medical Insurance Cost Prediction using Machine Learning

This project predicts **medical insurance charges** based on personal and lifestyle attributes. Using the **Linear Regression** model, the project demonstrates data preprocessing, visualization, encoding, model training, evaluation, and prediction.

---

## 🎯 Objective

To build a machine learning model that predicts **medical insurance charges** using the following features:

- **Age**
- **Sex**
- **BMI**
- **Number of children**
- **Smoking habit**
- **Region**
- **Medical charges (Target)**

---

## 📦 Libraries Used

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
````

---

## 📊 Dataset Overview

* Rows: **1338**
* Columns:

  * age
  * sex
  * bmi
  * children
  * smoker
  * region
  * charges

### ✔ Data Checks

* No missing values
* Data types:

  * Numerical → age, bmi, children, charges
  * Categorical → sex, smoker, region

---

## 📈 Exploratory Data Analysis (EDA)

### 🔹 Gender Distribution

* male: 676
* female: 662

### 🔹 Smoker Distribution

* no: 1064
* yes: 274

### 🔹 Region Distribution

* southeast: 364
* southwest: 325
* northwest: 325
* northeast: 324

### 🔹 Important Visualizations

* Countplots for **sex**, **smoker**, **region**, **children**
* Distribution plots for **bmi** and **charges**

---

## 🔧 Data Encoding

Categorical variables were encoded using label mapping:

```python
Insurance_data.replace({'sex':{'male':0,'female':1}}, inplace=True)
Insurance_data.replace({'smoker':{'yes':0,'no':1}}, inplace=True)
Insurance_data.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)
```

---

## ✂️ Feature Selection

### **Features (X)**

* age
* sex
* bmi
* children
* smoker
* region

### **Target (y)**

* charges

Data split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 🤖 Model Building — Linear Regression

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

---

## 📈 Model Evaluation

### Training Performance:

| Metric       | Score      |
| ------------ | ---------- |
| **MAE**      | 4214.89    |
| **MSE**      | 37,337,214 |
| **RMSE**     | 6110.41    |
| **R² score** | 0.741      |

### Testing Performance:

```python
r2_score(y_test, testing_data_prediction) → 0.7830
```

### ✔ Interpretation

* The model performs fairly well with **78% accuracy**.
* Linear Regression captures general trends but may struggle with outliers (high charges for smokers).

---

## 🔮 Prediction System

Example input:

```
(31, 1, 25.74, 0, 1, 0)
```

Where:

| Feature | Meaning            |
| ------- | ------------------ |
| 31      | Age                |
| 1       | Sex (Female)       |
| 25.74   | BMI                |
| 0       | Children           |
| 1       | Non-smoker         |
| 0       | Region = Southeast |

Output:

```
[4016.99]
```

---

## 🏁 Conclusion

* Medical charges are highly influenced by **smoking**, **BMI**, and **age**.
* Linear Regression provides a baseline solution with decent performance.
* Future models like **Random Forest**, **XGBoost**, or **Neural Networks** could improve accuracy further.

---

## 🔧 Future Enhancements

* Apply advanced regression models
* Feature engineering (BMI categories, smoker interaction terms)
* Hyperparameter tuning
* Deployment using Flask/Streamlit

---

## 👨‍💻 Author

**Adarsh C**
MERN Stack Developer & Data Science Enthusiast

---


```
