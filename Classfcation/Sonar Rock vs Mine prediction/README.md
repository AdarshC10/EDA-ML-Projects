# 🎯 Sonar Rock vs Mine Prediction Using Machine Learning

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/e8f62ac9-a8d7-442b-9394-2214158b00ad" />


## 📌 Project Overview

This project predicts whether an object detected by **sonar** is a
**Rock (R)** or a **Mine (M)** using **Machine Learning**.
We use the **Logistic Regression** algorithm to classify sonar signals
based on 60 numerical features.

------------------------------------------------------------------------

## 📚 Libraries Used

``` python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

------------------------------------------------------------------------

## 📥 Dataset Information

**Dataset:** sonar_data.csv
**Rows:** 208
**Columns:** 61 (60 numeric features + 1 label)

------------------------------------------------------------------------

## 📊 Dataset Sample

    0.0200 0.0371 0.0428 ... 0.0180 M
    0.0453 0.0523 0.0843 ... 0.0140 R
    0.0262 0.0582 0.1099 ... 0.0316 R
    ...

------------------------------------------------------------------------

## 📖 Data Dictionary

-   **0--59:** Sonar frequency amplitude values
-   **60:** Output Label
    -   **M → Mine**
    -   **R → Rock**

------------------------------------------------------------------------

## 📏 Dataset Shape

    (208, 61)

------------------------------------------------------------------------

## 🧪 Missing Values

    0 missing values

------------------------------------------------------------------------

## 📊 Output Distribution

    M → 53.36%
    R → 46.63%

------------------------------------------------------------------------

## 🛠 Data Preprocessing

### Split Features and Label

``` python
X = sonar_data.drop(columns=60, axis=1)
y = sonar_data[60]
```

### Train-Test Split

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Train Shape:** (166, 60)
**Test Shape:** (42, 60)

------------------------------------------------------------------------

## 🤖 Model Training (Logistic Regression)

``` python
model = LogisticRegression()
model.fit(X_train, y_train)

X_train_pred = model.predict(X_train)
```

------------------------------------------------------------------------

## 📈 Model Evaluation

### Accuracy Scores

  Dataset    Accuracy
  ---------- ----------
  Training   83.73%
  Testing    78.57%

------------------------------------------------------------------------

## 🔮 Prediction System

``` python
input_data = (0.0307,0.0523,0.0653,...)

input_data_np = np.asarray(input_data)
input_reshaped = input_data_np.reshape(1, -1)

prediction = model.predict(input_reshaped)

if prediction[0] == 'R':
    print("The object is a Rock")
else:
    print("The object is a Mine")
```

------------------------------------------------------------------------

## 📌 Conclusion

✔ Logistic Regression performs well with **\~79% accuracy**\
✔ The model differentiates sonar signals between **Rocks** and
**Mines**\
✔ A complete pipeline: **load → process → train → evaluate → predict**

------------------------------------------------------------------------

## 🚀 Future Enhancements

-   Try SVM, Random Forest, XGBoost
-   Add model comparison
-   Perform hyperparameter tuning
-   Deploy using Flask or Streamlit

------------------------------------------------------------------------

## 📎 Project Assets

-   **Model Notebook**
-   **Dataset (sonar_data.csv)**
-   **Generated Header Image**
