# Big Mart Sales Prediction – Data Analysis & Machine Learning

This project analyzes the **Big Mart Sales dataset** and builds a machine learning model to predict **Item Outlet Sales** using Python.  
Below is the full code and explanation.

---

## 📁 Libraries Used

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")
```

---

## 📥 Load Data

```python
Big_mart = pd.read_csv("Train.csv")
Big_mart.head()
Big_mart.shape
Big_mart.info()
```

---

## 🔍 Check Missing Values

```python
Big_mart.isnull().sum()
```

---

## 🧹 Data Cleaning

### Fill Missing Item_Weight

```python
Big_mart['Item_Weight'] = Big_mart['Item_Weight'].fillna(Big_mart['Item_Weight'].median())
```

### Fill Missing Outlet_Size

```python
Big_mart['Outlet_Size'] = Big_mart['Outlet_Size'].fillna(Big_mart['Outlet_Size'].mode()[0])
```

---

## 🔁 Fix Inconsistent Categories

```python
Big_mart['Item_Fat_Content'] = Big_mart.Item_Fat_Content.replace({
    "LF": "Low Fat",
    "low fat": "Low Fat",
    "reg": "Regular"
})
```

---

## 📊 Exploratory Data Analysis (EDA)

```python
sns.set()

plt.figure(figsize=(6,6))
sns.distplot(Big_mart['Item_Weight'])
plt.show()
```

(Additional plots included in the PDF.)

---

## 🔠 Label Encoding

```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

Big_mart['Item_Identifier'] = encoder.fit_transform(Big_mart['Item_Identifier'])
Big_mart['Item_Fat_Content'] = encoder.fit_transform(Big_mart['Item_Fat_Content'])
Big_mart['Item_Type'] = encoder.fit_transform(Big_mart['Item_Type'])
Big_mart['Outlet_Identifier'] = encoder.fit_transform(Big_mart['Outlet_Identifier'])
Big_mart['Outlet_Size'] = encoder.fit_transform(Big_mart['Outlet_Size'])
Big_mart['Outlet_Location_Type'] = encoder.fit_transform(Big_mart['Outlet_Location_Type'])
Big_mart['Outlet_Type'] = encoder.fit_transform(Big_mart['Outlet_Type'])
```

---

## 🧪 Train-Test Split

```python
from sklearn.model_selection import train_test_split

X = Big_mart.drop(columns='Item_Outlet_Sales',axis=1)
y = Big_mart.Item_Outlet_Sales

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
```

---

## 🤖 Model Training – XGBoost Regressor

```python
from xgboost import XGBRegressor

regressor = XGBRegressor()
regressor.fit(X_train, y_train)

X_train_pred = regressor.predict(X_train)
```

---

## 📈 Model Performance

```python
from sklearn.metrics import r2_score
r2_score(y_train, X_train_pred)
y_test_pred = regressor.predict(X_test)
r2_score(y_test, y_test_pred)
```

---

## 🏁 Conclusion

- The model performs strongly on training data.
- Moderate performance on test data — can be improved with tuning.

---


